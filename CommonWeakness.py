import torch
from .optim import OptimAttacker
from torch.optim import Optimizer, Adam


def cosine_similarity(x: list):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x, dim=0)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=0).to(torch.bool)  # 只取上三角
    similarity = similarity[mask]
    return torch.mean(similarity).item()


class AdamCSE(OptimAttacker):
    '''
    Adam CSE. For more detail, refer to
    https://arxiv.org/abs/2303.09105
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty',
                 out_optimizer=Adam,
                 outer_lr=0.05):
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)
        self.outer_lr = outer_lr
        self.out_optimizer = out_optimizer

    @property
    def param_groups(self):
        return self.out_optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        if self.out_optimizer is not None:
            print(f'set outer optimizer is {self.out_optimizer}')
            print('-' * 100)
            self.out_optimizer = self.out_optimizer([self.optimizer.param_groups[0]['params'][0]], self.outer_lr)
        if self.detector_attacker.vlogger is not None:
            self.detector_attacker.vlogger.optimizer = self.out_optimizer

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            # print('confs', confs)
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            # print(loss)
            loss.backward()
            self.grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @torch.no_grad()
    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = self.optimizer.param_groups[0]['params'][0]
        if self.out_optimizer is None:
            patch.mul_(self.outer_lr)
            patch.add_((1 - self.outer_lr) * self.original_patch)
            self.original_patch = None
        else:
            fake_grad = - self.outer_lr * (patch - self.original_patch)
            self.out_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original_patch)
            patch.grad = fake_grad
            self.out_optimizer.step()

        grad_similarity = cosine_similarity(self.grad_record)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')
        del self.grad_record
