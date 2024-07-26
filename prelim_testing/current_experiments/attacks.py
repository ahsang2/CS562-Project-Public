# import torchattacks
import torch
from tqdm import tqdm

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (preds == true.data).sum().float()/float(true.size(0))
    return accuracy.item()

def pgd(X, target, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - target).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])

        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

    return X_adv

def adv_loss(input, target, model,eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None, attack='pgd'):
    if attack == 'pgd':
        X_adv = pgd(input, target, model, eps=eps, step_size=step_size, iters=iters, clamp_min=clamp_min, clamp_max=clamp_max, mask=mask)
    else:
        raise NotImplementedError
    return X_adv