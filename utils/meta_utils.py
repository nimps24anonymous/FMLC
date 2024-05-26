import torch
import torch.nn.functional as F
def replace_none_with_zero(tensor_list, reference):
    out = []
    for t, r in zip(tensor_list, reference):
        fixed = t if t is not None else torch.zeros_like(r)
        out.append(fixed)
    return tuple(out)


def to_vec(tensor_list, alpha=1.0):
    return torch.cat([alpha * t.reshape(-1) for t in tensor_list])

def set_grads(params, grads):
    """
    Set gradients for trainable parameters. ``params.grad = grads``

    :param params: Trainable parameters
    :type params: Sequence of Tensor
    :param grads: Calculated gradient
    :type grads: Sequence of Tensor
    """
    for param, grad in zip(params, grads):
        if grad is not None:
            if False or hasattr(param, "grad") and param.grad is not None:
                param.grad = param.grad + grad
            else:
                param.grad = grad

class MetaStep():
    def __init__(self, meta_net, main_heads_net, writer, r=0.01):
        self.r = r
        self.main_heads_net = main_heads_net
        self.meta_net = meta_net
        self.writer = writer

    def get_vector(self, heads_inputs, meta_labels, global_iteration):
        meta_pred_1, meta_pred_2, meta_pred_3 = self.main_heads_net(heads_inputs)

        clean_loss = F.cross_entropy(meta_pred_1, meta_labels)
        clean_loss += F.cross_entropy(meta_pred_2, meta_labels)
        clean_loss += F.cross_entropy(meta_pred_3, meta_labels)
        clean_loss /= 3

        vector = torch.autograd.grad(clean_loss.mean(), self.main_heads_net.parameters(), allow_unused=False)
        vector = replace_none_with_zero(vector, self.main_heads_net.parameters())
        eps = self.r / to_vec(vector).norm().add_(1e-12).item()

        if self.writer is not None:
            _, hard_output = torch.max(meta_pred_1, dim=-1)
            meta_acc = (hard_output.eq(meta_labels).float().sum() / max(len(meta_labels), 1))
            self.writer.add_scalar("Accuracy/meta", meta_acc.item(), global_iteration)
            self.writer.add_scalar('Loss/meta', clean_loss.item(), global_iteration)
        return vector, eps

    def finite_grad(self, pseudo_labels, corrected_labels, labels, weights, features, vector, eps):
        # positive perturbation
        with torch.no_grad():
            for param, v in zip(self.main_heads_net.parameters(), vector):
                if param.requires_grad:
                    param.data.add_(v.data, alpha=eps)
            pred_1, pred_2, pred_3 = self.main_heads_net(features)
        loss_p = F.cross_entropy(pred_1, pseudo_labels)
        loss_p += F.cross_entropy(pred_2, corrected_labels)
        loss_p += torch.mean(F.cross_entropy(pred_3, labels, reduction='none') * weights.squeeze())
        loss_p /= 3
        grad_p = torch.autograd.grad(loss_p, self.meta_net.parameters(), retain_graph=True, allow_unused=True)
        grad_p = replace_none_with_zero(grad_p, self.meta_net.parameters())

        # negative perturbation
        with torch.no_grad():
            for param, v in zip(self.main_heads_net.parameters(), vector):
                if param.requires_grad:
                    param.data.add_(v.data, alpha=-2 * eps)
            pred_1, pred_2, pred_3 = self.main_heads_net(features)
        loss_n = F.cross_entropy(pred_1, pseudo_labels)
        loss_n += F.cross_entropy(pred_2, corrected_labels)
        loss_n += torch.mean(F.cross_entropy(pred_3, labels, reduction='none') * weights.squeeze())
        loss_n /= 3
        grad_n = torch.autograd.grad(loss_n, self.meta_net.parameters(), retain_graph=False, allow_unused=True)
        grad_n = replace_none_with_zero(grad_n, self.meta_net.parameters())

        # reverse weight change
        with torch.no_grad():
            # u = True
            for param, v in zip(self.main_heads_net.parameters(), vector):
                if param.requires_grad:
                    param.data.add_(v.data, alpha=eps)

        return  [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]


    def __call__(self, pseudo_labels, corrected_labels, labels, weights, features,meta_heads_inputs, meta_labels,
                 global_iteration, *args, **kwargs):

        vector, eps = self.get_vector(meta_heads_inputs, meta_labels, global_iteration)
        return self.finite_grad(pseudo_labels, corrected_labels, labels, weights, features, vector, eps)



