import torch
from .meta_utils import MetaStep, set_grads
import torch.nn as nn
from tqdm import tqdm
def write_accuracy(writer, predictions, labels, name, global_iteration):
    proba, hard_pseudo = torch.max(predictions, dim=-1)
    acc = (hard_pseudo.eq(labels).float().sum() / max(len(labels), 1))
    writer.add_scalar(name, acc.item(), global_iteration)
    return acc.item()

def write_weights(writer, weights, global_iteration):
    writer.add_scalar('Weights/mean', weights.mean().item(), global_iteration)
    writer.add_scalar('Weights/min', weights.min().item(), global_iteration)
    writer.add_scalar('Weights/max', weights.max().item(), global_iteration)


def compute_loss_accuracy(net, data_loader, device):
    # net.eval()
    losses = {}
    accs = {}
    # with torch.no_grad():
    criterion = nn.CrossEntropyLoss(reduction='none').to(device=device)
    for batch_idx, d in enumerate(tqdm(data_loader)):
        if len(d)==3:
            inputs, _, labels = d
        else:
            inputs, labels = d
        labels = labels.to(device)

        with torch.no_grad():
            outputs = net(inputs.to(device))
        for head, output in enumerate(outputs):
            if head not in losses:
                losses[head] = []
                accs[head] = []
            l = criterion(output, labels)
            losses[head] += l

            _, hard_pred = torch.max(output, dim=-1)
            accs[head] += hard_pred.eq(labels).float()


    # normalise:
    for head in losses:
        losses[head] = torch.mean(torch.stack(losses[head])).item()
        accs[head] = torch.mean(torch.stack(accs[head])).item()
    return losses, accs



