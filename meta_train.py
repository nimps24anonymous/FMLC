import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import write_weights, write_accuracy, set_grads

def meta_train_one_epoch(net, meta_net, optimizer, meta_optimizer, meta_step, noisy_dataloader, meta_dataloader,
                        scheduler, meta_scheduler, epoch, device, writer):
    global_iteration = len(noisy_dataloader) * epoch
    pbar = enumerate(noisy_dataloader)
    pbar = tqdm(pbar, total=len(noisy_dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for iteration, d in pbar:
        # break
        global_iteration += 1
        lr = optimizer.param_groups[0]['lr']

        if len(d)==3:
            inputs, labels, clean_labels = d
        else:
            inputs, labels = d
            clean_labels = labels
        labels = labels.to(device)
        inputs = inputs.to(device)
        clean_labels = clean_labels.to(device)

        #########################################################################################
        # Main Step
        #########################################################################################

        pred, pred1, pred2, model_features = net(inputs, return_features=True)
        #
        pseudo_labels, weights = meta_net(model_features.detach(), labels)
        corrected_labels = (1 - weights) * pseudo_labels + weights * F.one_hot(labels, 14)

        loss = F.cross_entropy(pred, pseudo_labels.detach())
        loss += F.cross_entropy(pred1, corrected_labels.detach())
        loss += torch.mean(F.cross_entropy(pred2, labels, reduction='none') * weights.squeeze().detach())
        loss /= 3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None:
            acc = write_accuracy(writer, pseudo_labels, clean_labels, "Accuracy/pseudo_labels", global_iteration)
            write_accuracy(writer, corrected_labels, clean_labels, "Accuracy/corrected_labels", global_iteration)
            write_weights(writer, weights, global_iteration)
            writer.add_scalar('Loss/train', loss.item(), global_iteration)



        #########################################################################################
        # Meta Step
        #########################################################################################
        try:
            meta_inputs,  meta_labels = next(data_iterator)
        except:
            V = 0
            data_iterator = iter(meta_dataloader)
            meta_inputs, meta_labels = next(data_iterator)

        meta_inputs, meta_labels = meta_inputs.to(device), meta_labels.to(device)
        with torch.no_grad():
            _, _, _, meta_heads_inputs = net(meta_inputs, return_features=True)

        implicit_grad = meta_step(pseudo_labels, corrected_labels, labels, weights, model_features.detach(),
                                  meta_heads_inputs,meta_labels, global_iteration)

        # Update meta net
        meta_optimizer.zero_grad()
        set_grads(meta_net.parameters(), implicit_grad)
        meta_optimizer.step()

        #########################################################################################################
        #########################################################################################################
        #########################################################################################################

        scheduler.step()
        meta_scheduler.step()

        pbar.set_description('e: {}, loss_cat: {:.4f}, acc_cat: {:.0%} LR: {}'.format(epoch,
                                                                                      round(loss.item(), 2),
                                                                                      round(acc, 2),
                                                                                      round(lr, 6)
                                                                                      )
                             )
