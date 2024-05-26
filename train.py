import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from test import test_and_save
from data import prepare_clothing1m_data
from models import Resnet50
from utils import write_accuracy


def train_one_epoch(net, dataloader, optimizer, scheduler, device, epoch, writer):
    net.train()
    global_iteration = len(dataloader) * epoch
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
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
        clean_labels = clean_labels.to(device)
        inputs = inputs.to(device)

        #########################################################################################
        # Main Step
        #########################################################################################

        pred, pred1, pred2, model_features = net(inputs, return_features=True)

        loss = F.cross_entropy(pred, labels)
        loss += F.cross_entropy(pred1, labels)
        loss += F.cross_entropy(pred2, labels)
        loss /= 3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = write_accuracy(writer, pred, clean_labels, "Accuracy/pseudo_labels", global_iteration)
        pbar.set_description('e: {}, loss_cat: {:.4f}, acc_cat: {:.0%} LR: {}'.format(epoch,
                                                                                      round(loss.item(), 2),
                                                                                      round(acc, 2),
                                                                                      round(lr, 6)
                                                                                      )
                             )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fmlc_clothing1m")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nested", type=float, default=0)
    parser.add_argument("--dampening", type=float, default=0.0)
    parser.add_argument("--nesterov", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--log-dir", type=str, default="outputs/")
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)

    val_dataloader, _, test_dataloader = prepare_clothing1m_data(args)

    net = Resnet50(features_size=2048, out_size=14).to(args.device)
    if args.pretrained:
        print("load model pretrained weight : ", args.pretrained)
        net.load_state_dict(torch.load(args.pretrained))
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch * len(val_dataloader),
                                                           eta_min=0.0001)
    best_acc = 0
    for epoch in range(args.max_epoch):
        train_one_epoch(net, val_dataloader, optimizer, scheduler, args.device, epoch, writer)
        print('Computing Test Result...')
        net.eval()
        test_and_save(net, val_dataloader, args.device, writer, data_name='val', epoch=epoch)
        test_and_save(net, test_dataloader, args.device, writer, data_name='test', epoch=epoch)