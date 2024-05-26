import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from train import train_one_epoch
from meta_train import meta_train_one_epoch
from test import test_and_save
from data import prepare_clothing1m_data
from models import Resnet50, MetaNet, HeadsNet
from utils import MetaStep
parser = argparse.ArgumentParser(description="fmlc_clothing1m")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_epoch", type=int, default=4)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--finetune_lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--nested", type=float, default=0)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=5e-4)

parser.add_argument("--meta-lr", type=float, default=2e-4)
parser.add_argument("--meta-weight-decay", type=float, default=5e-4)
parser.add_argument("--temperature", type=float, default=1)

parser.add_argument("--data-path", type=str, default="")
parser.add_argument("--log-dir", type=str, default="outputs/clothing1M/")
parser.add_argument('--meta_pretrained', type=str, default="")
parser.add_argument('--pretrained', type=str, default="")
parser.add_argument('--resume', type=bool, default=False)

args = parser.parse_args()

writer = SummaryWriter(log_dir=args.log_dir)

val_dataloader, noisy_dataloader, test_dataloader = prepare_clothing1m_data(args)

net = Resnet50(features_size=2048, out_size=14).to(args.device)
meta_net = MetaNet(features_size=2048, embed_dim=128, out_size=14).to(args.device)
main_heads_net = HeadsNet(net.head, net.head2, net.head3)

meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening,
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
meta_step = MetaStep(meta_net, main_heads_net, writer)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch * len(noisy_dataloader), eta_min=0.0001)
meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=args.max_epoch * len(noisy_dataloader), eta_min=1e-6)

criterion = nn.CrossEntropyLoss(reduction='none').to(device=args.device)
best_acc = 0

for epoch in range(args.max_epoch):
    meta_train_one_epoch(net, meta_net, optimizer, meta_optimizer, meta_step, noisy_dataloader, val_dataloader,
                        scheduler, meta_scheduler, epoch, args.device, writer)

    print('Computing Test Result...')
    net.eval()
    test_and_save(net, val_dataloader, args.device, writer, data_name='val', epoch=epoch)
    _, acc = test_and_save(net, test_dataloader, args.device, writer, data_name='test', epoch=epoch)
    torch.save(net.state_dict(), args.log_dir + "last.pth")
    torch.save(meta_net.state_dict(), args.log_dir + "meta_last.pth")
    if acc[1] > best_acc:
        print("best_acc", acc[1])
        best_acc = acc[1]
        torch.save(net.state_dict(), args.log_dir + "best.pth")
        torch.save(meta_net.state_dict(), args.log_dir + "meta_best.pth")
    net.train()

##########################################
# Finetune
##########################################
optimizer = torch.optim.SGD(net.parameters(), lr=args.finetune_lr, momentum=args.momentum, dampening=args.dampening,
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(noisy_dataloader), eta_min=0.0001)
train_one_epoch(net, val_dataloader, optimizer, scheduler, args.device, epoch+1, writer)
print('Computing Test Result...')
net.eval()
test_and_save(net, val_dataloader, args.device, writer, data_name='val', epoch=epoch+1)
test_and_save(net, test_dataloader, args.device, writer, data_name='test', epoch=epoch+1)
