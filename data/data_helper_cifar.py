import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from .load_corrupted_data_mlg import CIFAR10, CIFAR100


def prepare_data(gold_fraction, corruption_prob, args):
    # Following ideas from Nishi et. al
    transform_silver = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_gold = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    args.num_meta = int(50000 * gold_fraction)

    if args.dataset == 'cifar10':
        num_classes = 10

        train_data_gold = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_gold, download=True)

        train_data_silver = CIFAR10(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_silver, download=True, seed=args.seed)

        test_data = CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)

    elif args.dataset == 'cifar100':
        num_classes = 100

        train_data_gold = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_gold, download=True)

        train_data_silver = CIFAR100(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_silver, download=True, seed=args.seed)

        test_data = CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)

    test_sampler = None
    test_batch_size = args.batch_size
    eff_batch_size = args.batch_size

    train_gold_loader = DataLoader(train_data_gold, batch_size=eff_batch_size, drop_last=True, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)

    train_silver_loader = DataLoader(train_data_silver, batch_size=eff_batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.num_workers, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=(test_sampler is None),
                             num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

    return train_gold_loader, train_silver_loader, test_loader, num_classes
