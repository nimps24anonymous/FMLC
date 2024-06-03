import torch
import torchvision
import torchvision.transforms as transforms

def prepare_clothing1m_data(args):
    num_classes = 14
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    interpolation = transforms.InterpolationMode.BICUBIC
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    train_data_gold = torchvision.datasets.ImageFolder(args.data_path + '/clean_train', transform=transform_train)
    train_data_silver = torchvision.datasets.ImageFolder(args.data_path + '/noisy_train', transform=transform_train)
    test_data = torchvision.datasets.ImageFolder(args.data_path + '/clean_test', transform=transform_val)
    batch_size = args.batch_size

    train_gold_loader = torch.utils.data.DataLoader(train_data_gold, batch_size=batch_size, shuffle=True,
                                                      num_workers=args.num_workers, pin_memory=True)

    train_silver_loader = torch.utils.data.DataLoader(train_data_silver, batch_size=batch_size,
                                                      shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size*3, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    return train_gold_loader, train_silver_loader, test_loader
