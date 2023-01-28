from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch


def cifar10_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    dataset = datasets.CIFAR10(root=dataset_base_path, train=train_flag,
                               download=True, transform=transform)
    return dataset


def mnist_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    dataset = datasets.MNIST(root=dataset_base_path, train=train_flag, download=True, transform=transform)
    return dataset


def cifar100_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    dataset = datasets.CIFAR100(root=dataset_base_path, train=train_flag,
                                download=True,transform=transform)
    return dataset


def svhn_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transform
        ])
    if train_flag:
        dataset = datasets.SVHN(root=dataset_base_path, split='train', transform=transform, download=True)
    else:
        dataset = datasets.SVHN(root=dataset_base_path, split='test', transform=transform, download=True)
    return dataset


def get_sl_sampler(labels, valid_num_per_class, num_classes):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_valid, sampler_train


# build federated learning dataset

def get_federated_sampler(labels, num_client, num_classes, validation_num=500):
    sampler_clients = [[] for i in range(num_client)]
    sampler_valid = []
    # n clients + 1 validation
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        loc = loc[torch.randperm(loc.size(0))]
        num_per_sampler = int((loc.size(0) - validation_num) / num_client)
        for k in range(num_client):
            begin = k * num_per_sampler
            end = (k + 1) * num_per_sampler
            sampler_clients[k].extend(loc[begin:end].tolist())
        sampler_valid.extend(loc[-validation_num:].tolist())
    sampler_clients = [SubsetRandomSampler(s) for s in sampler_clients]
    sampler_valid = SubsetRandomSampler(sampler_valid)
    return sampler_clients, sampler_valid


if __name__ == "__main__":
    from glob import glob
    from torch.utils.data import DataLoader
