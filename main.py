import argparse
import ast
import os
from os import path


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Finite Difference Training For Federated Learning')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default="./")
parser.add_argument('--dataset', default="Cifar10", type=str, help="The dataset name")
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-c', '--client-number', default=10, type=int,
                    metavar="N", help="client number for federated learning (default: 10)")
# Model Building Parameters
parser.add_argument('--net-name', default="lenet", type=str, help="the name for network to use")
parser.add_argument('--activation', default="Hardswish", type=str, help="the activation function for network")
parser.add_argument('--normalization', default="GN", type=str,
                    help="the normalization function for network, can choose BN GN NoNorm")
parser.add_argument('--depth', default=10, type=int, metavar='D', help="the depth of neural network")
parser.add_argument('--width', default=2, type=int, metavar='W', help="the width of neural network")
parser.add_argument('--fd', '--finite-difference', action='store_false',
                    help='Use Finite Difference method and stein theory to get gradient')
parser.add_argument("--fd-format", default="forward", type=str, help="the difference format of stein's identity")
parser.add_argument('--K', '--sample-num', default=500, type=int,
                    metavar='N', help='The sample number for stein theory (default: 500)')
parser.add_argument('--sigma', default=1e-4, type=float,
                    help='Sigma for the Gaussian distribution in stein theory')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument("--fedopt", default="FedSGD", type=str,
                    help="The optimization method in federated learning, usually FedSGD or FedAvg")
parser.add_argument('--optimizer', default="Adam", type=str, metavar="Optimizer Name")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wul', '--warm-up-lr', default=0.02, type=float, help='the learning rate for warm up method')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--nesterov', action='store_true', help='nesterov in sgd')
parser.add_argument('-ad', "--adjust-lr", default=[60], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--lr-decay-ratio', default=0.2, type=float)
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
# Using EMA to mitigrate the noise
parser.add_argument('--ema', action='store_true', help='Whether We Use Exponential Moving Average')
parser.add_argument('--ed', default=0.995, help='Whether We Use Exponential Moving Average')
# GPU Parameters
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import wandb

wandb.init(project="BAFFLE{}".format(args.fedopt), entity="",
           name="Time: {}, Data: {}, Model: {}".format(args.train_time, args.dataset,
                                                              args.net_name),
           config=vars(args))

from utils.avgmeter import AverageMeter
from utils.dataloader import cifar10_dataset, cifar100_dataset, svhn_dataset, get_sl_sampler, mnist_dataset, \
    get_federated_sampler
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model.wideresnet import WideResNet
from model.lenet import LeNet
from utils.reproducibility import setup_seed
from utils.federated_utils import avg_util, create_client_weight
from torch_ema import ExponentialMovingAverage

setup_seed()


def main(args=args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # build dataset
    if args.dataset == "Cifar10":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar10_dataset(dataset_base_path)
        test_dataset = cifar10_dataset(dataset_base_path, train_flag=False)
        sampler_clients, sampler_valid = get_federated_sampler(
            torch.as_tensor(train_dataset.targets, dtype=torch.int32),
            args.client_number, 10)
        num_classes = 10
        input_channels = 3
    elif args.dataset == "Cifar100":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar100_dataset(dataset_base_path)
        test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
        sampler_clients, sampler_valid = get_federated_sampler(
            torch.as_tensor(train_dataset.targets, dtype=torch.int32),
            args.client_number, 100, 50)
        num_classes = 100
        input_channels = 3
    elif args.dataset == "SVHN":
        dataset_base_path = path.join(args.base_path, "dataset", "svhn")
        train_dataset = svhn_dataset(dataset_base_path)
        test_dataset = svhn_dataset(dataset_base_path, train_flag=False)
        sampler_clients, sampler_valid = get_federated_sampler(
            torch.as_tensor(train_dataset.labels, dtype=torch.int32),
            args.client_number, 10)
        num_classes = 10
        input_channels = 3
    elif args.dataset == "MNIST":
        dataset_base_path = path.join(args.base_path, "dataset", "mnist")
        train_dataset = mnist_dataset(dataset_base_path)
        test_dataset = mnist_dataset(dataset_base_path, train_flag=False)
        sampler_clients, sampler_valid = get_federated_sampler(
            torch.as_tensor(train_dataset.targets, dtype=torch.int32),
            args.client_number, 10)
        num_classes = 10
        input_channels = 1
    else:
        raise NotImplementedError("Dataset {} Not Implemented".format(args.dataset))
    test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                               sampler=sampler_valid)
    train_dloaders = []
    models = []
    optimizers = []
    optimizer_schedulers = []
    # set the global model
    if args.net_name == "wideresnet":
        global_model = WideResNet(num_input_channels=input_channels, depth=args.depth, width=args.width,
                                  num_classes=num_classes, K=args.K, sigma=args.sigma, activation=args.activation,
                                  normalization=args.normalization,
                                  fd_format=args.fd_format)
    elif "lenet" in args.net_name:
        if args.dataset == "Cifar100":
            channels = [32, 128]
        else:
            channels = [16, 64]
        global_model = LeNet(input_channel=input_channels, num_classes=num_classes, channels=channels, K=args.K,
                             sigma=args.sigma, activation=args.activation, normalization=args.normalization,
                             fd_format=args.fd_format)
    else:
        raise NotImplementedError("model {} not implemented".format(args.net_name))
    global_model = global_model.cuda()
    if args.fedopt == "FedSGD":
        if args.optimizer == "SGD":
            global_optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum,
                                               weight_decay=args.wd,
                                               nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                                weight_decay=args.wd)
        else:
            raise NotImplementedError("{} not find".format(args.optimizer))
        global_scheduler = MultiStepLR(global_optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
    else:
        global_optimizer = None
        global_scheduler = None
    if args.ema:
        ema = ExponentialMovingAverage(global_model.parameters(), decay=args.ed)
    else:
        ema = None
    # set local client model for each sampler
    for sampler in sampler_clients:
        if args.fedopt == "FedSGD":
            args.workers = 0
        client_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                    pin_memory=True, sampler=sampler)
        # set the client model
        if args.net_name == "wideresnet":
            client_model = WideResNet(num_input_channels=input_channels, depth=args.depth, width=args.width,
                                      num_classes=num_classes, K=args.K, sigma=args.sigma, activation=args.activation,
                                      normalization=args.normalization,
                                      fd_format=args.fd_format)
        elif "lenet" in args.net_name:
            if args.dataset == "Cifar100":
                channels = [32, 128]
            else:
                channels = [16, 64]
            client_model = LeNet(input_channel=input_channels, num_classes=num_classes, channels=channels, K=args.K,
                                 sigma=args.sigma, activation=args.activation, normalization=args.normalization,
                                 fd_format=args.fd_format)
        else:
            raise NotImplementedError("model {} not implemented".format(args.net_name))
        client_model = client_model.cuda()
        # initialize all client model with the same parameter to global model
        client_model.load_state_dict(global_model.state_dict())
        # set the optimizer of each client model
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.wd,
                                        nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
        else:
            raise NotImplementedError("{} not find".format(args.optimizer))
        scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
        train_dloaders.append(client_dloader)
        models.append(client_model)
        optimizers.append(optimizer)
        optimizer_schedulers.append(scheduler)
    federated_weight = create_client_weight(args.client_number)
    for epoch in range(args.epochs):
        if args.fedopt == "FedAvg":
            fedavg(global_model, train_dloaders, models, optimizers, optimizer_schedulers, epoch, federated_weight, ema)
        elif args.fedopt == "FedSGD":
            fedsgd(global_model, global_optimizer, global_scheduler, train_dloaders, models, epoch, federated_weight,
                   ema)
        else:
            raise NotImplementedError("Federated Optimizer {} not find".format(args.fedopt))
        # use the federated model to evaluate
        test(valid_dloader, test_dloader, model=global_model, epoch=epoch, num_classes=num_classes, ema=ema)
        # save checkpoints
        if epoch == 0 or (epoch + 1) % 10 == 0:
            if ema is not None:
                with ema.average_parameters():
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'args': args,
                        "state_dict": global_model.state_dict()
                    })
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'args': args,
                    "state_dict": global_model.state_dict()
                })


def fedavg(global_model, train_dloaders, models, optimizers, optimizer_schedulers, epoch, federated_weight, ema=None):
    def individual_train(dloader, net, opt, e, ci, estimate_grad=args.fd):
        # some records
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        net.train()
        end = time.time()
        opt.zero_grad()
        for i, (image, label) in enumerate(dloader):
            data_time.update(time.time() - end)
            image = image.float().cuda()
            label = label.long().cuda()
            if estimate_grad:
                with torch.no_grad():
                    _, loss = net(image, label, estimate_grad=True)
            else:
                _, loss = net(image, label, estimate_grad=False)
                loss.backward()
            losses.update(float(loss.item()), image.size(0))
            opt.step()
            opt.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                train_text = 'Client ID: {0}\t' \
                             'Epoch: [{1}][{2}/{3}]\t' \
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                             'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})'.format(ci,
                                                                                       e, i + 1, len(dloader),
                                                                                       batch_time=batch_time,
                                                                                       data_time=data_time,
                                                                                       cls_loss=losses)
                print(train_text)
        wandb.log({"Client {} Train".format(ci): {"cls_loss": losses.avg}}, step=e + 1)
        return losses.avg

    if epoch == 0 and args.dataset != "MNIST":
        # do warmup
        for optimizer in optimizers:
            modify_lr_rate(opt=optimizer, lr=args.wul)
    for idx, (train_dloader, model, optimizer, scheduler) in enumerate(
            zip(train_dloaders, models, optimizers, optimizer_schedulers)):
        individual_train(train_dloader, net=model, opt=optimizer, e=epoch, ci=idx,
                         estimate_grad=args.fd)
        scheduler.step()
    avg_util(model_list=models, coefficient_matrix=federated_weight)
    # update the global model, notice that all local model has been updated
    global_model.train()
    global_model.load_state_dict(models[0].state_dict())
    if ema is not None:
        ema.update()
    if epoch == 0 and args.dataset != "MNIST":
        for optimizer in optimizers:
            modify_lr_rate(opt=optimizer, lr=args.lr)


def fedsgd(global_model, global_optimizer, global_scheduler, train_dloaders, models, epoch, federated_weight, ema):
    if epoch == 0 and args.dataset != "MNIST":
        # do warmup
        modify_lr_rate(opt=global_optimizer, lr=args.wul)
    train_dloaders = [enumerate(d) for d in train_dloaders]
    end_flag = False
    losses = AverageMeter()
    global_model.train()
    for model in models:
        model.train()
    batch_id = 0
    while True:
        batch_id += 1
        if batch_id % 10 == 0:
            print("Epoch : {}, Batch : {}, Loss : {}".format(epoch, batch_id, losses.avg))
        for c_i in range(len(models)):
            dloader = train_dloaders[c_i]
            model = models[c_i]
            try:
                _, (image, label) = next(dloader)
            except StopIteration:
                end_flag = True
                break
            image = image.float().cuda()
            label = label.long().cuda()
            if args.fd:
                with torch.no_grad():
                    _, loss = model(image, label, estimate_grad=True)
            else:
                _, loss = model(image, label, estimate_grad=False)
                loss.backward()
            losses.update(float(loss.item()), image.size(0))
            # move the local model gradient to global model after each update
            # notice our setting is that we do not have batchnorm layer,
            # so we do not have to deal with running mean or running variance
            for (p_g, p_c) in zip(global_model.parameters(), model.parameters()):
                if p_g.grad is None:
                    p_g.grad = p_c.grad.detach().clone() * federated_weight[c_i]
                else:
                    p_g.grad += p_c.grad.detach().clone() * federated_weight[c_i]
            # set the local gradient to zero
            model.zero_grad()
        if end_flag:
            del train_dloaders
            break
        # in this time, the global model combined all gradients from the clients
        # we use it to perform SGD
        global_optimizer.step()
        global_optimizer.zero_grad()
        # send back all parameters to local
        for model in models:
            model.load_state_dict(global_model.state_dict())
    # after one epoch, we perform scheduler and ema
    global_scheduler.step()
    if ema is not None:
        ema.update()
    if epoch == 0 and args.dataset != "MNIST":
        # do warmup
        modify_lr_rate(opt=global_optimizer, lr=args.lr)
    print("Epoch :{} Loss:{}".format(epoch + 1, losses.avg))
    wandb.log({"Train": {"cls_loss": losses.avg}}, step=epoch + 1)


def test(valid_dloader, test_dloader, model, epoch, num_classes, ema=None):
    model.eval()
    # calculate index for valid dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    for i, (image, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            if ema is not None:
                with ema.average_parameters():
                    cls_result, loss = model(image, label, estimate_grad=False)
            else:
                cls_result, loss = model(image, label, estimate_grad=False)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    wandb.log({"Valid": {"cls_loss": losses.avg}}, step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    wandb.log({"Valid": {"top1 accuracy": top_1_accuracy}}, step=epoch + 1)
    if args.dataset == "Cifar100":
        wandb.log({"Valid": {"top1 accuracy": top_1_accuracy}}, step=epoch + 1)
        wandb.log({"Valid": {"top5 accuracy": top_5_accuracy}}, step=epoch + 1)
    # calculate index for test dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    # don't use roc
    # roc_list = []
    for i, (image, label) in enumerate(test_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            if ema is not None:
                with ema.average_parameters():
                    cls_result, loss = model(image, label, estimate_grad=False)
            else:
                cls_result, loss = model(image, label, estimate_grad=False)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    wandb.log({"Test": {"cls_loss": losses.avg}}, step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    # don't use roc auc
    # all_score = all_score.cpu().numpy()
    # all_label = all_label.cpu().numpy()
    # for i in range(num_classes):
    #     roc_list.append(roc_auc_score(all_label[:, i], all_score[:, i]))
    # ap_list.append(average_precision_score(all_label[:, i], all_score[:, i]))
    # calculate accuracy by hand
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    wandb.log({"Test": {"top1 accuracy": top_1_accuracy, }}, step=epoch + 1)
    if args.dataset == "Cifar100":
        wandb.log({"Test": {"top1 accuracy": top_1_accuracy, }}, step=epoch + 1)
        wandb.log({"Test": {"top5 accuracy": top_5_accuracy}}, step=epoch + 1)
    return top_1_accuracy


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :return:
    """
    filefolder = "{}/FL_parameter/{}/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
