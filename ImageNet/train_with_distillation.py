import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gc

import models.MobileNet as Mov
import models.ResNet as ResNet
import distiller

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet-1k Training')
parser.add_argument('--data_path', type=str, help='path to dataset')
parser.add_argument('--net_type', default='resnet', type=str, help='networktype: resnet, mobilenet')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=500, type=int, help='print frequency (default: 500)')

best_err1 = 100
best_err5 = 100

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize])
                                         )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])
                                       )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if args.net_type == 'mobilenet':
        t_net = ResNet.resnet50(pretrained=True)
        s_net = Mov.MobileNet()
    elif args.net_type == 'resnet':
        t_net = ResNet.resnet152(pretrained=True)
        s_net = ResNet.resnet50(pretrained=False)
    else:
        print('undefined network type !!!')
        raise

    d_net = distiller.Distiller(t_net, s_net)

    print ('Teacher Net: ')
    print(t_net)
    print ('Student Net: ')
    print(s_net)
    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

    t_net = torch.nn.DataParallel(t_net).cuda()
    s_net = torch.nn.DataParallel(s_net).cuda()
    d_net = torch.nn.DataParallel(d_net).cuda()

    # define loss function (criterion) and optimizer
    criterion_CE = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(list(s_net.parameters()) + list(d_net.module.Connectors.parameters()), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    print('Teacher network performance')
    validate(val_loader, t_net, criterion_CE, 0)

    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch)
        # evaluate on validation set
        err1, err5 = validate(val_loader, s_net, criterion_CE, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        print ('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': s_net.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        gc.collect()

    print ('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
 

def validate(val_loader, model, criterion_CE, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)

        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion_CE(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test (on val set): [Epoch {0}/{1}][Batch {2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
          .format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg

def train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)

        loss_CE = criterion_CE(outputs, targets)
        loss = loss_CE + loss_distill.sum() / batch_size / 10000

        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train with distillation: [Epoch %d/%d][Batch %d/%d]\t Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, top1.avg, top5.avg))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.net_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.net_type) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

if __name__ == '__main__':
    main()
