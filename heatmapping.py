#!/usr/bin/python3
#
# Example run:
# ./main.py --model msdnet -b 2 -j 2 cifar10 --msd-blocks 10 --msd-base 4 --msd-step 2 \
#  --msd-stepmode even --growth 6-12-24 --gpu 0
# For evaluation / resume add: --resume --evaluate

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from utils import measure_model
from opts import args


# Init Torch/Cuda
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manual_seed)
torch.manual_seed(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_prec1 = 0


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


def msd_loss(output, target_var, criterion):
    losses = []
    for out in range(0, len(output)):
        losses.append(criterion(output[out], target_var))
    mean_loss = sum(losses) / len(output)
    return mean_loss


def msdnet_accuracy(output, target, x, val=False):
    """
    Calculates multi-classifier accuracy

    :param output: A list in the length of the number of classifiers,
                   including output tensors of size (batch, classes)
    :param target: a tensor of length batch_size, including GT
    :param x: network input input
    :param val: A flag to print per class validation accuracy
    :return: mean precision of top1 and top5
    """

    top1s = []
    top5s = []
    if torch.cuda.is_available():
        prec1 = torch.FloatTensor([0]).cuda()
        prec5 = torch.FloatTensor([0]).cuda()
    else:
        prec1 = torch.FloatTensor([0])
        prec5 = torch.FloatTensor([0])


    for out in output:
        tprec1, tprec5 = accuracy(out.data, target, topk=(1, 5))
        prec1 += tprec1
        prec5 += tprec5
        top1s.append(tprec1[0])
        top5s.append(tprec5[0])

    if val:
        for c in range(0, len(top1s)):
            print("Classifier {} top1: {} top5: {}".
              format(c, top1s[c], top5s[c]))
    prec1 = prec1 / len(output)
    prec5 = prec5 / len(output)
    return prec1, prec5, (top1s, top5s)

def load_checkpoint(args):

    if args.evaluate_from:
        print("Evaluating from model: ", args.evaluate_from)
        model_filename = args.evaluate_from
    else:
        model_dir = os.path.join(args.savedir, 'save_models')
        latest_filename = os.path.join(model_dir, 'latest.txt')
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
        else:
            return None
    print("=> loading checkpoint '{}'".format(model_filename))
    if torch.cuda.is_available():
        state = torch.load(model_filename)
    else:
        state = torch.load(model_filename, map_location=lambda storage, loc: storage)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        state['state_dict'] = new_state_dict

        # model.load_state_dict(new_state_dict)

    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def main(**kwargs):

    global args, best_prec1

    # Override if needed
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    args.maxC = 9
    imgNo, ClfrNo = 1,args.maxC-1
    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)

    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE, args.debug)

    if 'measure_only' in args and args.measure_only:
        return n_flops, n_params

    print('Starting.. FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % \
        (args.model, int(n_params), int(n_flops))
    del(model)

    # Create model
    model = getattr(models, args.model)(args)

    if args.debug:
        print(args)
        print(model)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    # Define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    checkpoint = load_checkpoint(args)
    args.start_epoch = checkpoint['epoch'] + 1
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


    cudnn.benchmark = True

    ### Data loading
    if args.data == "cifar10":
        train_set = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
        val_set = datasets.CIFAR10('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
    elif args.data == "cifar100":
        train_set = datasets.CIFAR100('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
        val_set = datasets.CIFAR100('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_per_cls = [AverageMeter() for i in range(0, model.num_blocks)]
    top5_per_cls = [AverageMeter() for i in range(0, model.num_blocks)]

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        (inputM, targetM) = (input, target)
        # import pdb
        # pdb.set_trace()
        
        input, target = input[imgNo], target[imgNo].view(1)

        input = input.view(1,input.shape[0], input.shape[1],input.shape[2])

        if torch.cuda.is_available():
            target = target.cuda(async=True)
        
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        # ### Compute output

        # scores, feats = model(input_var, 0.0, p=1)

        # if args.model == 'msdnet':
        #     loss = msd_loss(scores, target_var, criterion)
        # else:
        #     loss = criterion(scores, target_var)

        # loss.backward(create_graph=True)
        # grads = model.gradients
        # print(len(output))
        # print(len(output[0]))
        # # print(target)
        # # print(args.batch_size)
        # finalProb = []
        # for j in range(10):
        #     clfrProbs = []
        #     for i in range(args.batch_size):
        #         clfrProbs.append(output[j][i][target[i]])#[i]
        #         out = probab[j][i][target[i]]
        #         print(out) 
        #     finalProb.append(clfrProbs)


    ####################### Business ############################################
        from gradCam import *
        # print(len(output), len(grads), len(feats))
        grad_cam = GradCam(model = model)
        # print(model.gradients)
        # print("--")
        # mask = grad_cam(None, features=feats[ClfrNo][0], scores=scores[ClfrNo][0])#target_index)
        mask = grad_cam(None, input_var, 0, ClfrNo)#target_index)

        img = input[0].cpu().data.numpy().transpose(1,2,0)
        img = cv2.resize(img, (256, 256))
        show_cam_on_image(img, mask)

        gb_model = GuidedBackpropReLUModel(model)
        gb = gb_model(None, input_var, 0, ClfrNo).transpose(2,0,1)
        # import pdb
        # pdb.set_trace()
        utils.save_image(torch.from_numpy(gb), 'gb.jpg')

        cam_mask = np.zeros(gb.shape)
        for i in range(0, gb.shape[0]):
            cam_mask[i, :, :] = mask

        cam_gb = np.multiply(cam_mask, gb)
        utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
        img = cv2.resize(input.cpu().data.numpy()[0].transpose(1,2,0), (256, 256))
        # import pdb
        # pdb.set_trace()

        utils.save_image(torch.from_numpy(img.transpose(2,0,1)), 'input.jpg')
        exit()
    #############################################################################




        ### Measure accuracy and record loss
    #     if hasattr(output, 'data'):
    #         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    #     elif args.model == 'msdnet':
    #         prec1, prec5, _ = msdnet_accuracy(output, target, input)
    #     losses.update(loss.data[0], input.size(0))
    #     top1.update(prec1[0], input.size(0))
    #     top5.update(prec5[0], input.size(0))

    #     ### Measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     if i % args.print_freq == 0:
    #         print('Test: [{0}/{1}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                   i, len(val_loader), batch_time=batch_time, loss=losses,
    #                   top1=top1, top5=top5))
        
    #     _, _, (ttop1s, ttop5s) = msdnet_accuracy(output, target, input,
    #                                          val=True)
    #     for c in range(0,len(top1_per_cls)):
    #         top1_per_cls[c].update(ttop1s[c], input.size(0))
    #         top5_per_cls[c].update(ttop5s[c], input.size(0))

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    # for c in range(0, len(top1_per_cls)):
    #     print(' * For class {cls}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #               .format(cls=c,top1=top1_per_cls[c], top5=top5_per_cls[c]))
    # return 100. - top1.avg, 100. - top5.avg


if __name__ == '__main__':
    main()