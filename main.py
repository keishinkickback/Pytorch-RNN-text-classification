from __future__ import print_function

import time
import gc
import os
import argparse

import numpy as np
from sklearn.externals import  joblib
import torch
from torch import nn

from vocab import  VocabBuilder
from dataloader import TextClassDataLoader
from model import RNN
from util import AverageMeter, accuracy
from util import adjust_learning_rate

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--embedding-size', default=128, type=int,
                    metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=32, type=int,
                    metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=1, type=int,
                    metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=8, type=int,
                    metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=2, type=int,
                    metavar='N', help='min number of tokens')

args = parser.parse_args()


# create vocab
print("===> creating vocabs ...")
end = time.time()
v_builder = VocabBuilder(path_file='data/train.tsv', min_sample=args.min_samples)
d_word_index = v_builder.get_word_index()
vocab_size = len(d_word_index)

if not os.path.exists('gen'):
    os.mkdir('gen')

joblib.dump(d_word_index, 'gen/d_word_index.pkl', compress=3)
print('===> vocab creatin: {t:.3f}'.format(t=time.time()-end))


# create trainer
print("===> creating dataloaders ...")
end = time.time()
train_loader = TextClassDataLoader('data/train.tsv', d_word_index, batch_size=args.batch_size)
val_loader = TextClassDataLoader('data/test.tsv', d_word_index, batch_size=args.batch_size)
print('===> dataloader creatin: {t:.3f}'.format(t=time.time()-end))


# create model
print("===> creating rnn model ...")
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size,
            num_output=args.classes, hidden_size=args.hidden_size,
            num_layers=args.layers, batch_first=True)
print(model)


# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
print(optimizer)
print(criterion)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i!= 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            gc.collect()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


# training and testing
for epoch in range(1, args.epochs+1):

    adjust_learning_rate(args.lr, optimizer, epoch)
    train(train_loader, model, criterion, optimizer, epoch)
    test(val_loader, model, criterion)

    # save current model
    name_model = 'rnn_{}.pkl'.format(epoch)
    path_save_model = os.path.join('gen', name_model)
    joblib.dump(model, path_save_model, compress=3)

