#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from math import log10
from torch.autograd import Variable
from torch.utils.data import DataLoader

import bgs_model
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Fast Matting Portrait')
parser.add_argument(
    '--batchSize', type=int, default=64, help='training batch size')
parser.add_argument(
    '--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument(
    '--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument(
    '--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument(
    '--cuda', action='store_true', help='use cuda?')
parser.add_argument(
    '--threads',
    type=int,
    default=4,
    help='number of threads for data loader to use')
parser.add_argument(
    '--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
training_data_loader = DataLoader(
    dataset=train_set,
    num_workers=opt.threads,
    batch_size=opt.batchSize,
    shuffle=True)
test_set = get_test_set()
testing_data_loader = DataLoader(
    dataset=test_set,
    num_workers=opt.threads,
    batch_size=opt.testBatchSize,
    shuffle=False)

print('===> Building model')
model = bgs_model.MattNet()
criterion = nn.MSELoss()
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch):
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input_ = Variable(batch[0]).float()
        target = Variable(batch[1]).float()
        targetQ = Variable(batch[2]).float()

        if cuda:
            input_ = input_.cuda()
            target = target.cuda()
            targetQ = targetQ.cuda()
        input_ = input_.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
        targetQ = targetQ.permute(0, 3, 1, 2)

        optimizer.zero_grad()
        prediction = model(input_)
        q_prediction = prediction * input_
        loss = torch.mean(
            (q_prediction - targetQ) ** 2 + (prediction - target) ** 2)
        # loss = criterion(prediction, target)
        epoch_loss += loss.data.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss.data.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))

def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input_, target = Variable(batch[0]).float(), Variable(batch[1]).float()
        if cuda:
            input_ = input_.cuda()
            target = target.cuda()
        input_ = input_.permute(0, 3, 1, 2)

        prediction = model(input_)
        mse = criterion(prediction, target)
        print("Testing loss: {:.4f}".format(mse.data.item()))
        psnr = 10 * log10(1 / mse.data.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

for epoch in range(opt.nEpochs):
    print('===> Training model')
    train(epoch + 1)
    print('===> Testing model')
    # test()
    print('===> Checkpointing')
    checkpoint(epoch)
