import argparse
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet


def main():

    # parameters
    d = 2
    n = 64
    randseed = 5
    nEpochs = 100
    Ktrain = 256
    Kval = 100
    Ktest = 100

    # seed rng
    np.random.seed(randseed)

    # --- generate inequalities to make convex set ---
    print('Making data...')
    ineq = makeTestData(d, n, randseed)
    print('done.')

    # --- generate point/projected point pairs ---
    print('Making training/validation/test sets:')
    print('Training set...')
    dataTrain = makePointProjectionPairs(ineq, Ktrain)
    print('Validation set...')
    dataVal = makePointProjectionPairs(ineq, Kval)
    print('Test set...')
    dataTest = makePointProjectionPairs(ineq, Ktest)
    print('done.')
    #debugPlot(ineq, data['P'], data['Pproj'])

    # --- train network ---
    print('Constructing network...')
    model = Network(d)
    model.cuda()
    print(model)
    print('done.')

    print('Making training dataset...')
    trainDataset = ProjectionDataset(dataTrain['P'], dataTrain['Pproj'])
    print('sample points:')
    for i in range(3):
        sample = trainDataset[i]
        print(sample)
    print('done.')

    print('Making validation loader...')
    valDataset = ProjectionDataset(dataVal['P'], dataVal['Pproj'])
    print('sample points:')
    for i in range(3):
        sample = valDataset[i]
        print(sample)
    print('done.')

    print('Making optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 80])
    criterion = torch.nn.CrossEntropyLoss().cuda()
    print('done.')

    print('Training...')
    for epoch in range(nEpochs):

        # train one epoch
        print('Current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(trainDataset, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(valDataset, model, criterion)

    print('done!')
    
    # 4. evaluate network
    # TODO


def makeTestData(d, n, randseed, debugPlot=False):
    """
    Generate test data (intersection of linear inequality constraints)

    Inputs:
        - d         : ambient dimension
        - n         : number of constraints
        - randseed  : value to seed random number generator
        - debugPlot : (if d=2) generate plot of inequality constraints
    """

    # ground truth point
    x = np.random.random((d,1)) - 0.5

    # inequalities
    A = np.random.random((n,d))*2.0 - 1.0
    #A = np.array([[0.5,2.0],[0.333,3.0],[0.25,4.0]])
    b = np.random.random((n,1))*2.0 - 1.0
    #b = np.array([[1.0],[0.0],[-0.5]])
    bprime = A @ x < b

    for i in (i for i in range(n) if not bprime[i]):
        A[i,:] = -1 * A[i,:]
        b[i] = -1 * b[i]

    return {'x':x, 'A':A, 'b':b}


def makePointProjectionPairs(inequalities, K):
    
    d = inequalities['x'].shape[0]
    n = inequalities['b'].shape[0]
    P = np.random.random((d,K))*2.0 - 1.0
    Pproj = np.zeros((d,K))
    for i in range(K):
        p = P[:,i]
        pproj = p
        for k in range(n): # project onto each of n hyperplanes, n times
            for j in range(n):
                if inequalities['A'][j,:] @ pproj >= inequalities['b'][j]:
                    # project pproj onto hyperplane $$H = \left\{ x \in \R^d \colon x^T a^{(j)} = b^{(j)} \right\}$$:
                    #  $$P_H(x) = x - \frac{x^T a^{(j)} - b}{\norm{a^{(j)}}_2^2} a^{(j)}$$
                    aj = inequalities['A'][j,:]
                    pproj = pproj - 1/(np.transpose(aj)@aj)*(np.transpose(pproj)@aj-inequalities['b'][j]) * aj
                    #debugPlot(inequalities, p, pproj)
        Pproj[:,i] = pproj

    return {'P':torch.from_numpy(P), 'Pproj':torch.from_numpy(Pproj)}


def debugPlot(inequalities, P=np.nan, Pproj=np.nan, savefile="testdata.png"):

    if inequalities['x'].shape[0] is not 2:
        print('Ambient dimension must be 2 to plot.')
        return

    # plot inequalities
    A = inequalities['A']
    b = inequalities['b']
    x1plt = np.linspace(-1.0,1.0,1000)
    fig, ax = plt.subplots()
    ax.plot(inequalities['x'][0], inequalities['x'][1], 'kx')
    for i in range(b.shape[0]):
        ax.plot(x1plt, (b[i]-A[i,0]*x1plt)/A[i,1])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.grid()

    # plot point / projected point pairs
    if not np.isnan(P).any() and not np.isnan(Pproj).any():
        for i in range(P.shape[1]):
            ax.plot(P[0,i],P[1,i],'b.')
            ax.plot(Pproj[0,i],Pproj[1,i],'r.')

    fig.savefig(savefile)


class ProjectionDataset(torch.utils.data.Dataset):
    """ Dataset for point / projected point. """

    def __init__(self, P, Pproj):
        self.P = P
        self.Pproj = Pproj
        self.d = P.shape[0]
        self.K = P.shape[1]

    def __len__(self):
        return self.K

    def __getitem__(self, i):
        sample = {'p':self.P[:,i], 'pproj':self.Pproj[:,i]}
        return sample


class Network(nn.Module):
    """ Defines the simple network that will perform projections """

    def __init__(self, d):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(d,2*d)
        self.fc2 = torch.nn.Linear(2*d,4*d)
        self.fc3 = torch.nn.Linear(4*d,2*d)
        self.fc4 = torch.nn.Linear(2*d,d)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ToTensor(object):
    """ Converts numpy array to torch Tensor """

    def __call__(self, sample):
        p, pproj = sample['p'], sample['pproj']
        return {'p': torch.from_numpy(p),
                'pproj': torch.from_numpy(pproj)}


def train(trainDataset, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(len(trainDataset)):
    
        sample = trainDataset[i]
        p = sample['p']
        pproj = sample['pproj']

        print('index:')
        print(i)
        print('point:')
        print(p)
        print('projected point:')
        print(pproj)
        input_var = torch.autograd.Variable(p).cuda()
        target_var = torch.autograd.Variable(pproj).cuda()

        # compute output
        print(input_var)
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, pproj)[0]
        losses.update(loss.data[0], p.size(0))
        top1.update(prec1[0], p.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


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


if __name__ == '__main__':
    main()
