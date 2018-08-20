import os
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.datasets as datasets

import resnet
import util


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
    trainDataset = ProjectionDataset(dataTrain['P'], dataTrain['Pproj'])
    print('Validation set...')
    dataVal = makePointProjectionPairs(ineq, Kval)
    valDataset = ProjectionDataset(dataVal['P'], dataVal['Pproj'])
    print('Test set...')
    dataTest = makePointProjectionPairs(ineq, Ktest)
    testDataset = ProjectionDataset(dataTest['P'], dataVal['Pproj'])
    print('done.')
    #debugPlot(ineq, data['P'], data['Pproj'])

    # --- train network ---
    print('Constructing network...')
    model = Network(d)
    model.cuda()
    print(model)
    print('done.')

    print('Making optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 80])
    criterion = torch.nn.SmoothL1Loss().cuda() # huber loss
    print('done.')

    print('Training...')
    for epoch in range(nEpochs):

        # train one epoch
        currentLR = optimizer.param_groups[0]['lr']
        train(trainDataset, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        avgValLoss = validate(valDataset, model, criterion)
        print('Epoch {0:d}/{1:d}\tlr = {2:.5e}\tmean l2 err = {3:.5f}'.format(
            epoch+1, nEpochs, currentLR, avgValLoss))

    print('Training ({0:d} epochs) complete!'.format(nEpochs))

    # --- save results on training/eval set ---
    print('Saving results...')
    saveTestResults(trainDataset, model, 'results_train.mat')
    saveTestResults(valDataset, model, 'results_val.mat')
    saveTestResults(testdataset, model, 'results_test.mat')
    print('done!')


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
    """
        Defines the simple network that will perform projections
    """

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


def train(trainDataset, model, criterion, optimizer):
    """
        Run one train epoch
    """
    lossAvg = util.Average()

    # switch to train mode
    model.train()

    for i in range(len(trainDataset)):
    
        # get training sample
        sample = trainDataset[i]
        p = sample['p']
        pproj = sample['pproj']

        # send to gpu
        input_var = torch.autograd.Variable(p).cuda().float()
        target_var = torch.autograd.Variable(pproj).cuda().float()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        lossAvg.update(loss.data.item())


def validate(valDataset, model, criterion):
    """
        Run evaluation on validation set
    """
    l2err_avg = util.Average()
    total_l2err = 0.0

    # switch to evaluate mode
    model.eval()

    for i in range(len(valDataset)):

        # get validation sample
        sample = valDataset[i]
        p = sample['p']
        pproj = sample['pproj']
        p_input = torch.autograd.Variable(p).cuda().float()

        # compute output
        pproj_hat = model(p_input).cpu()
        sample_l2err = np.linalg.norm(pproj.numpy() - pproj_hat.detach().numpy())
        total_l2err += sample_l2err

    avg_l2err = total_l2err / len(valDataset)

    return avg_l2err

def saveTestResults(dataset, model, filename):
    """
        Save model outputs to a mat file
    """

    model.eval()
    sample = dataset[0]
    d = sample['p'].shape[0]
    P         = np.zeros(d,len(dataset))
    Pproj     = np.zeros(d,len(dataset))
    Pproj_hat = np.zeros(d,len(dataset))
    errs      = np.zeros(len(dataset),1)

    for i in range(len(dataset)):

        # get sample
        sample = dataset[i]
        p = sample['p']
        pproj = sample['pproj']
        p_input = torch.autograd.Variable(p).cuda().float()

        # compute output
        pproj_hat = model(p_input).cpu()
        l2err = np.linalg.norm(pproj.numpy() - pproj_hat.detach().numpy())

        # store
        P[:,i] = p
        Pproj[:,i] = pproj
        Pproj_hat[:,i] = pproj_hat
        errs[i] = l2err

    # save
    scipy.io.savemat(filename, {'P':P, 'Pproj':Pproj, 'Pproj_hat':Pproj_hat, 'errs':errs})


if __name__ == '__main__':
    main()
