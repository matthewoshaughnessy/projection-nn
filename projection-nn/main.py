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

import util
import linearIneqTestData


args = {'d'        : 2,
        'nIneq'    : 16,
        'randseed' : 5,
        'nEpochs'  : 250,
        'Ktrain'   : 1024,
        'Kval'     : 100,
        'Ktest'    : 100,
        'useCuda'  : False}


def main():

    # parameters
    args['useCuda'] = torch.cuda.is_available()
    print('CUDA enabled: {0:}'.format(args['useCuda']))

    # seed rng
    np.random.seed(args['randseed'])

    # --- generate inequalities to make convex set ---
    print('Making data...')
    ineq = linearIneqTestData.makeRandomData(args['d'], args['nIneq'], args['randseed'])
    print('done.')

    # --- generate point/projected point pairs ---
    print('Making training/validation/test sets:')
    print('Training set...')
    dataTrain = linearIneqTestData.makePointProjectionPairs(ineq, args['Ktrain'])
    trainDataset = ProjectionDataset(dataTrain['P'], dataTrain['Pproj'])
    print('Validation set...')
    dataVal = linearIneqTestData.makePointProjectionPairs(ineq, args['Kval'])
    valDataset = ProjectionDataset(dataVal['P'], dataVal['Pproj'])
    print('Test set...')
    dataTest = linearIneqTestData.makePointProjectionPairs(ineq, args['Ktest'])
    testDataset = ProjectionDataset(dataTest['P'], dataVal['Pproj'])
    print('done.')
    #linearIneqTestData.plot(ineq, data['P'], data['Pproj'])

    # --- train network ---
    print('Constructing network...')
    model = Network()
    if args['useCuda']:
        model.cuda()
    print(model)
    print('done.')

    print('Making optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[150, 200])
    if args['useCuda']:
        criterion = torch.nn.SmoothL1Loss().cuda() # huber loss
    else:
        criterion = torch.nn.SmoothL1Loss()
    print('done.')

    print('Training...')
    for epoch in range(args['nEpochs']):

        # train one epoch
        currentLR = optimizer.param_groups[0]['lr']
        train(trainDataset, model, criterion, optimizer)
        lr_scheduler.step()

        # evaluate on validation set
        avgValLoss = validate(valDataset, model, criterion)
        print('Epoch {0:d}/{1:d}\tlr = {2:.5e}\tmean l2 err = {3:.7f}'.format(
            epoch+1, args['nEpochs'], currentLR, avgValLoss))

    print('Training ({0:d} epochs) complete!'.format(args['nEpochs']))

    # --- save results on training/eval set ---
    print('Saving results...')
    saveTestResults(trainDataset, model, 'results_train.mat')
    saveTestResults(valDataset, model, 'results_val.mat')
    saveTestResults(testDataset, model, 'results_test.mat')
    print('done!')


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

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(args['d'],2*args['d'])
        self.fc2 = torch.nn.Linear(2*args['d'],8*args['d'])
        self.fc3 = torch.nn.Linear(8*args['d'],4*args['d'])
        self.fc4 = torch.nn.Linear(4*args['d'],args['d'])

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
        if args['useCuda']:
            input_var = torch.autograd.Variable(p).cuda().float()
            target_var = torch.autograd.Variable(pproj).cuda().float()
        else:
            input_var = torch.autograd.Variable(p).float()
            target_var = torch.autograd.Variable(pproj).float()

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
        lossAvg.update(loss.data[0])


def validate(valDataset, model, criterion):
    """
        Run evaluation on validation set
    """
    l2err_avg = util.Average()

    # switch to evaluate mode
    model.eval()

    for i in range(len(valDataset)):

        # get validation sample
        sample = valDataset[i]
        p = sample['p']
        pproj = sample['pproj']
        if args['useCuda']:
            p_input = torch.autograd.Variable(p).cuda().float()
        else:
            p_input = torch.autograd.Variable(p).float()

        # compute output
        if args['useCuda']:
            pproj_hat = model(p_input).cpu().detach()
        else:
            pproj_hat = model(p_input).data
        l2err_avg.update(np.linalg.norm(pproj.numpy() - pproj_hat.numpy()))

    return l2err_avg.get()

def saveTestResults(dataset, model, filename):
    """
        Save model outputs to a mat file
    """

    model.eval()
    sample = dataset[0]
    d = sample['p'].shape[0]
    P         = np.zeros((d,len(dataset)))
    Pproj     = np.zeros((d,len(dataset)))
    Pproj_hat = np.zeros((d,len(dataset)))
    errs      = np.zeros((len(dataset)))

    for i in range(len(dataset)):

        # get sample
        sample = dataset[i]
        p = sample['p']
        pproj = sample['pproj']
        if args['useCuda']:
            p_input = torch.autograd.Variable(p).cuda().float()
        else:
            p_input = torch.autograd.Variable(p).float()

        # compute output
        if args['useCuda']:
            pproj_hat = model(p_input).cpu().detach()
        else:
            pproj_hat = model(p_input).data
        l2err = np.linalg.norm(pproj.numpy() - pproj_hat.numpy())

        # store
        P[:,i] = p
        Pproj[:,i] = pproj
        if args['useCuda']:
            Pproj_hat[:,i] = pproj_hat.detach().numpy()
        else:
            Pproj_hat[:,i] = pproj_hat.numpy()
        errs[i] = l2err

    # save
    scipy.io.savemat(filename, {'P':P, 'Pproj':Pproj, 'Pproj_hat':Pproj_hat, 'errs':errs})


if __name__ == '__main__':
    main()
