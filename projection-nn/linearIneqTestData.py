import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import Animation
import scipy.io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.datasets as datasets


def makeSimplePolygonData():
    """
    Generates 4 inequalities that form a simple convex set: the square rotated 45 degrees that is circumscribed by [-1,1]^2.
    """

    A = np.array([[+1.0, +1.0], [+1.0, -1.0], [-1.0, -1.0], [-1.0, +1.0]])
    b = np.array([+0.5, +0.5, +0.5, +0.5]).transpose()

    return {'A':A, 'b':b}


def makeSimpleTriangleData():
    """
    Generates 3 inequalities that form a simple 45-45-90 triangluar convex set
    (base parallel to the x-axis and circumscribed by [-1,1]^2)
    """

    A = np.array([[-1.0, -1.0], [+1.0, -1.0], [0.0, +1.0]])
    b = np.array([-0.5, -0.5, -0.5]).transpose() # should be [-.5 -.5 +0.5] ?

    return {'A':A, 'b':b}


def makeRandomData(d, n):
    """
    Generate test data (intersection of RANDOM linear inequality constraints)

    Inputs:
        - d         : ambient dimension
        - n         : number of constraints
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

    return {'A':A, 'b':b}


def makePointProjectionPairs(inequalities, K):
    
    d = inequalities['A'].shape[1]
    n = inequalities['b'].shape[0]
    P = np.random.random((d,K))*2.0 - 1.0
    Pproj = np.zeros((d,K))
    for i in range(K):
        p = P[:,i]
        pproj = p
        for k in range(n): # project onto each of n hyperplanes, n times
            for j in range(n):
                if inequalities['A'][j,:] @ pproj > inequalities['b'][j]:
                    # project pproj onto hyperplane $$H = \left\{ x \in \R^d \colon x^T a^{(j)} = b^{(j)} \right\}$$:
                    #  $$P_H(x) = x - \frac{x^T a^{(j)} - b}{\norm{a^{(j)}}_2^2} a^{(j)}$$
                    aj = inequalities['A'][j,:]
                    pproj = pproj - 1/(np.transpose(aj)@aj)*(np.transpose(pproj)@aj-inequalities['b'][j]) * aj
                    #debugPlot(inequalities, p, pproj)
        Pproj[:,i] = pproj

    return {'P':torch.from_numpy(P), 'Pproj':torch.from_numpy(Pproj)}


def plot(inequalities, P=None, Pproj=None, Pproj_hat=None, showplot=False, savefile="testdata.png"):

    if inequalities['A'].shape[1] is not 2:
        print('Ambient dimension must be 2 to plot.')
        return

    # plot inequalities
    A = inequalities['A']
    b = inequalities['b']
    x1plt = np.linspace(-1.0,1.0,1000)
    fig, ax = plt.subplots()
    if 'x' in inequalities:
        ax.plot(inequalities['x'][0], inequalities['x'][1], 'kx')
    for i in range(b.shape[0]):
        ax.plot(x1plt, (b[i]-A[i,0]*x1plt)/A[i,1], 'k-')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ax.grid()
    ax.axis('equal')

    # plot point / projected point pairs
    if P is not None and Pproj is not None:
        for i in range(P.shape[1]):
            ax.plot(P[0,i], P[1,i], 'b.', markersize=1)
            ax.plot(Pproj[0,i], Pproj[1,i], 'r.', markersize=1)
    if Pproj_hat is not None:
        for i in range(P.shape[1]):
            ax.plot(Pproj_hat[0,i], Pproj[1,i], 'g.', markersize=1)

    if showplot:
        plt.show()

    if savefile is not None:
        fig.savefig(savefile)


def makevideo(inequalities, P, Pproj, Pproj_hat, savefile="trainvideo.mp4", errs=None, losses=None):

    if inequalities['A'].shape[1] is not 2:
        print('Ambient dimension must be 2 to plot.')
        return

    nframes = Pproj_hat.shape[2]
    npoints = P.numpy().shape[1]

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=15)

    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')

    x1plt = np.linspace(-1.0, 1.0, 1000)
    A = inequalities['A']
    b = inequalities['b']

    with writer.saving(fig, savefile, 300):

        for framenum in range(nframes):

            print('Making video: frame {0:d}/{1:d}'.format(framenum+1,nframes))
            
            plt.cla()

            # plot inequalities
            for i in range(b.shape[0]):
                plt.plot(x1plt, (b[i]-A[i,0]*x1plt)/A[i,1], 'k-')

            # plot line between ground truth and reconstructed projected point
            for i in range(npoints):
                plt.plot([Pproj[0,i], Pproj_hat[0,i,framenum]], [Pproj[1,i], Pproj_hat[1,i,framenum]],
                         color='0.2', linewidth=0.2)

            # plot ground truth point / projected point pairs
            for i in range(npoints):
                plt.plot(P[0,i], P[1,i], 'b.', markersize=5)
                plt.plot(Pproj[0,i], Pproj[1,i], 'r.', markersize=5)

            # plot network-projected points
            for i in range(npoints):
                plt.plot(Pproj_hat[0,i,framenum], Pproj_hat[1,i,framenum], 'g.', markersize=3)

            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.grid()
            if errs is not None and losses is not None:
                plt.title('Epoch {0:d}/{1:d} (mean l2 error = {2:.5f}; loss = {3:.5f})'.format(
                          framenum+1,nframes,errs[framenum],losses[framenum]))
            elif losses is not None:
                plt.title('Epoch {0:d}/{1:d} (mean l2 error = {2:.5f})'.format(
                          framenum+1,nframes,errs[framenum]))
            else:
                plt.title('Epoch {0:d}/{1:d}'.format(framenum+1,nframes))

            writer.grab_frame()

