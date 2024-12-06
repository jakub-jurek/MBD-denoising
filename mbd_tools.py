
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch import from_numpy
import torch.nn as nn
import numpy as np
import nibabel as nib
from scipy.special import i0, i1


def get_contour(msx):
    cnt = np.zeros(msx.shape,int)
    for x in range(msx.shape[0]):
        for y in range(msx.shape[1]):
            if msx[x,y]:
                if msx[x-1,y]==0 or msx[x+1,y]==0 or \
                msx[x,y-1]==0 or msx[x,y+1]==0: cnt[x,y]=1
    return cnt

def plot_contour(ax,msx):
    cnt = get_contour(msx)
    ax.scatter(np.where(cnt==1)[1],np.where(cnt==1)[0],c='r',s=0.3)
    

class SRData(Dataset):

    def __init__(self, data, pu = 'gpu'):

        if pu == 'gpu':
          self.input=from_numpy(data[0].astype(np.float32)).cuda()
          self.target=from_numpy(data[1].astype(np.float32)).cuda()
        elif pu == 'cpu':
          self.input=from_numpy(data[0].astype(np.float32))
          self.target=from_numpy(data[1].astype(np.float32))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index,...], self.target[index:index+1,...]

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
    
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 'same'
        
        features = 18*channels 
        layers = []

        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, groups=channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, groups=1, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):

        out = self.dncnn(x[:,:,:,:])
        return x[:,-1:,:,:]-out
        # return out

class DnCNNe(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNNe, self).__init__()
        kernel_size = 3
        padding = 'same'
        
        features = 54
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, groups=channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, groups=1, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
    
        out = self.dncnn(x[:,:,:,:])
        return out
    
def get_dimg_patches(img,patchsizex, stridex, patchsizey, stridey):
    
    (x, y, z, d) = np.shape(img);
    n=0;
    trainingdata = np.zeros([d*z*len(np.arange(0,x-patchsizex+1,stridex))*len(np.arange(0,y-patchsizey+1,stridey)),patchsizex,patchsizey])
    for l in range(d):
        for k in range(z):            
            for i in range(0,x-patchsizex+1,stridex):
                for j in range(0,y-patchsizey+1,stridey):
                    trainingdata[n,:,:]=img[i:i+patchsizex,j:j+patchsizey,k,l];
                    n+=1;

    return trainingdata

def addRicianNoise(img,sigma, output_type='M'):
    g1 = np.random.normal(0,sigma,img.shape)
    g2 = np.random.normal(0,sigma,img.shape)
    Mn = np.sqrt((img/np.sqrt(2) + g1)**2 + (img/np.sqrt(2) + g2)**2)

    if output_type == 'MN':
       
        Gn = g1+1j*g2
        Rn = Mn-img
        return Mn, Rn, Gn
    
    elif output_type == 'M':
        return Mn

def getRicianM(A, sigma):
    term = A**2/(4*sigma**2)
    M = 1/(2*sigma**2)*np.exp(-term)*np.sqrt(np.pi/2)*sigma*((A**2+2*sigma**2)*i0(term)+A**2*i1(term))
    M[np.isinf(M)] = A[np.isinf(M)]
    M[np.isnan(M)] = A[np.isnan(M)]
    
    return M