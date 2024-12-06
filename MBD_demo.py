# -*- coding: utf-8 -*-
"""

MBD: Multi b-value Denoising of Diffusion Magnetic Resonance Images
Jakub Jurek (1), Andrzej Materka (1), Kamil Ludwisiak (2), Agata Majos (3), Filip Szczepankiewicz (4)

(1) Institute of Electronics, Lodz University of Technology, Aleja Politechniki 10, PL-93590 Lodz, Poland
(2) Department of Diagnostic Imaging, Independent Public Health Care, Central Clinical Hospital, Medical University of Lodz, Pomorska 2 51, PL-92213 Lodz, Poland
(3) Department of Radiology, Medical University of Lodz, Lodz, Poland
(4) Medical Radiation Physics, Lund University, Barngatan 4, 22185 Lund, Sweden

This code demonstrates how to use MBD. A brain phantom dataset is used.

"""

import os

# set path to the working directory
path = "/home/likewise-open/ADM/jakub.jurek/MBD/case2/"
os.chdir(path)

import numpy as np
import matplotlib.pyplot as plt
from mbd_tools import addRicianNoise, get_dimg_patches, getRicianM, SRData, DnCNN, DnCNNe, try_gpu, plot_contour
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import nibabel as nib

np.random.seed(100)

# theoretical intensity of pure white matter voxels for spin echo MRI and parameters:
# proton density = 0.77, T1 = 500 ms, T2 = 70 ms, m0=1000, TR=6700, TE = 100
wm = 0.77*1000*np.exp(-100/70)*(1-np.exp(-6700/500))

data = []

# IDs of lesion sets merged with the phantom
ids = [1,2,3] #[1,2,3,4,5,6,7,8]

for pat_id in ids:
    data.append(nib.load(path+'data//dti_L%sl.nii.gz'%pat_id).get_fdata()[:,:,60:120,:])
data = np.concatenate(data,-2)

# mask for voxels where lesion content is > 0%
masks = []
for pat_id in range(len(ids)):
    masks.append(nib.load(path+'data//dtiMask2_L1l.nii.gz').get_fdata()[:,:,60:120])
masks = np.concatenate(masks,-1)

# # mask for voxels where lesion content is 100%
# masks2 = []
# for pat_id in range(len(ids)):
#     masks2.append(nib.load(path+'data//dtiMask_L1l.nii.gz').get_fdata()[:,:,60:120])
# masks2 = np.concatenate(masks2,-1)

# noise levels relative to WM intensity
nlevels = np.array([0.01, 0.03, 0.05])*wm
nlev = 1 # set noise level index relative to nlevels

m  = addRicianNoise(data,nlevels[nlev]) # input images
targetx  = addRicianNoise(data[:,:,:,2],nlevels[nlev]) # target images

s = 72; d = -1
vmx1 = np.mean(m[:,:,s,0])+2*np.std(m[:,:,s,0])
vmx2 = np.mean(m[:,:,s,1])+2*np.std(m[:,:,s,1])
vmx3 = np.mean(m[:,:,s,2])+2*np.std(m[:,:,s,2])

fig, ax = plt.subplots(1,5,figsize=(10,5))
ax[0].imshow(m[:,:,s,0],cmap='gray', vmin=0, vmax=vmx1)
ax[0].set_axis_off()
ax[1].imshow(m[:,:,s,1],cmap='gray', vmin=0, vmax=vmx2)
ax[1].set_axis_off()
ax[2].imshow(m[:,:,s,2],cmap='gray', vmin=0, vmax=vmx3)
ax[2].set_axis_off()
ax[3].imshow(targetx[:,:,s],cmap='gray', vmin=0, vmax=vmx3)
ax[3].set_axis_off()
ax[4].imshow(data[:,:,s,-1],cmap='gray', vmin=0, vmax=vmx3)
ax[4].set_axis_off()

##################################################

rilrx = m[:,:,:,m.shape[-1]-1] # take the DWI at b-val to denoise

psize = 30 # training patch size
stride = 30 # step between patch centers (for no gap/overlap set stride=psize )
psizexv = stridexv =  rilrx.shape[0] # validation patch size (full slice)
psizeyv = strideyv =  rilrx.shape[1]

trains = list(range(targetx.shape[2])) # all slice indices

sv = np.random.choice(np.arange(0,rilrx.shape[2]),20,replace=False) # 20 validation slices
example = 5 # set validation example index to display during training

[trains.remove(i) for i in sv] # training slice indices after removing validation slices

# train patches
p=[]
p.append(get_dimg_patches(rilrx[:,:,trains,np.newaxis],psize,stride,psize,stride))
x = np.stack(p,1)
y = get_dimg_patches(targetx[:,:,trains,np.newaxis],psize,stride,psize,stride)

# validation patches
pv = []
pv.append(get_dimg_patches(rilrx[:,:,sv,np.newaxis],psizexv,stridexv,psizeyv,strideyv))
xv = np.stack(pv,1)
yv = get_dimg_patches(targetx[:,:,sv,np.newaxis],psizexv,stridexv,psizeyv,strideyv)

# adding extra b-values to the input arrays already containing b4000. Choose between indices to play. 1 is b1000, 0 is b0, 2 is b4000.
for b in [1, 0]: 

    rilrx2 = addRicianNoise(np.mean(m[:,:,:,[b]],-1),nlevels[nlev])

    p=[]
    pv = []
    p.append(get_dimg_patches(rilrx2[:,:,trains,np.newaxis],psize,stride,psize,stride))
    pv.append(get_dimg_patches(rilrx2[:,:,sv,np.newaxis],psizexv,stridexv,psizeyv,strideyv))

    x2 = np.stack(p,1)
    xv2 = np.stack(pv,1)
    
    x = np.concatenate([x2,x],axis=1)
    xv = np.concatenate([xv2,xv],axis=1)

# # augmentation by rotation 90, 180, 270 degrees. Uncomment to turn on
# x = np.concatenate([x,np.rot90(x,axes=(2,3)),np.rot90(x,2,axes=(2,3)),np.rot90(x,3,axes=(2,3))],0)
# y = np.concatenate([y,np.rot90(y,axes=(1,2)),np.rot90(y,2,axes=(1,2)),np.rot90(y,3,axes=(1,2))],0)

print("Data shapes")
print("Training inputs:", x.shape, "Training targets:", y.shape, "Validation inputs:", xv.shape, "Validation targets:", yv.shape)

pu = 'gpu' # set to 'cpu' if 'gpu' not accessible
training_set = SRData([x,y], pu = pu)
validation_set = SRData([xv,yv], pu = pu)

mbdmodel = DnCNN(x.shape[1],5).to(try_gpu()) # create MBD model instance
n2nmodel = DnCNN(1,5).to(try_gpu()) # create N2N model instance
cnnemodel = DnCNNe(x.shape[1]-1,5).to(try_gpu()) # create CNNe model instance

"""
N2N and CNNe are used here for comparison only. Early stopping/best weights selection is done for MBD only. 
"""

# Training hyperparameters
loss_function = MSELoss()
optimizer = Adam(mbdmodel.parameters(), lr=0.01)
data_loader = DataLoader(training_set, batch_size=64, shuffle=True)

loss_function2 = MSELoss()
optimizer2 = Adam(n2nmodel.parameters(), lr=0.01)

loss_function3 = MSELoss()
optimizer3 = Adam(cnnemodel.parameters(), lr=0.01)

# Empirical value of minimal MSE loss in training, according to Lehtinen's Noise2Noise
bottom = np.var(getRicianM(data[:,:,sv,-1],nlevels[nlev])-targetx[:,:,sv])

# Empirical value of maximal MSE loss in training, according to Lehtinen's Noise2Noise
top = np.var(rilrx[:,:,sv]-targetx[:,:,sv])

max_epochs = 1000 # training will stop after max_epochs unless early stopping is executed

clean = np.moveaxis(getRicianM(data[:,:,sv,-1],nlevels[nlev]),-1,0)

tloss=np.zeros([max_epochs,1]) # training loss vector (MBD)
vloss=np.zeros([max_epochs,1]) # validation loss vector
tloss2=np.zeros([max_epochs,1]) # training loss vector (N2N)
vloss2=np.zeros([max_epochs,1]) # validation loss vector
tloss3=np.zeros([max_epochs,1]) # training loss vector (CNNe)
vloss3=np.zeros([max_epochs,1]) # validation loss vector

# Parameters of the custom early stopping module
patience = 20
min_delta = 10e-9
epstep = 1
model_history = {}
firstc = 1000000000
flag = 0

for k in range(max_epochs):
    for i, batch in enumerate(data_loader):
        train_input, train_target = batch
        if k==0 and i==0:

            val_input = validation_set.input
            val_target = validation_set.target
            
            # display range for the validation images
            maxv1 = np.mean(val_input[example,0,:,:].detach().cpu().numpy()) + 3*np.std(val_input[example,0,:,:].detach().cpu().numpy())
            maxv2 = np.mean(val_input[example,1,:,:].detach().cpu().numpy()) + 3*np.std(val_input[example,1,:,:].detach().cpu().numpy())
            maxv3 = np.mean(val_target[example,:,:].detach().cpu().numpy()) + 3*np.std(val_target[example,:,:].detach().cpu().numpy())
            
        train_output = mbdmodel(train_input)
        loss_train = loss_function(train_output, train_target)
        train_output2 = n2nmodel(train_input[:,-1:,:,:])
        loss_train2 = loss_function(train_output2, train_target)
        train_output3 = cnnemodel(train_input[:,:-1,:,:])
        loss_train3 = loss_function(train_output3, train_target)
        
        # update models
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        optimizer2.zero_grad()
        loss_train2.backward()
        optimizer2.step()
        
        optimizer3.zero_grad()
        loss_train3.backward()
        optimizer3.step()

    val_output = mbdmodel(val_input)
    val_output2 = n2nmodel(val_input[:,-1:,:,:])
    val_output3 = cnnemodel(val_input[:,:-1,:,:])

    loss_val = loss_function(val_output[:,0,:,:], val_target)
    loss_val2 = loss_function(val_output2[:,0,:,:], val_target)
    loss_val3 = loss_function(val_output3[:,0,:,:], val_target)

    tloss[k]=loss_train.item()
    vloss[k]=loss_val.item()
    tloss2[k]=loss_train2.item()
    vloss2[k]=loss_val2.item()
    tloss3[k]=loss_train3.item()
    vloss3[k]=loss_val3.item()
  
    # Check validation cost every epstep epochs
    the_loss=np.copy(vloss)
    if (k) % epstep == 0:
        min_cost=np.min(the_loss[:k+1])
        if the_loss[k] == min_cost:
            model_history['0']=mbdmodel
            epoch_at_which = k
        if k>0:
            n=int(((k)/epstep))
            delta=(the_loss[n-1]-the_loss[n])
            if delta<min_delta or firstc<the_loss[n]:
                if flag==0:
                    firstc=the_loss[n-1]
                flag+=1
            elif firstc>=the_loss[n] and delta>=min_delta:
                flag=0
                model_history['prev']=mbdmodel
            if flag>=patience:
                break

    if k % 10 == 0:

        plt.figure(figsize=(20,5))
        plt.title("Learning curves"); plt.xlabel('Epochs'); plt.ylabel('Validation MSE loss')
        plt.subplot(121)
        plt.plot(vloss[:k+1],'g',label='MBD')
        plt.plot(vloss2[:k+1],'r',label='N2N')
        plt.plot(vloss3[:k+1],'b',label='CNNe')
        plt.plot(vloss[:k+1]*0 + top,'k-.',label='max error')
        plt.plot(vloss[:k+1]*0 + bottom,'k--',label='min error')
        plt.xlim([0,k+1])
        plt.ylim([0.9*bottom, 1.1*top])
        plt.legend()
        plt.show()
        
        diff = val_target[example,:,:].detach().cpu().numpy()-val_output[example,0,:,:].detach().cpu().numpy()
        diff2 = val_target[example,:,:].detach().cpu().numpy()-val_output2[example,0,:,:].detach().cpu().numpy()
        diff3 = val_target[example,:,:].detach().cpu().numpy()-val_output3[example,0,:,:].detach().cpu().numpy()
        
        plt.figure(figsize=(15,15))
        plt.suptitle("Learning progress")
        plt.subplot(441); plt.title("Input, b4000")
        plt.imshow(val_input[example,-1,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3, cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(442); plt.title("Input, b1000")
        plt.imshow(val_input[example,1,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv2, cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(443); plt.title("Input, b0")
        plt.imshow(val_input[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv1, cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(444); plt.title("Target, b4000")
        plt.imshow(val_target[example,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(445); plt.title("MBD Output, b4000")
        plt.imshow(val_output[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(446); plt.title("N2N Output, b4000")
        plt.imshow(val_output2[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(447); plt.title("CNNe Output, b4000")
        plt.imshow(val_output3[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(448); plt.title("Clean Rician Biased, b4000")
        plt.imshow(clean[example,:,:],vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none'); plt.axis('off'); plt.colorbar()
        plt.subplot(449); plt.title("MBD residuals, MAE=%.2f"%np.mean(np.abs(diff)))
        plt.imshow(diff,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        plt.subplot(4,4,10); plt.title("N2N residuals, MAE=%.2f"%np.mean(np.abs(diff2)))
        plt.imshow(diff2,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        plt.subplot(4,4,11); plt.title("CNNe residuals, MAE=%.2f"%np.mean(np.abs(diff3)))
        plt.imshow(diff3,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        plt.subplot(4,4,13); plt.title("MBD error")
        plt.imshow(val_output[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        plt.subplot(4,4,14); plt.title("N2N error")
        plt.imshow(val_output2[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        plt.subplot(4,4,15); plt.title("CNNe error")
        plt.imshow(val_output3[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5); plt.axis('off'); plt.colorbar()
        
        plt.show()
        
        print("Smallest val. loss so far:", min(vloss[:k+1]))

model_mbd = model_history['0']

##################################################

# Show final result with lesion area contours

fig, ax = plt.subplots(4,4,figsize=(15,15))

ax[0][0].set_title("Input, b4000")
c1 = ax[0][0].imshow(val_input[example,-1,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3, cmap="Greys_r", interpolation = 'none')
ax[0][1].set_title("Input, b1000")
c2 = ax[0][1].imshow(val_input[example,1,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv2, cmap="Greys_r", interpolation = 'none')
ax[0][2].set_title("Input, b0")
c3 = ax[0][2].imshow(val_input[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv1, cmap="Greys_r", interpolation = 'none')
ax[0][3].set_title("Target, b4000")
c4 = ax[0][3].imshow(val_target[example,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none')
ax[1][0].set_title("MBD Output, b4000")
c5 = ax[1][0].imshow(val_output[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none')
ax[1][1].set_title("N2N Output, b4000")
c6 = ax[1][1].imshow(val_output2[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none')
ax[1][2].set_title("CNNe Output, b4000")
c7 = ax[1][2].imshow(val_output3[example,0,:,:].detach().cpu().numpy(),vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none')
ax[1][3].set_title("Clean Rician Biased, b4000")
c8 = ax[1][3].imshow(clean[example,:,:],vmin=0,vmax=maxv3,cmap="Greys_r", interpolation = 'none')
ax[2][0].set_title("MBD residuals, MAE=%.2f"%np.mean(np.abs(diff)))
c9 = ax[2][0].imshow(diff,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)
ax[2][1].set_title("N2N residuals, MAE=%.2f"%np.mean(np.abs(diff2)))
c10 = ax[2][1].imshow(diff2,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)
ax[2][2].set_title("CNNe residuals, MAE=%.2f"%np.mean(np.abs(diff3)))
c11 = ax[2][2].imshow(diff3,cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)
ax[3][0].set_title("MBD error")
c12 = ax[3][0].imshow(val_output[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)
ax[3][1].set_title("N2N error")
c13 = ax[3][1].imshow(val_output2[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)
ax[3][2].set_title("CNNe error")
c14 = ax[3][2].imshow(val_output3[example,0,:,:].detach().cpu().numpy()-clean[example,:,:],cmap="Greys_r", interpolation = 'none',vmin=-maxv3/5, vmax=maxv3/5)

for a in ax:
    for axx in a:
        axx.set_axis_off()

fig.colorbar(c1, ax=ax[0][0], label='', fraction=0.054)
fig.colorbar(c2, ax=ax[0][1], label='', fraction=0.054)
fig.colorbar(c3, ax=ax[0][2], label='', fraction=0.054)
fig.colorbar(c4, ax=ax[0][3], label='', fraction=0.054)
fig.colorbar(c5, ax=ax[1][0], label='', fraction=0.054)
fig.colorbar(c6, ax=ax[1][1], label='', fraction=0.054)
fig.colorbar(c7, ax=ax[1][2], label='', fraction=0.054)
fig.colorbar(c8, ax=ax[1][3], label='', fraction=0.054)
fig.colorbar(c9, ax=ax[2][0], label='', fraction=0.054)
fig.colorbar(c10, ax=ax[2][1], label='', fraction=0.054)
fig.colorbar(c11, ax=ax[2][2], label='', fraction=0.054)
fig.colorbar(c12, ax=ax[3][0], label='', fraction=0.054)
fig.colorbar(c13, ax=ax[3][1], label='', fraction=0.054)
fig.colorbar(c14, ax=ax[3][2], label='', fraction=0.054)

plot_contour(ax[0][0],masks[:,:,sv[example]]>2)
plot_contour(ax[0][1],masks[:,:,sv[example]]>2)
plot_contour(ax[0][2],masks[:,:,sv[example]]>2)
plot_contour(ax[0][3],masks[:,:,sv[example]]>2)
plot_contour(ax[1][0],masks[:,:,sv[example]]>2)
plot_contour(ax[1][1],masks[:,:,sv[example]]>2)
plot_contour(ax[1][2],masks[:,:,sv[example]]>2)
plot_contour(ax[1][3],masks[:,:,sv[example]]>2)
plot_contour(ax[2][0],masks[:,:,sv[example]]>2)
plot_contour(ax[2][1],masks[:,:,sv[example]]>2)
plot_contour(ax[2][2],masks[:,:,sv[example]]>2)
plot_contour(ax[3][0],masks[:,:,sv[example]]>2)
plot_contour(ax[3][1],masks[:,:,sv[example]]>2)
plot_contour(ax[3][2],masks[:,:,sv[example]]>2)