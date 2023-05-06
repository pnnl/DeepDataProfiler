#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 01:08:51 2021

"""
import os
import sys
import numpy as np
import pandas as pd
import collections
import torch
import torchvision
import torchvision.transforms as tv_transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from functools import partial
import cv2
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import h5py
from cifar_train import ResNet18
from tqdm import tqdm

class BasicBlock_fake(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_fake, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1.weight.data.fill_(0.001)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight.data.fill_(0.001)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #     # nn.BatchNorm2d(self.expansion * planes)
            # )
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut.weight.data.fill_(0.001)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet_fake(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_fake, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight.data.fill_(0.001)
        # self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4, stride=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_fake(num_classes):
    return ResNet_fake(BasicBlock_fake, [2, 2, 2, 2], num_classes)

def load_model(path, num_classes):
    model = ResNet18(num_classes)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'])
    return model

def inv_normalize(data, mean, std):
    return torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(data)


def get_activations(model, dataset, restrict_to_label):
    # storage_dict keeps saving the activations as they come
    storage_dict = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        storage_dict[name].append(out)

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    # Save all hooks to remove at the end
    hook_list = []
    for name, m in model.named_modules():
        if type(m) == torch.nn.BatchNorm2d and not 'shortcut' in name:
            # partial to assign the layer name to each hook
            hook_list.append(m.register_forward_hook(partial(save_activation, name)))

    # forward pass through the full dataset
    icons = []
    img_list = []
    with torch.no_grad():
        for batch, labels in tqdm(dataset, desc=f'Processing label={restrict_to_label}'):
            batch_sub = batch[labels == restrict_to_label]
            img_list.append(batch_sub.numpy())
            out = net(batch_sub)
            icons += [np.expand_dims(inv_normalize(img, norm_mean, norm_std).permute(1, 2, 0), 0)
                      for img in batch[labels == restrict_to_label]]

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    # storage_dict = {name: sample_batch_activation(torch.cat(outputs, 0)) for name, outputs in storage_dict.items()}
    storage_dict = {name: torch.cat(outputs, 0) for name, outputs in storage_dict.items()}
    storage_dict['icons'] = np.vstack(icons)
    storage_dict['imgs'] = np.vstack(img_list)

    # Remove all hooks
    [h.remove() for h in hook_list]
    return storage_dict

def get_activations2(model, dataset, restrict_to_label):
    # storage_dict keeps saving the activations as they come
    storage_dict = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        storage_dict[name].append(out)

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    # Save all hooks to remove at the end
    hook_list = []
    for name, m in model.named_modules():
        if type(m) == torch.nn.BatchNorm2d and not 'shortcut' in name:
            # partial to assign the layer name to each hook
            hook_list.append(m.register_forward_hook(partial(save_activation, name)))

    # forward pass through the full dataset
    icons = []
    with torch.no_grad():
        for batch, labels in tqdm(dataset, desc=f'Processing label={restrict_to_label}'):
            out = net(batch[labels == restrict_to_label])
            icons += [np.expand_dims(inv_normalize(img, norm_mean, norm_std).permute(1, 2, 0), 0)
                      for img in batch[labels == restrict_to_label]]

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    storage_dict = {name: sample_batch_activation(torch.cat(outputs, 0)) for name, outputs in storage_dict.items()}
    # storage_dict = {name: torch.cat(outputs, 0) for name, outputs in storage_dict.items()}
    storage_dict['icons'] = np.vstack(icons)

    # Remove all hooks
    [h.remove() for h in hook_list]
    return storage_dict


def sample_batch_activation(batch_activation):
    # Return one activation for layer while avoiding boundary patches
    batch_size, dim, patch_n, patch_m = batch_activation.shape
    print("sampling", batch_activation.shape)
    assert patch_n == patch_m

    batch_indices = np.arange(batch_size)

    # Sample while ignoring boundary indices
    rand_indices = np.random.randint(1, patch_n, size=(batch_size, 2))

    return F.relu(batch_activation[batch_indices, :, rand_indices[:, 0], rand_indices[:, 1]]).numpy()


def read_activation(filepath, layer):
    with h5py.File(filepath, 'r') as f:
        activation = f[layer][:]
        return activation


def get_mask_activations(model):
    img_size = 32
    pixel_mask = []
    for i in range(img_size):
        for j in range(img_size):
            mask_i = np.zeros(32*32).reshape(32,32)
            mask_i[i,j] = 1
            mask_i = [mask_i, mask_i, mask_i]
            pixel_mask.append(mask_i)
    pixel_mask = np.array(pixel_mask)
    pixel_mask = torch.from_numpy(pixel_mask)
    # mask_i = mask_i.type(torch.LongTensor)
    
    storage_dict = collections.defaultdict(list)

    def save_activation(name, mod, inp, out):
        storage_dict[name].append(out)


    hook_list = []
    for name, m in model.named_modules():
        # print(name, type(m))
        # if type(m) == BasicBlock_fake:
        if type(m) == torch.nn.Conv2d and not 'shortcut' in name:
            # partial to assign the layer name to each hook
            hook_list.append(m.register_forward_hook(partial(save_activation, name)))
    with torch.no_grad():
        out = model(pixel_mask.float())
        
    storage_dict = {name: torch.cat(outputs, 0) for name, outputs in storage_dict.items()}
    [h.remove() for h in hook_list]
    return storage_dict

def get_activations_norm(activations, l1=False):
    batch_size, dim, patch_n, patch_m = activations.shape
    assert patch_n == patch_m
    
    activations_norm = []
    for i in range(patch_n):
        av_norm_i = []
        for j in range(patch_m):
            av_norm_ij = [] 
            for k in range(batch_size):
                av = activations[k,:,i,j]
                if l1==True:
                    av_norm_ijk = np.linalg.norm(av, ord=1)
                else:
                    av_norm_ijk = np.linalg.norm(av)
                av_norm_ij.append(av_norm_ijk)
            av_norm_i.append(av_norm_ij)
        activations_norm.append(av_norm_i)
    activations_norm = np.array(activations_norm)
    return activations_norm

def get_foreground_masks(indices, data, iter_num = 2):
    print("get foreground masks...")
    indices_to_mask = {}
    for idx in indices:
        if idx%1000 == 0:
            print(idx)
        img_i = data[idx,:,:,:]
        img_i_new = foreground_detection(img_i, iter_num=iter_num)
        indices_to_mask[idx] = cv2.cvtColor(img_i_new, cv2.COLOR_BGR2GRAY)
    return indices_to_mask

def foreground_detection(img, iter_num=3):
    mask = np.zeros(img.shape[:2], dtype="uint8")

    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    rect = (0, 0, img.shape[0]-1, img.shape[1]-1)
    cv2.grabCut(img, mask, rect,  
            backgroundModel, foregroundModel,
            iter_num, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_BGD)|(mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    # mask2 = np.where((mask == cv2.GC_BGD), 0, 1).astype('uint8')

    img_new = img * mask2[:, :, np.newaxis]
    return img_new

def get_activation_weights(foreground_mask, av_norm, patch_size, l1=False):
    img_size = foreground_mask.shape[0]
    foreground_idx = []
    av_weights = np.zeros(patch_size * patch_size).reshape(patch_size, patch_size)
    for i in range(img_size):
        for j in range(img_size):
            if foreground_mask[i,j]!=0:
                pixel_idx = get_pixel_idx(i,j, img_size)
                foreground_idx.append(pixel_idx)
                # av_weights = np.sum(av_norm[:,:,pixel_idx])
    for i in range(patch_size):
        for j in range(patch_size):
            av_new = av_norm[i,j,foreground_idx]
            if l1==True:
                av_weights[i,j] = np.linalg.norm(av_new, ord=1)
            else:
                av_weights[i,j] = np.linalg.norm(av_new)
    return av_weights
                
            
    
def get_pixel_idx(i,j, img_size):
    return i*img_size + j
            
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    DATASET = "CIFAR10"
    num_classes = 10
    
    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(norm_mean.tolist(), norm_std.tolist())])
    
    
    train = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=False, transform=transforms)
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=2048, num_workers=4)

    net = load_model(f'../logs/{DATASET}_ResNet18_Custom_Aug/checkpoints/best.pth', num_classes)
    net.eval()
    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations/'

    # for label_filter in range(num_classes):
    #     activations = get_activations(net, trainloader, label_filter)
    #     with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
    #         [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
    #           activations.items()]
    #     del activations
    
    ############## get weights #####################
    print("getting pixel weights")
    model = ResNet18_fake(num_classes)
    mask_activations = get_mask_activations(model)
    del mask_activations['conv1']
    
    activation_norm_dict = {}
    activation_weights_dict = {}
    max_weights_dict = {}
    for key in mask_activations.keys():
        print(key)
        layer = key
        mask_activations_layer = mask_activations[layer]
        activations_norm_layer = get_activations_norm(mask_activations_layer)
        activation_norm_dict[layer] = activations_norm_layer
        patch_n, patch_m, pixel_size = activations_norm_layer.shape
        assert patch_n == patch_m
        activation_weights = np.zeros(patch_n * patch_m).reshape(patch_n, patch_m)
        for i in range(patch_n):
            for j in range(patch_m):
                activation_weights[i,j] = np.linalg.norm(activations_norm_layer[i,j,:])
                
        max_weights = np.max(activation_weights)
        max_weights_dict[layer] = max_weights
        activation_weights /= max_weights
        activation_weights_dict[layer] = activation_weights
    ############## get weights #####################
    names = ['airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck']
    
    
    layers_name = ['layer1.0.bn1', 'layer1.0.bn2', 'layer1.1.bn1', 'layer1.1.bn2',
   'layer2.0.bn1', 'layer2.0.bn2', 'layer2.1.bn1', 'layer2.1.bn2', 'layer3.0.bn1', 'layer3.0.bn2', 'layer3.1.bn1',
   'layer3.1.bn2', 'layer4.0.bn1', 'layer4.0.bn2', 'layer4.1.bn1', 'layer4.1.bn2']
    
    layers_name_selected = ['layer4.1.bn2', 'layer4.1.bn1', 'layer4.0.bn2', 'layer3.1.bn2', 'layer2.1.bn2', 'layer1.1.bn2']
    
    bn2conv = {}
    for i in range(len(mask_activations.keys())):
        conv_key = list(mask_activations.keys())[i]
        bn_key = layers_name[i]
        bn2conv[bn_key] = conv_key
        
    
    # for layer in layers_name:
    for layer in layers_name_selected:
        print(layer)
        layer_activations_5 = []
        layer_activations_1 = []
        # num_no_fg = 0
        for i in range(num_classes):
            print("class", i)
            layer_activations_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), layer)
            layer_imgs_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), "imgs")
            layer_activations_i_5 = []
            layer_activations_i_1 = []
            layer_activations_foreground = []
            batch_size = layer_activations_i.shape[0]
            patch_size = layer_activations_i.shape[2]
            for j in range(batch_size):
                if j%1000 == 0:
                    print(j)
                img_ij = layer_imgs_i[j,:,:,:]
                img_ij = np.einsum('ijk->kji', img_ij).astype('uint8')
                img_ij_fg = foreground_detection(img_ij, iter_num=5)
                img_ij_fg_grey = cv2.cvtColor(img_ij_fg, cv2.COLOR_BGR2GRAY)
                av_weights_ij = get_activation_weights(img_ij_fg_grey, activation_norm_dict[bn2conv[layer]], patch_size)
                av_weights_ij_5_val = av_weights_ij.flatten()[av_weights_ij.flatten().argsort()[-5:]][0]
                av_weights_ij_1_val = av_weights_ij.flatten()[av_weights_ij.flatten().argsort()[-1:]][0]
                
                layer_activations_ij = layer_activations_i[j,:,:,:]
                layer_activations_ij_5 = []
                layer_activations_ij_1 = []
                for k in range(patch_size):
                    for l in range(patch_size):
                        if av_weights_ij[k,l] >= av_weights_ij_5_val: 
                            layer_activations_ij_5.append(layer_activations_ij[:,k,l])
                        if av_weights_ij[k,l] >= av_weights_ij_1_val:
                            layer_activations_ij_1.append(layer_activations_ij[:,k,l])
                layer_activations_ij_5 = np.vstack([ll for ll in layer_activations_ij_5])
                layer_activations_ij_1 = np.vstack([ll for ll in layer_activations_ij_1])
                if len(layer_activations_ij_5) > 5:
                    indices = np.arange(len(layer_activations_ij_5))
                    np.random.shuffle(indices)
                    indices = indices[0:5]
                    layer_activations_ij_5 = layer_activations_ij_5[indices]
                if len(layer_activations_ij_1) > 1:
                    # num_no_fg += 1
                    # print(av_weights_ij_1_val)
                    layer_activations_ij_1 = layer_activations_ij_1[np.random.randint(len(layer_activations_ij_1))]                
                    
                layer_activations_i_5.append(layer_activations_ij_5)
                layer_activations_i_1.append(layer_activations_ij_1)
            layer_activations_i_5 = np.vstack([l for l in layer_activations_i_5])
            layer_activations_i_1 = np.vstack([l for l in layer_activations_i_1])
            label_i_5 = np.repeat(i, len(layer_activations_i_5)).reshape(-1,1)
            label_i_1 = np.repeat(i, len(layer_activations_i_1)).reshape(-1,1)
            layer_activations_i_5 = np.hstack([label_i_5, layer_activations_i_5])
            layer_activations_i_1 = np.hstack([label_i_1, layer_activations_i_1])

            layer_activations_5.append(layer_activations_i_5)
            layer_activations_1.append(layer_activations_i_1)
        
        layer_activations_5 = np.vstack([layer_activations_i for layer_activations_i in layer_activations_5])
        layer_activations_1 = np.vstack([layer_activations_i for layer_activations_i in layer_activations_1])
        
        layer_activations_5_df = pd.DataFrame(layer_activations_5)
        layer_activations_1_df = pd.DataFrame(layer_activations_1)
        
        print("top 5", layer_activations_5.shape)
        print("top 1", layer_activations_1.shape)
        # print("num_no_fg", num_no_fg)

    
        cols = np.array(['label'])
        cols = np.concatenate((cols,np.arange(1,layer_activations_5.shape[1]).astype("str")))
        
        df_5_dir = "../datasets/cifar10_full_fg_5_df/"
        df_1_dir = "../datasets/cifar10_full_fg_1_df/"
    
        layer_activations_5_df.columns = cols 
        layer_activations_5_df['label'] = [names[int(layer_activations_5_df['label'].iloc[i])] for i in range(len(layer_activations_5_df))]
        layer_activations_5_df.to_csv(df_5_dir+"train_full_fg_5_"+layer+".csv", index=False)
    
        layer_activations_1_df.columns = cols 
        layer_activations_1_df['label'] = [names[int(layer_activations_1_df['label'].iloc[i])] for i in range(len(layer_activations_1_df))]
        layer_activations_1_df.to_csv(df_1_dir+"train_full_fg_1_"+layer+".csv", index=False)
    
        
    
    


