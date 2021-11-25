#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:11:20 2020

@author: cwx
"""

import sys
sys.path.append('.')
import torch
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from model.centerline_net import CenterlineNet_Discrimintor_2D_Radii_32 
from functions.utlis import generate_sphere, local_max, compute_trees, build_nodelist, connected
from functions.io1 import read3dtiff
from functions.CNNTracker_v2D_radii import Tracker


'''
#----------------------- Parameters -----------------------#
'''
Ma = 32
Mp = 32
K = 1024
Mc = int(np.sqrt(K))
Lambda = 4 # can be tuned between 1 to 4 for optimal result
angle_T = np.pi/3 # angle threshold
n = 10 # maximun radius of the spherical core
psize = n
node_step = 1
max_iter = 100
step_size = 1
mask_size = 3


'''
#-------------------- Prepare CNN Model --------------------#
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model and weights
model = CenterlineNet_Discrimintor_2D_Radii_32(NUM_ACTIONS=K, n=n).to(device)
checkpoint_path = './checkpoint/classification_checkpoints/'
checkpoint = torch.load(checkpoint_path + 'weight.pkl')
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['net_dict'])    
model.eval()


'''
#---------------- Prepare Images and Paths-----------------#
'''
test_name = 'Test_SPE_DNR'
root_path = './test_samples/test_images/'
root_path_seg = './test_samples/seed_maps/'
root_soma_path = './test_samples/soma_masks/'
results_path = './test_results/' + test_name + '/'
if not os.path.exists(results_path):
    os.mkdir(results_path) # make dir

# get test image names
files = os.listdir(root_path)
img_cand = []
for i in range(len(files)):
    tmp = files[i]
    img_cand.append(tmp)


'''
#----------------- Neuron Reconstruction -----------------#
'''
sphere_core, _, _ = generate_sphere(Ma, Mp) # for spherical patches extraction
sphere_core_label, _, _ = generate_sphere(Mc, Mc) # for direction determination

# start loop for each test image
for i in range(len(img_cand)):
    img_ind = img_cand[i]
    print(img_ind)
    
    # load test image
    img_path1 = root_path + img_ind + '/'
    new_ind = []
    files = os.listdir(img_path1)
    for j in range(len(files)):
        filename = files[j]
        if filename[-3:] == 'tif':
            img = read3dtiff(img_path1 + filename)
            img = img.astype('float')
            
    # load soma mask
    soma_mask = read3dtiff(root_soma_path + img_ind + '_soma.tif')    
    soma_mask = np.flip(soma_mask, 1)
    origin_shape = img.shape
    
    # image padding
    img2 = np.pad(img, ((psize,psize),(psize,psize),(psize,psize)), 'constant',
                  constant_values=((0.0,0.0),(0.0,0.0),(0.0,0.0)))  
    soma_mask2 = np.pad(soma_mask, ((psize,psize),(psize,psize),(psize,psize)),
                        'constant', constant_values=((0.0,0.0),(0.0,0.0),(0.0,0.0)))    
    Zb,Xb,Yb = np.shape(img2)     
    
    # load seed map and extract seed points
    img_path_seg = root_path_seg + img_ind +'_seg.tif'
    img_seg = read3dtiff(img_path_seg)
    suppress, candidate_sup = local_max(img_seg, wsize=3, thre=0.5*255)
    candidate_file = np.array(candidate_sup)
    candidate_file = np.flipud(candidate_file[np.argsort(candidate_file[:,-1])])        
    candidate_file[:, :3] += psize         
        
    # tracing
    tracker = Tracker(img2, soma_mask2, candidate_file, Lambda, K,
                      angle_T, max_iter, step_size, node_step, mask_size, 
                      Ma, Mp, n, Xb, Yb, Zb, 
                      model, sphere_core, sphere_core_label, device)    
    tracker.trace_JointDecision()
    
    # graph reconstruction
    n0 = tracker.ndlist
    tree = compute_trees(n0)
    swc = build_nodelist(tree)
    swc[:,2:5] = swc[:,2:5] - psize        
    swc[:,2:5] = swc[:,2:5] + 1 # Vaa3d starts from 1 but python from 0   
    distance_transform = distance_transform_edt(soma_mask)
    # build soma shape
    for i in range(len(swc)):
        if swc[i,-1] == -1:
            swc[i, 5] = distance_transform[int(swc[i, 4]), int(swc[i, 3]), int(swc[i, 2])]
    # use this result for multi-neuron reconstruction
    save_swc_path = results_path + img_ind + '.swc'
    connected(soma_mask, img, swc, save_swc_path, distance_transform)
    
    # preserve the largest tree
    # use this result for single-neuron reconstruction
    save_swc_path_singletree = results_path + img_ind + 'singletree.swc'
    max_treeId = np.argmax(np.bincount(swc[:,1].astype(int)))
    swc_single_tree = swc[np.where(swc[:,1]==max_treeId),:].squeeze()
    connected(soma_mask, img, swc_single_tree, save_swc_path_singletree, distance_transform)
    print('Done')