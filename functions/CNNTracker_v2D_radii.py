#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm
from functions.utlis import Spherical_Patches_Extraction, Node

class Tracker(object):
    def __init__(self, img2, soma_mask2, terminations, Lambda, K, 
                 angle_T, max_iter, step_size, node_step, mask_size, 
                 Ma, Mp, n, Xb, Yb, Zb, 
                 model, sphere_core, sphere_core_label, device):
        
        self.img2 = img2
        self.soma_mask2 = soma_mask2
        self.indx_map = np.zeros_like(img2, dtype=np.int64) # index map to label the index of traced point
        self.terminations = terminations
        self.angle_T = angle_T
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_step = node_step
        self.Ma = Ma
        self.Mp = Mp
        self.n = n
        self.Xb = Xb
        self.Yb = Yb
        self.Zb = Zb
        self.device = device
        self.model = model
        self.sphere_core = sphere_core
        self.sphere_core_label = sphere_core_label 
        self.traced_seed = []     
        self.track = []   
        self.direction_track = []
        self.confidence_track = []
        self.boundary_location = []
        self.success_location = []
        self.terminated_location = []
        self.soma_location = []
        self.mask_size = mask_size
        self.pt_index = 0 # index of the points
        self.ndlist = [] # final node list
        self.rlist = []
        self.relu = torch.nn.ReLU(inplace=True)
        self.R = Lambda*Lambda
        self.K = K

    def _mask_point(self, nd, radii, index=0):
        n = np.rint(nd).astype(int)
        radii = 2*np.rint(radii).astype(int)
        X, Y, Z = np.meshgrid(
                    constrain_range(n[0] - radii, n[0] + radii + 1, self.n, self.Xb - self.n - 1),
                    constrain_range(n[1] - radii, n[1] + radii + 1, self.n, self.Yb - self.n - 1),
                    constrain_range(n[2] - radii, n[2] + radii + 1, self.n, self.Zb - self.n - 1))
        if index == 0:
            self.indx_map[Z, X, Y] = self.pt_index
        else:
            self.indx_map[Z, X, Y] = index
            
    def trace_JointDecision(self):        
        lent = self.terminations.shape[0]
        
        for i in tqdm(range(lent)):
            position = self.terminations[i, 0:3]
            pix_id = self.indx_map[position[2], position[0], position[1]]
            if pix_id > 0:
                print('Traced Seed')
                self.traced_seed.append(position)
                continue           

            if position[0] <= self.n or position[1] <= self.n or \
               position[2] <= self.n or position[0] >= self.Xb-self.n-1 or \
               position[1] >= self.Yb - self.n - 1 or \
               position[2] >= self.Zb - self.n - 1:
                continue
            
            # SPE for feature extraction
            Spherical_patch = Spherical_Patches_Extraction(self.img2, position,
                                                           self.n, self.sphere_core, 
                                                           self.node_step)
            SP = Spherical_patch.reshape([1, self.Ma, self.Mp, self.n-1]).transpose([0,3,1,2])
            SP = np.asarray(SP)
            pmax = SP.max()
            
            if pmax > 0:
                SP = SP/pmax
            
            data = torch.from_numpy(SP)
            inputs = data.type(torch.FloatTensor).to(self.device)   
            outputs1, stop_flag = self.model(inputs)
            
            outputs = outputs1
            if stop_flag.shape[1] > 2:
                radii = self.relu(stop_flag[:,-1,:,:]).cpu().detach().numpy().squeeze() + 1
                stop_flag = stop_flag[:,:2,:,:]
            else:
                radii = 1
            
            outputs = torch.nn.functional.softmax(outputs,1)
            stop_flag = torch.nn.functional.softmax(stop_flag,1)
            direction_vector = outputs.cpu().detach().numpy().reshape([self.K,1])
            stop_flag = stop_flag.cpu().detach().numpy().squeeze().argmax()
            if stop_flag == 1:
                # skip without adding it to ndlist
                self.terminated_location.append(position)
                print('terminated by discriminater', 0)
                continue

            # determine two initial direction        
            max_id = np.argmax(direction_vector)
            direction1 = self.sphere_core_label[max_id, :]
            
            cos_angle = np.sum(direction1*self.sphere_core_label, axis=1)
            cos_angle[cos_angle>1] = 1
            cos_angle[cos_angle<-1] = -1
            angle = np.arccos(cos_angle).reshape([self.K,1])
            direction_vector[angle<=np.pi/2] = 0
            max_id = np.argmax(direction_vector)            
            direction2 = self.sphere_core_label[max_id, :]
            #direction2 = -direction1.copy()
            _confidence = direction_vector[max_id]
            self.confidence_track.append(_confidence)
            
            soma_reached = self.soma_mask2[position[2], position[0], position[1]]
            
            if soma_reached:
                # skip after adding it to ndlist
                self.soma_location.append(position)
                print('Soma Reached', 0)
                nd = Node(position, _confidence, radii, 1)
                self.ndlist.append(nd)
                self.rlist.append(radii)
                self.pt_index += 1 # current value of pt_index = len(ndlist)
                self.track.append(position)
                continue
            
            nd = Node(position, _confidence, radii)
            self.ndlist.append(nd)
            self.rlist.append(radii)
            self.pt_index += 1 # current value of pt_index = len(ndlist)
            self.track.append(position)
            self.position_id = self.pt_index # used for masking position location after bidirectional tracking                       
            previous_nd_len = len(self.ndlist) # length of ndlist in previous iteration
            
            # trace towards direction1
            track_neg = False
            self._Track_Pos(position, direction1, track_neg, radii)
            
            # trace towards direction2
            position = self.terminations[i, 0:3]
            track_neg = True
            self._Track_Pos(position, direction2, track_neg, radii)
            
            # label the traced branches
            self._mask_point(position, radii, self.position_id)
            len_branch = len(self.ndlist) - previous_nd_len # length of new added branches
            for j in range(len_branch):
                position_m = self.ndlist[previous_nd_len + j].position
                self._mask_point(position_m, self.rlist[previous_nd_len + j], previous_nd_len+j+1) 
            
    def _Track_Pos(self, position, direction1, track_neg, radii):
        cc = 0 # steps counter
        correct_flag = 0
        
        while cc < self.max_iter:
            cc += 1
            if correct_flag == 0: # next_position is determined by seed points if correct_flag==1
                next_position = position + direction1 * np.max([radii, self.step_size])
            correct_flag = 0
            
            if next_position[0]<=self.n or next_position[1]<=self.n or \
               next_position[2]<=self.n or next_position[0]>=self.Xb-self.n-1 or \
               next_position[1]>=self.Yb-self.n-1 or \
               next_position[2]>=self.Zb-self.n-1:
                print('reached boundary', next_position, cc)
                self.boundary_location.append(next_position)
                break
            
            position = next_position.copy()     
            position_1 = np.round(position).astype(int)
            pix_id = self.indx_map[position_1[2], position_1[0], position_1[1]]
            if pix_id > 0:
                print('Meet Traced Region', cc)
                # biuld connection between the met points
                radii = self.rlist[pix_id]
                nd = Node(position, direction1, radii)
                self.ndlist.append(nd)
                self.rlist.append(radii)
                self.pt_index += 1
                
                if track_neg==False:
                    self.ndlist[self.pt_index-2].nbr.append(self.pt_index-1) 
                    self.ndlist[self.pt_index-1].nbr.append(self.pt_index-2)
                else:
                    self.ndlist[self.position_id].nbr.append(self.pt_index-1) 
                    self.ndlist[self.pt_index-1].nbr.append(self.position_id)
                    track_neg = False
                    
                self.ndlist[pix_id-1].nbr.append(self.pt_index-1)
                self.ndlist[self.pt_index-1].nbr.append(pix_id-1)
                break
            
            # SPE for feature extraction
            Spherical_patch = Spherical_Patches_Extraction(self.img2, position, 
                                                           self.n, self.sphere_core, 
                                                           self.node_step)
            SP = Spherical_patch.reshape([1, self.Ma, self.Mp, self.n-1]).transpose([0,3,1,2])
            SP = np.asarray(SP)
            pmax = SP.max()
            
            if pmax > 0:
                SP = SP/pmax
            
            data = torch.from_numpy(SP)
            inputs = data.type(torch.FloatTensor).to(self.device) 
            outputs1, stop_flag = self.model(inputs)
            
            outputs = outputs1
            if stop_flag.shape[1] > 2:
                radii = self.relu(stop_flag[:,-1,:,:]).cpu().detach().numpy().squeeze() + 1
                stop_flag = stop_flag[:,:2,:,:]
            else:
                radii = 1
            
            outputs = torch.nn.functional.softmax(outputs, 1)
            stop_flag = torch.nn.functional.softmax(stop_flag, 1)
            direction_vector = outputs.cpu().detach().numpy().reshape([self.K, 1])
            stop_flag = stop_flag.cpu().detach().numpy().squeeze().argmax()
            if stop_flag == 1:
                self.terminated_location.append(position)
                print('terminated by discriminater', cc)
                break
            
            soma_reached = self.soma_mask2[position_1[2], position_1[0], position_1[1]]
            if soma_reached:
                self.soma_location.append(position)
                print('Soma Reached', cc)
                nd = Node(position, direction_vector.max(), radii, 1)
                self.ndlist.append(nd)
                self.rlist.append(radii)
                self.pt_index += 1
                # biuld connection between the met points
                if track_neg==False:
                    self.ndlist[self.pt_index - 2].nbr.append(self.pt_index - 1) 
                    self.ndlist[self.pt_index - 1].nbr.append(self.pt_index - 2)
                else:
                    self.ndlist[self.position_id].nbr.append(self.pt_index - 1) 
                    self.ndlist[self.pt_index - 1].nbr.append(self.position_id)
                    track_neg = False
                break
            
            cos_angle = np.sum(direction1 * self.sphere_core_label, axis=1)
            cos_angle[cos_angle > 1] = 1
            cos_angle[cos_angle < -1] = -1
            angle = np.arccos(cos_angle).reshape([self.K, 1])
            direction_vector[angle > self.angle_T] = 0
            max_id = np.argmax(direction_vector)
            _confidence = direction_vector[max_id]
            
            nd = Node(position, _confidence, radii)
            self.ndlist.append(nd)
            self.rlist.append(radii)
            self.pt_index += 1
            
            # biuld connection between the met points
            if track_neg==False:
                self.ndlist[self.pt_index - 2].nbr.append(self.pt_index - 1) 
                self.ndlist[self.pt_index - 1].nbr.append(self.pt_index - 2)
            else:
                self.ndlist[self.position_id].nbr.append(self.pt_index - 1) 
                self.ndlist[self.pt_index - 1].nbr.append(self.position_id)
                track_neg = False
            
            self.track.append(position)
             
            self.confidence_track.append(_confidence)
            
            # joint decision
            dist_position_seed = np.sum(np.square(position - self.terminations[:, :3]), axis=1)
            seed_remain = np.where((dist_position_seed < (self.R * radii * radii)) & (dist_position_seed > 1.5))# find the closest seed point to current position
            remain_size = seed_remain[0].size
            if remain_size == 0:
                direction1 = self.sphere_core_label[max_id, :]
            else:
                seed_remain_position = self.terminations[seed_remain[0], :3]
                ds_vectors = (seed_remain_position - position) / np.linalg.norm(seed_remain_position - position, axis=1).reshape([remain_size, 1])# vector from current positon to the closest seed
                ds_cos = np.sum(ds_vectors * direction1, axis=1)
                ds_cos[ds_cos > 1] = 1
                ds_cos[ds_cos < -1] = -1
                ds_angle = np.arccos(ds_cos)
                ds_angle_min = ds_angle.min()
                if ds_angle_min > self.angle_T:
                    direction1 = self.sphere_core_label[max_id, :]
                else:
                    vid = np.argmin(ds_angle)
                    if _confidence * 2 < self.terminations[seed_remain[0][vid], 3] / 255:
                        direction1 = ds_vectors[vid]
                        correct_flag = 1
                        next_position = seed_remain_position[vid, :]
                    else:                    
                        direction1 = self.sphere_core_label[max_id, :]
            
        if cc == self.max_iter:
            self.success_location.append(position)
            
def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))
        