#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.io1 import saveswc
from tqdm import tqdm

class Node(object):
   def __init__(self, position, conf, radius, node_type=3):
       self.position = position
       self.conf = conf
       self.radius = radius
       self.nbr = []
       self.node_type = node_type

def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1

def compute_trees(n0):
    n0_size = len(n0)
    treecnt = 0
    q = [] # bfs queue
    n1 = []
    dist = np.ones([n0_size,1], dtype=np.int)*100000
    nmap = np.ones([n0_size,1], dtype=np.int)*-1 # index in output tree n1
    parent = np.ones([n0_size,1], dtype=np.int)*-1 # parent index in current tree n0
    print('Search for Soma')
    for i in range(n0_size):
        if n0[i].node_type==1:
            q.append(i)
            dist[i] = 0
            nmap[i] = -1
            parent[i] = -1
    # BFS
    while len(q)>0:
        curr = q.pop(0)    
        n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt+2)
        if parent[curr]>0:
            n.nbr.append(nmap[parent[curr]])
        n1.append(n)
        nmap[curr] = len(n1)
        for j in range(len(n0[curr].nbr)):
            adj = n0[curr].nbr[j]
            if dist[adj] == 100000:
                dist[adj] = dist[curr] + 1
                parent[adj] = curr
                q.append(adj)
                
    while ((get_undiscover(dist))>=0):
        treecnt += 1
        seed = get_undiscover(dist)
        dist[seed] = 0
        nmap[seed] = -1
        parent[seed] = -1
        q.append(seed)
        while len(q)>0:
            curr = q.pop(0)    
            n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt+2)
            if parent[curr]>0:
                n.nbr.append(nmap[parent[curr]])
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    q.append(adj)      
    return n1

def build_nodelist(tree):
    _data = np.zeros((1, 7))
    cnt_recnodes = 0
    for i in range(len(tree)):
        if len(tree[i].nbr)==0:
            cnt_recnodes += 1
            pid = -1
            new_node = np.asarray([cnt_recnodes, 
                        tree[i].node_type, 
                        tree[i].position[1], 
                        tree[i].position[0], 
                        tree[i].position[2], 
                        tree[i].radius, 
                        pid])
            _data = np.vstack((_data, new_node))
            
        else:
            for j in range(len(tree[i].nbr)):
                cnt_recnodes += 1
                pid = tree[i].nbr[j].squeeze()
                new_node = np.asarray([cnt_recnodes, 
                        tree[i].node_type, 
                        tree[i].position[1], 
                        tree[i].position[0], 
                        tree[i].position[2], 
                        tree[i].radius, 
                        pid])
                _data = np.vstack((_data, new_node))
    _data = _data[1:,:]
    return _data

def local_max(Im, wsize=3, thre=255*0.5):
    nZ, nY, nX = Im.shape
    suppress = np.zeros_like(Im)
    # thre =0.01*255
    potential_points = np.where(Im>thre)
    num_points = len(potential_points[0])
    coordinates = []
    for i in tqdm(range(num_points)):
        z = potential_points[0][i]
        y = potential_points[1][i]
        x = potential_points[2][i]
        if wsize == 3:
            if x < 1 or y < 1 or z < 1 or x > nX-2 or y > nY-2 or z > nZ-2:
                continue
            
            img_patch = Im[z-1:z+2,y-1:y+2,x-1:x+2]
            if img_patch.max() == img_patch[1,1,1]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])
        if wsize == 5:
            if x < 2 or y < 2 or z < 2 or x > nX-3 or y > nY-3 or z > nZ-3:
                continue
            
            img_patch = Im[z-2:z+3,y-2:y+3,x-2:x+3]
            if img_patch.max() == img_patch[2,2,2]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])
                
        if wsize == 7:
            if x < 3 or y < 3 or z < 3 or x > nX-4 or y > nY-4 or z > nZ-4:
                continue
            
            img_patch = Im[z-3:z+4,y-3:y+4,x-3:x+4]
            if img_patch.max() == img_patch[3,3,3]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])            
    return suppress, coordinates

def Spherical_Patches_Extraction(img2, position, n, sphere_core, node_step=1):

    x = position[0]
    y = position[1]
    z = position[2]
    radius = 0
    j=np.arange(radius+1,n*node_step+radius+1,node_step).reshape(-1,n)
    ray_x = x+(sphere_core[:,0].reshape(-1,1))*j
    ray_y = y+(sphere_core[:,1].reshape(-1,1))*j
    ray_z = z+(sphere_core[:,2].reshape(-1,1))*j
    
    
    Rray_x=np.rint(ray_x).astype(int)
    Rray_y=np.rint(ray_y).astype(int)
    Rray_z=np.rint(ray_z).astype(int)

    Spherical_patch_temp = img2[Rray_z,Rray_x,Rray_y]
    Spherical_patch = Spherical_patch_temp[:,1:n]
    
    return Spherical_patch

def savemarker(filepath,marker):    
    with open(filepath, 'w') as f:
        for i in range(marker.shape[0]):
            markerp=[marker[i,1],marker[i,0],marker[i,2],0,1,' ',' ']        
            print('%.3f, %.3f, %.3f, %d, %d, %s, %s'  %  (markerp[0], markerp[1], markerp[2], markerp[3],markerp[4], markerp[5], markerp[6]),file=f)
                     
def generate_sphere(Ma,Mp):
    #generate 3d sphere
    m1=np.arange(1,Ma+1,1).reshape(-1,Ma)
    m2=np.arange(1,Mp+1,1).reshape(-1,Mp)
    alpha=2*(np.pi)*m1/Ma
    phi=-(np.arccos(2*m2/(Mp+1)-1)-(np.pi))
    xm=(np.cos(alpha).reshape(Ma,1))*np.sin(phi)
    ym=(np.sin(alpha).reshape(Ma,1))*np.sin(phi)
    zm=np.cos(phi)
    zm=np.tile(zm,(Mp,1))
    sphere_core=np.concatenate([xm.reshape(-1,1), ym.reshape(-1,1), zm.reshape(-1,1)],axis=1) #y_axis=alpha[0:Ma],x_axis=phi[0:Mp]
    return sphere_core, alpha, phi

def soma_point(soma_img, distance_transform):

    maxr = np.max(distance_transform)
    print(maxr)
    position = np.argwhere(distance_transform == maxr)
    soma_point = np.argwhere(soma_img >= 200)
    temp = position[0] - soma_point
    tmp1 = np.linalg.norm(temp, axis=1)
    radius = np.max(tmp1)
    print(radius)
    return position, radius

def inSphere(x, y, z,ball_center_x, ball_center_y, ball_center_z, radius):

    dist = (x - ball_center_x) ** 2 + (y - ball_center_y) ** 2 + (z - ball_center_z) ** 2
    return dist < (radius ** 2)

def connected(soma_img, img_tif, singletree_swc, swc_path, distance_transform):

    swc_connected = np.copy(singletree_swc)
    # find soma center and soma radius
    ballcenter_candidate, radius = soma_point(soma_img, distance_transform)
    ball_center_x = ballcenter_candidate[0, 2]
    ball_center_y = ballcenter_candidate[0, 1]
    ball_center_z = ballcenter_candidate[0, 0]

    swc_connected[:, 0] = swc_connected[:, 0] + 1
    swc_connected[:, 6] = swc_connected[:, 6] + 1

    for i in range(len(singletree_swc)):
        if singletree_swc[i, 6] == -1:
            if i+1 in singletree_swc[:, 6]:
                if inSphere(singletree_swc[i, 2], singletree_swc[i, 3], singletree_swc[i, 4], ball_center_x, ball_center_y, ball_center_z, radius+30):
                    swc_connected[i, 6] = 1
                    swc_connected[i, 1] = 2
                else:
                    swc_connected[i, 6] = -1

    first_row = np.array([1, 2, ball_center_x, ball_center_y, ball_center_z, radius, -1])
    swc_connected = np.insert(swc_connected, 0, values=first_row, axis=0)

    a = []
    for j in range(len(swc_connected)):
        if swc_connected[j, 6] == 0:
            a.append(j)
    swc_connected = np.delete(swc_connected, a, axis=0)

    save_swc_path1 = swc_path + '_connected_soma.swc'
    saveswc(save_swc_path1, swc_connected)
    print('--------')
