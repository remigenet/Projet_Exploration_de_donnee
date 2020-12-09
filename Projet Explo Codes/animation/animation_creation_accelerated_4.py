#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 00:58:59 2020

@author: rgenet
"""



import networkx as nx
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


def make_edge_dict(df_edges):
    edges=df_edges.values
    my_dict={}
    for iter in range(len(edges)):
        if my_dict.get(edges[iter,0]) is not None:
            current_list=my_dict.get(edges[iter,0])
        else:
            current_list=[]
        my_dict.update({edges[iter,0]: current_list+[edges[iter,1]]})
        if my_dict.get(edges[iter,1]) is not None:
            current_list=my_dict.get(edges[iter,1])
        else:
            current_list=[]
        my_dict.update({edges[iter,1]: current_list+[edges[iter,0]]})
    return my_dict

def make_dict_type(df_target):
    target=df_target.values
    my_dict={}
    for iter in range(np.shape(target)[0]):
        my_dict.update({target[iter,0]: target[iter,3]})
    return my_dict

def get_points_list(edge_dict, start_point, nb_depth_max):
    my_points=[start_point]
    done_point=[]
    current_start=0
    depth=0
    while depth<nb_depth_max:
        nodes_to_do=len(my_points)
        for node_number in range(current_start,nodes_to_do):
            current_start+=1
            if my_points[node_number] not in done_point:
                my_points+=edge_dict.get(my_points[node_number])
                done_point+=[my_points[node_number]]
        my_uniques_nodes=[]
        for node in my_points:
            if node not in my_uniques_nodes:
                my_uniques_nodes.append(node)
        my_points=copy.deepcopy(my_uniques_nodes)
        depth+=1
    return(my_points)

def dict_pos(t):#,h):
    my_dict={}
    pos=0
    for node in t:
        if my_dict.get(node) is None:
            my_dict.update({node: pos})
            pos+=1
#    for node in h:
#        if my_dict.get(node) is None:
#            my_dict.update({node: pos})
#            pos+=1    
    return my_dict

df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")
edge_dict=make_edge_dict(df_edges)
type_dict=make_dict_type(df_target)
nb_point=len(df_target['id'].values)
mode='size_by_depth'
depth_max=2
points_taken=list(map(lambda x: np.random.choice(df_target.shape[0]), range(nb_point)))
first_point=np.random.choice(df_target.shape[0])
edge_color_dict={1: 'black', 2: 'dimgray', 3: 'gray', 4: 'darkgray',5:'silver', 6:'lightgray',7:'whitesmoke',8:'azure' }
t=[]
#for depth_max in range(1,2):
depth_max=0
my_point_list=[]
while len(t)< nb_point and depth_max<8:
    t=get_points_list(edge_dict, first_point, depth_max)
    my_point_list.append(t)
    depth_max+=1

for iter in range(len(my_point_list)-1,0,-1):
    temp=[]
    for point in my_point_list[iter]:
        if point not in my_point_list[iter-1]:
            temp.append(point)
    my_point_list[iter]=temp


    
t=[]
point_size=[]
point_part=[]
for depth in range(len(my_point_list)):
    t+=my_point_list[depth]
    if mode=='size_by_depth':
        for point in range(len(my_point_list[depth])):
            point_size.append(2*len(my_point_list)/(depth+1))
            point_part.append(depth+1)

for iter_2 in range(len(point_size)):
    point_size[iter_2]=5*point_size[iter_2]/np.sqrt(len(t))
    
        
    
pos_dict=dict_pos(t)

color_map=[]
for node in t:
    if type_dict.get(node)=='government':
        color_map.append('blue')
    elif type_dict.get(node)=='politician':
        color_map.append('red')
    elif type_dict.get(node)=='company':
        color_map.append('black')
    else:
        color_map.append('green')
    

    
#ed_ls = [(x,y) for x,y in zip(t, h)]
G = nx.Graph()
#G.add_edges_from(ed_ls)
G.add_nodes_from(t)

edge_color_map=[]
edge_size_map=[]
list_start=[]
list_reached=[]
node_number=0
for start_node in t:
    t_edges=edge_dict.get(start_node)
    for node_reached in t_edges:
        if node_reached in t[:node_number]:
            G.add_edge(start_node, node_reached)
            edge_color_map.append(edge_color_dict.get(point_part[node_number]))
            edge_size_map.append(point_size[node_number]/15)
    node_number+=1


        
#G.add_nodes_from(h)
graph_pos = nx.spring_layout(G, dim=3)
xyz = np.array([graph_pos[v] for v in G])
print(xyz.shape)
    
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
edge_number=0
not_done=True
iter=0
angle=0
while not_done: 
    print(iter)
    nb_points_added=min((point_part[min(iter, len(point_part)-1)]**2), len(t)-iter)
    ax.scatter(xyz[iter:iter+nb_points_added,0], xyz[iter:iter+nb_points_added,1], xyz[iter:iter+nb_points_added,2], s=point_size[iter:iter+nb_points_added], c=color_map[iter:iter+nb_points_added])
    list_start=[]
    list_reached=[]
    count=0
    for start_node in t[iter:iter+nb_points_added]:
        t_edges=edge_dict.get(start_node)
        for node_reached in t_edges:
            temp_1=[]
            temp_2=[]
            if node_reached in t[:iter+count]:
                for col in range(3):
                    temp_1.append(xyz[pos_dict.get(start_node),col])
                    temp_2.append(xyz[pos_dict.get(node_reached),col])
                list_start.append(temp_1)
                list_reached.append(temp_2)
        count+=1
    
    my_edges=np.zeros((len(list_reached), 6))
    for iter_2 in range(len(list_start)):
        for col in range(3):
            my_edges[iter_2,2*col]=list_start[iter_2][col]
            my_edges[iter_2,2*col+1]=list_reached[iter_2][col]
    
    for i in range(my_edges.shape[0]):
        ax.plot(my_edges[i,0:2], my_edges[i,2:4],
                    my_edges[i,4:6],linewidth=edge_size_map[i], color=edge_color_map[edge_number])#, color=(0.2, 1 - 0.1 * i, 0.8))
        edge_number+=1

    ax.view_init(elev=10., azim=angle*3)
    if iter<10:
        plt.savefig("/tmp/rgenet/image_animation/imgax0000"+str(iter)+".png")
    elif iter<100:
        plt.savefig("/tmp/rgenet/image_animation/imgax000"+str(iter)+".png")
    elif iter<1000:
        plt.savefig("/tmp/rgenet/image_animation/imgax00"+str(iter)+".png")
    elif iter<10000:
        plt.savefig("/tmp/rgenet/image_animation/imgax0"+str(iter)+".png")
    else:
        plt.savefig("/tmp/rgenet/image_animation/imgax"+str(iter)+".png")
    angle+=1
    iter+=nb_points_added
    if iter>=len(point_part)-1:
        not_done=False
        
image_folder = '/tmp/rgenet/image_animation/'
video_name = 'my_animationspeedspeed3.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()