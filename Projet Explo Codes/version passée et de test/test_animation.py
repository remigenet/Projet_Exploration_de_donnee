# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:19:13 2020

@author: remi
"""

import networkx as nx
import numpy as np
from mayavi import mlab
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
depth_max=2
points_taken=list(map(lambda x: np.random.choice(df_target.shape[0]), range(nb_point)))
first_point=np.random.choice(df_target.shape[0])
t=[]
#for depth_max in range(1,2):
depth_max=0
while len(t)< nb_point:
    t=get_points_list(edge_dict, first_point, depth_max)
    depth_max+=1
    #t = df_target['id'].values[first_point]
    #h = df_edges['id_2'].values[points_taken]
    

#    
#    
#    
#    pos_dict=dict_pos(t)
#    
#    color_map=[]
#    for node in t:
#        if type_dict.get(node)=='government':
#            color_map.append('blue')
#        elif type_dict.get(node)=='politician':
#            color_map.append('red')
#        elif type_dict.get(node)=='company':
#            color_map.append('black')
#        else:
#            color_map.append('green')
#    
#    
#        
#    #ed_ls = [(x,y) for x,y in zip(t, h)]
#    G = nx.Graph()
#    #G.add_edges_from(ed_ls)
#    G.add_nodes_from(t)
#    
#    
#    
#    list_start=[]
#    list_reached=[]
#    for start_node in t:
#        t_edges=edge_dict.get(start_node)
#        for node_reached in t_edges:
#            if node_reached in t:
#                G.add_edge(start_node, node_reached)
#    
#    
#    
#            
#    #G.add_nodes_from(h)
#    graph_pos = nx.spring_layout(G, dim=3)
#    xyz = np.array([graph_pos[v] for v in G])
#    print(xyz.shape)
#    
#    fig = plt.figure(figsize=(20,10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_axis_off()
#    # make the panes transparent
#    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#    # make the grid lines transparent
#    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
#    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=20/np.sqrt(len(t)), c=color_map)
#    list_start=[]
#    list_reached=[]
#    for start_node in t:
#        t_edges=edge_dict.get(start_node)
#        for node_reached in t_edges:
#            temp_1=[]
#            temp_2=[]
#            if node_reached in t:
#                for col in range(3):
#                    temp_1.append(xyz[pos_dict.get(start_node),col])
#                    temp_2.append(xyz[pos_dict.get(node_reached),col])
#                list_start.append(temp_1)
#                list_reached.append(temp_2)
#    
#    my_edges=np.zeros((len(list_reached), 6))
#    for iter in range(len(list_start)):
#        for col in range(3):
#            my_edges[iter,2*col]=list_start[iter][col]
#            my_edges[iter,2*col+1]=list_reached[iter][col]
#    
#    for i in range(my_edges.shape[0]):
#        ax.plot(my_edges[i,0:2], my_edges[i,2:4],
#                    my_edges[i,4:6],linewidth=0.5/np.sqrt(len(list_reached)), color='black')#, color=(0.2, 1 - 0.1 * i, 0.8))
#        
#    for ii in range(0,360,1):
#        ax.view_init(elev=10., azim=ii)
#        if ii<10:
#            plt.savefig("C:/Users/remi/Desktop/Projet Exploration grands volume de donne/image_animation/imgax"+str(depth_max)+"number00"+str(ii)+".png")
#        elif ii<100:
#            plt.savefig("C:/Users/remi/Desktop/Projet Exploration grands volume de donne/image_animation/imgax"+str(depth_max)+"number0"+str(ii)+".png")
#        else:
#            plt.savefig("C:/Users/remi/Desktop/Projet Exploration grands volume de donne/image_animation/imgax"+str(depth_max)+"number"+str(ii)+".png")
#            
#            
#image_folder = 'C:/Users/remi/Desktop/Projet Exploration grands volume de donne/image_animation/'
#video_name = 'my_animation.avi'
#
#images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#images.sort()
#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
#
#video = cv2.VideoWriter(video_name, 0, 60, (width,height))
#
#for image in images:
#    video.write(cv2.imread(os.path.join(image_folder, image)))
#
#cv2.destroyAllWindows()
#video.release()