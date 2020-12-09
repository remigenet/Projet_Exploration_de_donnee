# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:26:46 2020

@author: remi
"""

import networkx as nx
import numpy as np
from mayavi import mlab
import pandas as pd

# some graphs to try
# H=nx.krackhardt_kite_graph()
#H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
#H=nx.grid_2d_graph(4,5)
#H = nx.cycle_graph(20)


df_edges=pd.read_csv("musae_facebook_edges.csv")

nb_point=50
points_taken=list(map(lambda x: np.random.choice(df_edges.shape[0]), range(nb_point)))
row = df_edges['id_1'].values[points_taken]
col = df_edges['id_2'].values[points_taken]
data = np.ones_like(row)

df_target=pd.read_csv("musae_facebook_target.csv")
node_types=list(df_target[['id','page_type']].values)
dict_node_types={}
for iter in range(len(node_types)):
    dict_node_types.update({node_types[iter][0]: node_types[iter][1]})

#Y = sparse.csr_matrix((data, (row, row)))
#G=nx.from_scipy_sparse_matrix(Y[:50,:50], parallel_edges=False, create_using=None)#, edge_attribute='weight')



H = nx.Graph()
H.add_nodes_from(row, page_type=df_target[df_target['id'].map(lambda x: True if x in row else False)]['page_type'].values)
color_map=[]
done_nodes=[]
for node in row:
    if not node in done_nodes:
        if dict_node_types.get(node)=='governmental':
            color_map.append('blue')
        elif dict_node_types.get(node)=='politicians':
            color_map.append('red')
        elif dict_node_types.get(node)=='companies':
            color_map.append('black')
        else:
            color_map.append('green')
        done_nodes.append(node)
        
H.add_nodes_from(col, page_type=df_target[df_target['id'].map(lambda x: True if x in col else False)]['page_type'].values)
for node in col:
    if node not in done_nodes:
        if dict_node_types.get(node)=='government':
            color_map.append('blue')
        elif dict_node_types.get(node)=='politician':
            color_map.append('red')
        elif dict_node_types.get(node)=='company':
            color_map.append('black')
        else:
            color_map.append('green')
        done_nodes.append(node)
            
for iter in range(len(row)):
    H.add_edge(row[iter],col[iter])

# reorder nodes from 0,len(G)-1
G = nx.convert_node_labels_to_integers(H)
#G=H
# 3d spring layout
pos = nx.spring_layout(G, dim=3)
# numpy array of x,y,z positions in sorted node order
xyz = np.array([pos[v] for v in sorted(G)])
# scalar colors
scalars = np.array(list(G.nodes())) + 5

pts = mlab.points3d(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    scalars,
    scale_factor=0.1,
    scale_mode="none",
    colormap="Blues",
    resolution=20,
)

pts.mlab_source.dataset.lines = np.array(list(G.edges()))
pts.update()
tube = mlab.pipeline.tube(pts, tube_radius=0.01)
mlab.pipeline.surface(tube, color=(0.4, 0.8, 0.9))
mlab.show()