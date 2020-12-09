# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:41:44 2020

@author: remi
"""

import networkx as nx
import scipy.sparse as sparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random

df_edges=pd.read_csv("musae_facebook_edges.csv")

nb_point=100
points_taken=list(map(lambda x: np.random.choice(2000), range(nb_point)))
row = df_edges['id_1'].values
col = df_edges['id_2'].values
data = np.ones_like(row)

df_target=pd.read_csv("musae_facebook_target.csv")
node_types=list(df_target[['id','page_type']].values)
dict_node_types={}
for iter in range(len(node_types)):
    dict_node_types.update({node_types[iter][0]: node_types[iter][1]})

#Y = sparse.csr_matrix((data, (row, row)))
#G=nx.from_scipy_sparse_matrix(Y[:50,:50], parallel_edges=False, create_using=None)#, edge_attribute='weight')



G = nx.Graph()
G.add_nodes_from(row)
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
        
G.add_nodes_from(col)
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
    G.add_edge(row[iter],col[iter])
    
# write edgelist to grid.edgelist
#nx.write_edgelist(G, path="grid.edgelist", delimiter=",")
# read edgelist from grid.edgelist
#H = nx.read_edgelist(path="grid.edgelist", delimiter=",")
options = { "node_color": color_map, "node_size": 50, "linewidths": 0.1, "width": 0.1,}
    #"node_color": color_map,

print("here")
nx.draw(G, **options)
plt.show()

edge_subset = random.sample(G.edges(), int(0.25 * G.number_of_edges()))
G_train = G.copy()
G_train.remove_edges_from(edge_subset)

prediction_jaccard=list(nx.jaccard_coefficient(G_train))
score, label=zip(*[(s,(u,v) in edge_subset) for (u,v,s) in prediction_jaccard])
#nx.draw_circular(G,node_color=color_map, with_labels=False)
#plt.show()