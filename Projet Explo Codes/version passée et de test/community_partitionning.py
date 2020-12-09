# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:11:11 2020

@author: remi
"""


import networkx as nx
import scipy.sparse as sparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import community
import matplotlib.cm as cm

df_edges=pd.read_csv("musae_facebook_edges.csv")

nb_point=1000
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
    
partition = community.best_partition(G)
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,  cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Community partitionning using community louvain algorithm")
plt.savefig("community_partitionning based on community louvain full 1000 edges ")
plt.show()

options = { "node_color": color_map, "node_size": 50, "linewidths": 0.1, "width": 0.1,}
    #"node_color": color_map,

print("here")
nx.draw(G, **options)
plt.savefig("graph of the dataset using networkX 1000  edges")
plt.show()