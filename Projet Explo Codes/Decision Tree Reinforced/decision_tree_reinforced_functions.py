#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:53:39 2020

@author: rgenet
"""

import numpy as np
import networkx as nx
import community 
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def get_number_link(id, df_know_infos, df_train_edges, page_types, link_depth):
    temp_df=df_train_edges[df_train_edges['id_2'].isin([id])]['id_1'].append(df_train_edges[df_train_edges['id_1'].isin([id])]['id_2'])
    for iter in range(link_depth):
        temp_df=df_train_edges[df_train_edges['id_2'].isin(temp_df)]['id_1'].append(df_train_edges[df_train_edges['id_1'].isin(temp_df)]['id_2'])
    know_links=df_know_infos[df_know_infos['id'].isin(temp_df)]
    dict={}
    nb_link=know_links.shape[0]
    dict.update({'nb_link': nb_link})
    if nb_link>0:
        for page_type in page_types:
            dict.update({page_type :know_links[know_links['page_type']==page_type]['id'].count()/nb_link})
    else:
        for page_type in page_types:
            dict.update({page_type : 0})
    
    return dict

def get_features_ratios(id,features):
    my_dict={}
    my_feature=features.get(str(id))
    my_dict.update({'mean' : np.mean(my_feature)})
    my_dict.update({'std' : np.std(my_feature)})
    my_dict.update({'sharpe' :  np.mean(my_feature)/np.std(my_feature)})
    my_dict.update({'nb_obs' :  len(my_feature)})
    my_dict.update({'sum' :  sum(my_feature)})
    return my_dict

def create_X(page_types, df_know_infos, df_edges, features,X_ids,partition_dict ):
    
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*4+10))
    for iter in range(nb_ids):
        id=X_ids[iter]
        direct_link=get_number_link(id, df_know_infos, df_edges, page_types, 0)
        second_link=get_number_link(id, df_know_infos, df_edges, page_types, 1)
        third_link=get_number_link(id, df_know_infos, df_edges, page_types, 2)
        fourth_link=get_number_link(id, df_know_infos, df_edges, page_types, 3)
        id_features=get_features_ratios(id, features)
        id_partition=partition_dict.get(id)
        row_values=[id_partition]+list(direct_link.values())+list(second_link.values())+list(third_link.values())+list(fourth_link.values())+list(id_features.values())
        X[iter,:]=row_values
    return X

def create_X_with_label_propagate(page_types, df_know_infos, df_edges, features,X_ids,partition_dict, result_propagation ):
    
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*4+14))
    for iter in range(nb_ids):
        id=X_ids[iter]
        direct_link=get_number_link(id, df_know_infos, df_edges, page_types, 0)
        second_link=get_number_link(id, df_know_infos, df_edges, page_types, 1)
        third_link=get_number_link(id, df_know_infos, df_edges, page_types, 2)
        fourth_link=get_number_link(id, df_know_infos, df_edges, page_types, 3)
        id_features=get_features_ratios(id, features)
        id_partition=partition_dict.get(id)
        row_values=[id_partition]+list(direct_link.values())+list(second_link.values())+list(third_link.values())+list(fourth_link.values())+list(id_features.values())+result_propagation.get(id)
        X[iter,:]=row_values
    return X

def create_X_reduced(page_types, df_know_infos, df_edges, features,X_ids ):
    
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*3))
    for iter in range(nb_ids):
        id=X_ids[iter]
        direct_link=get_number_link_reduced(id, df_know_infos, df_edges, page_types, 0)
        second_link=get_number_link_reduced(id, df_know_infos, df_edges, page_types, 1)
        third_link=get_number_link_reduced(id, df_know_infos, df_edges, page_types, 2)
        row_values=list(direct_link.values())+list(second_link.values())+list(third_link.values())
        X[iter,:]=row_values
    return X

def get_number_link_reduced(id, df_know_infos, df_train_edges, page_types, link_depth):
    temp_df=df_train_edges[df_train_edges['id_2'].isin([id])]['id_1'].append(df_train_edges[df_train_edges['id_1'].isin([id])]['id_2'])
    for iter in range(link_depth):
        temp_df=df_train_edges[df_train_edges['id_2'].isin(temp_df)]['id_1'].append(df_train_edges[df_train_edges['id_1'].isin(temp_df)]['id_2'])
    know_links=df_know_infos[df_know_infos['id'].isin(temp_df)]
    dict={}
    nb_link=know_links.shape[0]
    if nb_link>0:
        for page_type in page_types:
            dict.update({page_type :know_links[know_links['page_type']==page_type]['id'].count()/nb_link})
    else:
        for page_type in page_types:
            dict.update({page_type : 0})
    
    return dict

def get_partitions(X_ids, df_edges):
    G = nx.Graph()
    G.add_nodes_from(X_ids)
    ids_1 = df_edges['id_1'].values
    ids_2 = df_edges['id_2'].values
    for iter in range(len(ids_1)):
        G.add_edge(ids_1[iter],ids_2[iter])
    return(community.best_partition(G))
    
def create_trees(nb_iter,df_know_infos,df_train_target, partition_dict, features,page_types, df_edges,create_X_func):
    
    parameters = {'max_depth':range(1,23), 'criterion' :['gini', 'entropy'],'splitter': ['best','random'], 'min_samples_split' : range(3,6),'min_samples_leaf': range(1,5), 'min_impurity_decrease': [i/10 for i in range(40)] }
    known_informations=copy.deepcopy(df_know_infos)
    Y_train=df_train_target['page_type'].values
    my_decision_trees=[]
    for iter in range(nb_iter):
        print("starting iteration "+str(iter)+" on training part")
        X_ids_train=df_train_target['id'].values
        
    
        X_train=create_X_func( page_types, known_informations, df_edges,features,X_ids_train,partition_dict)
        
        
        clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
        clf.fit(X=X_train, y=Y_train)
        tree_model = clf.best_estimator_
        print (clf.best_score_, clf.best_params_) 
        
        new_informations=tree_model.predict(X_train)
        known_informations=df_know_infos.append(pd.DataFrame({'id':X_ids_train, 'page_type':new_informations}))
        my_decision_trees.append(tree_model)
    return my_decision_trees

def use_trees(my_decision_trees, test_ids, df_edges, partition_dict,df_know_infos, features,page_types,create_X_func):
    
    known_informations=copy.deepcopy(df_know_infos)

    for decision_tree in my_decision_trees:
        
        X_test=create_X_func(page_types, known_informations, df_edges,features,test_ids,partition_dict)
        
        new_informations=decision_tree.predict(X_test)
        known_informations=df_know_infos.append(pd.DataFrame({'id':test_ids, 'page_type':new_informations}))
    
    return new_informations


def create_trees_with_label_propagate(nb_iter,df_know_infos,df_train_target, partition_dict, features,page_types, df_edges,create_X_func, result_propagation):
    
    parameters = {'max_depth':range(1,23), 'criterion' :['gini', 'entropy'],'splitter': ['best','random'], 'min_samples_split' : range(3,6),'min_samples_leaf': range(1,5), 'min_impurity_decrease': [i/10 for i in range(40)] }
    known_informations=copy.deepcopy(df_know_infos)
    Y_train=df_train_target['page_type'].values
    my_decision_trees=[]
    for iter in range(nb_iter):
        print("starting iteration "+str(iter)+" on training part")
        X_ids_train=df_train_target['id'].values
        
    
        X_train=create_X_func( page_types, known_informations, df_edges,features,X_ids_train,partition_dict,result_propagation)
        
        
        clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
        clf.fit(X=X_train, y=Y_train)
        tree_model = clf.best_estimator_
        print (clf.best_score_, clf.best_params_) 
        
        new_informations=tree_model.predict(X_train)
        known_informations=df_know_infos.append(pd.DataFrame({'id':X_ids_train, 'page_type':new_informations}),sort=False)
        my_decision_trees.append(tree_model)
    return my_decision_trees

def use_trees_with_label_propagate(my_decision_trees, test_ids, df_edges, partition_dict,df_know_infos, features,page_types,create_X_func,result_propagation):
    
    known_informations=copy.deepcopy(df_know_infos)

    for decision_tree in my_decision_trees:
        
        X_test=create_X_func(page_types, known_informations, df_edges,features,test_ids,partition_dict,result_propagation)
        
        new_informations=decision_tree.predict(X_test)
        known_informations=df_know_infos.append(pd.DataFrame({'id':test_ids, 'page_type':new_informations}),sort=False)
    
    return new_informations