# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:23:28 2020

@author: remi
"""

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import json
import time
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import copy
import gc

t1=time.time()


def get_features_ratios(id,features):
    my_dict={}
    my_feature=features.get(str(id))
    my_dict.update({'mean' : np.mean(my_feature)})
    my_dict.update({'std' : np.std(my_feature)})
    my_dict.update({'sharpe' :  np.mean(my_feature)/np.std(my_feature)})
    return my_dict

def get_link_dict(df_know_infos, df_edges, page_types, depth,X_ids):
    my_dict={}
    for id in X_ids:    
        my_dict.update({id:[list(df_edges[df_edges['id_2'].isin([id])]['id_1'].append(df_edges[df_edges['id_1'].isin([id])]['id_2']).values)]})
    dict_known_infos={}
    known_infos=list(df_know_infos.values)
    for iter in range(len(known_infos)):
        dict_known_infos.update({str(known_infos[iter][0]): known_infos[iter][1]})
    for iter in range(depth):
        for id in X_ids:    
            link_list=[]
            current_list=my_dict.get(id)[iter]
            for linked_id in current_list:
                val=my_dict.get(linked_id)
                if val is not None:
                    link_list+=val[iter]
            my_dict.get(id).append(link_list)
    page_dict={}
    for iter in range(len(page_types)):
        page_dict.update({page_types[iter] : iter})
    final_dict={}
    for id in X_ids:
        total_list=[]
        for current_depth in range(depth+1):
            count=0
            page_occurence_list=[]
            for page_type in page_types:
                page_occurence_list.append(0)
                
            for link_id in my_dict.get(id)[current_depth]:
                val=dict_known_infos.get(link_id)
                if val is not None:
                    count+=1
                    page_occurence_list[page_dict.get(val)]+=1
            page_occurence_list=list(map(lambda x : x/count if count>0 else 0, page_occurence_list))
            total_list+= [count]+page_occurence_list
        final_dict.update({id:total_list})
    my_dict.clear()
    gc.collect()
    return final_dict

def create_X(df_target,page_types, df_know_infos, df_edges, features, depth ):
    X_ids=df_target['id'].values
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*3+6))
    dict_link=get_link_dict(df_know_infos, df_train_edges, page_types, depth, X_ids)
    for iter in range(nb_ids):
        id=X_ids[iter]
        id_features=get_features_ratios(id, features)
        row_values=dict_link.get(id)+list(id_features.values())
        X[iter,:]=row_values
    return X


df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")
features=json.load(open("musae_facebook_features.json"))
nb_pages=df_target.shape[0] 
page_types=df_target['page_type'].unique()

df_know_infos=df_target[df_target['id']<nb_pages*0.05][['id','page_type']]

df_train_edges=df_edges[df_edges['id_1']<nb_pages*0.7]
df_train_target=df_target[df_target['id'].map(lambda x: True if x<0.7*nb_pages and x>0.05*nb_pages else False)==True]
df_test_edges=df_edges
df_test_target=df_target[df_target['id'].map(lambda x: True if x>0.7*nb_pages else False)==True]

Y_train=df_train_target['page_type'].values
Y_test=df_test_target['page_type'].values


t1=time.time()
X_train=create_X(df_train_target, page_types, df_know_infos, df_edges,features,2)
print(time.time()-t1)
#nb_iter=3
#initial_known_informations=copy.deepcopy(df_know_infos)
#for iter in range(nb_iter):
#    print("starting iteration "+str(iter)+" on training part")
#    X_ids_train=df_train_target['id'].values
#    X_ids_train_test=df_train_test_target['id'].values
#    my_decision_trees=[]
#
#    X_train=create_X(df_train_target, page_types, df_know_infos, df_edges,features,X_ids_train)
#    X_train_test=create_X(df_train_test_target, page_types, df_know_infos, df_edges,features,X_ids_train_test)
#
#    max_accuracy=0
#    best_depth=0
#    for depth in range(1,15):
#        decision_tree = DecisionTreeClassifier(random_state=0, max_depth=depth)
#        decision_tree = decision_tree.fit(X_train, Y_train)
#        predict=decision_tree.predict(X_train_test)
#        accuracy=metrics.accuracy_score(Y_train_test,predict)
#        if accuracy>max_accuracy:
#            max_accuracy=accuracy
#            best_depth=depth
#    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=best_depth)
#    decision_tree = decision_tree.fit(X_train, Y_train)
#    my_decision_trees.append(decision_tree)
#    
#    new_informations=decision_tree.predict(X_train)
#    df_know_infos=initial_known_informations.append(pd.DataFrame({'id':X_ids_train, 'page_type':new_informations}))
#    
#df_know_infos=copy.deepcopy(initial_known_informations)
#print("starting on test set")
#for decision_tree in my_decision_trees:
#    X_ids_test=df_test_target['id'].values
#    
#    X_test=create_X(df_test_target, page_types, df_know_infos, df_edges,features,X_ids_test)
#    
#    new_informations=decision_tree.predict(X_test)
#    df_know_infos=initial_known_informations.append(pd.DataFrame({'id':X_ids_test, 'page_type':new_informations}))
#    
#    print("current classification report:")
#    classification_report=metrics.classification_report(Y_test,new_informations)
#    print(classification_report)
#    
#for decision_tree in my_decision_trees:
#    
#    new_informations=decision_tree.predict(X_test)
#    accuracy=metrics.accuracy_score(Y_test,new_informations)
#    classification_report=metrics.classification_report(Y_test,new_informations)
#    print(classification_report)
#    
#confusion_matrix=metrics.confusion_matrix(Y_test, new_informations)
#print(confusion_matrix)

#print("elapsed time "+str(time.time()-t1)+" seconds")
