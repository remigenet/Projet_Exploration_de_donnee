#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:52:38 2020

@author: rgenet
"""

import decision_tree_reinforced_functions as dtr_func
from sklearn import metrics

class decisition_tree_reinforced:
    
    def __init__(self,df_edges, nb_iter, df_know_infos, features, create_X_func):
        self.df_edges=df_edges
        self.nb_iter=nb_iter
        self.df_know_infos=df_know_infos
        self.features=features
        self.create_X_func=create_X_func
        
    def fit(self,df_train_target):
        X_ids=list(self.df_edges['id_1'].unique())
        X_ids+=[id for id in self.df_edges['id_2'].unique() if id not in X_ids]
        self.page_types=df_train_target['page_type'].unique()
        self.partition_dict=dtr_func.get_partitions(X_ids, self.df_edges)
        
        self.models=dtr_func.create_trees(self.nb_iter,self.df_know_infos,df_train_target, self.partition_dict, self.features,self.page_types, self.df_edges,self.create_X_func)
        
    def predict(self, test_ids):
        self.test_ids=test_ids
        self.predictions=dtr_func.use_trees(self.models, test_ids, self.df_edges, self.partition_dict,self.df_know_infos, self.features,self.page_types,self.create_X_func)
        return self.predictions
    
    def show_classification_report(self,df_target):
        classification_report=metrics.classification_report(df_target[df_target['id'].map(lambda x: x in self.test_ids)]['page_type'].values,self.predictions)
        print(classification_report)