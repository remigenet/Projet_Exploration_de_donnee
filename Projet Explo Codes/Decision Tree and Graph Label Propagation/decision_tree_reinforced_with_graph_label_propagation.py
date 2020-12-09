#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:36:29 2020

@author: rgenet
"""

import decision_tree_reinforced_functions as dtr_func
import graph_label_propagator as glp
from sklearn import metrics

class decisition_tree_reinforced_and_label_progragation:
    
    def __init__(self,df_edges, nb_iter, df_know_infos, features, create_X_func):
        self.df_edges=df_edges
        self.nb_iter=nb_iter
        self.df_know_infos=df_know_infos
        self.features=features
        self.create_X_func=create_X_func
        
    def fit(self,df_train_target,df_train_target_for_dtr,nb_reinforcment_list=[2,4,6], prct_max_reinforced_list=[0.2,0.4,0.6,0.8], depth_list=[1,2,3]):
        X_ids=list(self.df_edges['id_1'].unique())
        X_ids+=[id for id in self.df_edges['id_2'].unique() if id not in X_ids]
        self.page_types=df_train_target['page_type'].unique()
        self.partition_dict=dtr_func.get_partitions(X_ids, self.df_edges)
        
        my_model=glp.graph_label_propagator( self.df_edges, df_train_target, X_ids,pct_used=0, verbose=True,used_fixed_known_points=True, fixed_known_points=self.df_know_infos['id'].values)
        my_model.set_parameters_by_crossvalidation(nb_reinforcment_list=nb_reinforcment_list, prct_max_reinforced_list=prct_max_reinforced_list, depth_list=depth_list,max_workers=10, nb_splits=5)
        #my_model.set_parameters_manually(depth=2,nb_reinforcment=3, prct_max_reinforced=0.6)
        my_model.fit()
        
        labeled_dict, result_dict=my_model.predict(X_ids)
        self.result_dict=result_dict
        
        self.models=dtr_func.create_trees_with_label_propagate(self.nb_iter,self.df_know_infos,df_train_target_for_dtr, self.partition_dict, self.features,self.page_types, self.df_edges,self.create_X_func, result_dict)
        
    def predict(self, test_ids):
        self.test_ids=test_ids
        self.predictions=dtr_func.use_trees_with_label_propagate(self.models, test_ids, self.df_edges, self.partition_dict,self.df_know_infos, self.features,self.page_types,self.create_X_func, self.result_dict)
        return self.predictions
    
    def show_classification_report(self,df_target):
        classification_report=metrics.classification_report(df_target[df_target['id'].map(lambda x: x in self.test_ids)]['page_type'].values,self.predictions)
        print(classification_report)