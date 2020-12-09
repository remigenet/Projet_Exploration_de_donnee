#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:09:42 2020

@author: rgenet
"""
import decision_tree_reinforced as dtr
import decision_tree_reinforced_functions as dtr_func
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import json

t_init=time.time()

df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")
features=json.load(open("musae_facebook_features.json"))
Indexes=df_target['id']
test_size=0.4
train, test= train_test_split(Indexes, test_size=test_size)


df_train_target=df_target[df_target['id'].map(lambda x: True if x in train else False)==True]
df_test_target=df_target[df_target['id'].map(lambda x: True if x in test else False)==True]

test_target=df_test_target[['id','page_type']].values
test_ids=test_target[:,0]

nb_iter=3


for pct_used in [0.05,0.1,0.2,0.5,0.6]:
    known_id_infos_id=train_test_split(train,test_size=pct_used)[1]
    df_know_infos=df_target[df_target['id'].map(lambda x: True if x in known_id_infos_id else False)==True]
    
    t=time.time()
    
    my_model=dtr.decisition_tree_reinforced( df_edges, nb_iter, df_know_infos, features, dtr_func.create_X)
    my_model.fit(df_train_target)
    result_dict=my_model.predict(test_ids)
    my_model.show_classification_report(df_target)
    
    print("Execution time for this pct_used: "+str(time.time()-t)+" seconds")

print("Total execution time: "+str(time.time()-t_init)+" seconds")