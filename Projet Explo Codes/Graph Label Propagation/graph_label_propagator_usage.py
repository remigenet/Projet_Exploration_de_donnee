# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:59:41 2020

@author: remi
"""
import graph_label_propagator as glp
import pandas as pd
from sklearn.model_selection import train_test_split
import time

t_init=time.time()

df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")

Indexes=df_target['id']
test_size=0.4
train, test= train_test_split(Indexes, test_size=test_size)

df_train_target=df_target[df_target['id'].map(lambda x: True if x in train else False)==True]
df_test_target=df_target[df_target['id'].map(lambda x: True if x in test else False)==True]

test_target=df_test_target[['id','page_type']].values
test_ids=test_target[:,0]

nb_reinforcment_list=[2,4,6]
prct_max_reinforced_list=[0.2,0.4,0.6,0.8]
depth_list=[1,2,3]


for pct_used in [0.05,0.1,0.2,0.5,0.6]:
    print("pourcentage de valeurs connus utilisé: "+str(pct_used))
    t=time.time()
    
    my_model=glp.graph_label_propagator( df_edges, df_train_target, test_ids,pct_used=pct_used, verbose=True)
    #my_model.set_parameters_by_crossvalidation(nb_reinforcment_list=nb_reinforcment_list, prct_max_reinforced_list=prct_max_reinforced_list, depth_list=depth_list,max_workers=10, nb_splits=5)
    my_model.set_parameters_manually()
    my_model.fit()
    labeled_dict, result_dict=my_model.predict(test_ids)
    my_model.display_classification_report(df_target)
    
    print("Execution time for this pct_used: "+str(time.time()-t)+" seconds")

print("Total execution time: "+str(time.time()-t_init)+" seconds")