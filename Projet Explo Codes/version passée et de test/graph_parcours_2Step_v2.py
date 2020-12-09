# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:27:09 2020

@author: remi
"""


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy

def get_or_default(dictionnary, key, default):
    #Get the value of a key in a dictonnary or return default if not in the dictionnary
    val=dictionnary.get(key)
    if val==None:
        return default
    else:
        return val
    
def get_type_dict(df_target):
    #Create a dict affecting to each type a number
    count=0
    type_dict={}
    for type_ in df_target['page_type'].unique():
        type_dict.update({type_: count})
        count+=1
    return type_dict, count

def get_target_dict(target_array):
    #create dict containing the page type (value), for each id (key)
    target_dict={}
    for line in range(len(target_array)):
        target_dict.update({target_array[line,0]: target_array[line,1]})
    return target_dict


def get_edges_dict(df_edges, target=None, Use_target=False):
    #create a dict containing for each id the list of all the one it is connect to
    edges=df_edges.values
    edges_dict={}
    for edge in edges:
        if not (Use_target) or (edge[0] in target and edge[1] in target):
            edges_dict.update({edge[0]: get_or_default(edges_dict, edge[0], [])+[edge[1]]})
            edges_dict.update({edge[1]: get_or_default(edges_dict, edge[1], [])+[edge[0]]})
    return edges_dict

def add_start_point_to_edge_dict(edges_dict, df_edges, start_point):
    #add the start point known from the training part in the edges dict on the test part
    #Only add link from the start points to the point in test part but not in the other way
    edges=df_edges.values
    edges_dict_full=copy.deepcopy(edges_dict)
    for edge in edges:
        if edge[0] in start_point and not edges_dict.get(edge[1])==None and not edge[1] in start_point:
            edges_dict_full.update({edge[0]: get_or_default(edges_dict_full, edge[0], [])+[edge[1]]})
        elif edge[1] in start_point and not edges_dict.get(edge[0])==None and not edge[0] in start_point:
            edges_dict_full.update({edge[1]: get_or_default(edges_dict_full, edge[1], [])+[edge[0]]})
    edges_dict_full=check_dict(edges_dict_full)
    return edges_dict_full

def get_passage_probability(df_train_target, df_egdes, nb_random_start, nb_node_step):
    #First training on the set giving the probability of going from one node to another
    train_target=df_train_target[['id','page_type']].values
    nb_type=len(df_train_target['page_type'].unique())
    nb_nodes=len(train_target)
    
    type_dict, nb_type=get_type_dict(df_train_target)
    edges_dict=get_edges_dict(df_edges,train_target[:,0], True )
    target_dict=get_target_dict(train_target)
    
    passage_sum=np.zeros((nb_type, nb_type, nb_type))
    passage_count=np.zeros((nb_type, nb_type))
    a=0
    for start in range(nb_random_start):
        try:
            start_point_id=get_start_points(df_train_target, 0, False, 1)[0]
            start_type=target_dict.get(start_point_id)
            start_edges=edges_dict.get(start_point_id)
            reach_point_id=start_edges[np.random.randint(low=0, high=len(start_edges))]
            reach_type=target_dict.get(reach_point_id)
            last_type=start_type
            start_point_id=reach_point_id
            start_type=reach_type
            for step in range(nb_nodes*nb_node_step):
                start_edges=edges_dict.get(start_point_id)
                reach_point_id=start_edges[np.random.randint(low=0, high=len(start_edges))]
                reach_type=target_dict.get(reach_point_id)
                x_pos, y_pos, z_pos=type_dict.get(last_type),type_dict.get(start_type), type_dict.get(reach_type)
                passage_sum[x_pos,y_pos, z_pos]+=1
                passage_count[x_pos, y_pos]+=1
                start_point_id=reach_point_id
                last_type=copy.deepcopy(start_type)
                start_type=copy.deepcopy(reach_type)
        except:
            a+=1
    for x in range(nb_type):
        for y in range(nb_type):
            for z in range(nb_type):
                passage_sum[x,y,z]/=passage_count[x,y]
    return passage_sum
    
def initialyze_dict(df_test_target, nb_type):
    #Affect to each point in the test part an equivalent probability of being one type
    ids=df_test_target['id'].values
    result_dict={}
    initial_value=[1/nb_type for i in range(nb_type)]
    for id in ids:
        result_dict.update({id: initial_value})
    return result_dict
        
def calculate_type_probabilities(last_types, start_types, passage_matrix, current_guess):
    #Calculate a probability to be of a type 
    #Use in the same time the probability implied by the last point but also the current type on the point
    reach_type=[]
    for end_type in range(len(start_types)):
        tmp=0
        for start_type in range(len(start_types)):
            if last_types==-1:
                tmp+=start_types[start_type]*np.sum(passage_matrix[:,start_type,end_type ])/np.sum(passage_matrix[:,start_type, :])
            else:
                for last_type in range(len(last_types)):
                    tmp+=last_types[last_type]*start_types[start_type]*passage_matrix[last_type, start_type,end_type]
        tmp=(tmp**2+current_guess[end_type]**2)
        reach_type.append(tmp)
    tot=sum(reach_type)
    for i in range(len(reach_type)):
        reach_type[i]=reach_type[i]/tot
    return reach_type

def delete_doublon(my_list):
    list_2=[]
    for item in my_list:
        if item not in list_2:
            list_2.append(item)
    return list_2
    
def check_dict(edge_dict):
    for key in edge_dict.keys():
        edge_dict.update({key: delete_doublon(edge_dict.get(key))}) 
    return edge_dict

def reverse_type_dict(my_dict):
    #reverse the type_dict in order to change the sens between key and value
    #work fine for every bijective dictionnary (no keys with same values)
    reversed_dict={}
    for key in my_dict.keys():
        reversed_dict.update({my_dict.get(key): key})
    return reversed_dict
    
def relabel_results(result_dict, type_dict):
    #regivve a label to the result using the reversed type dict
    reversed_type_dict=reverse_type_dict(type_dict)
    labeled_dict={}
    for key in result_dict.keys():
        if max(result_dict.get(key))<0.5:
            labeled_dict.update({key: 'not found'})
        else:
            labeled_dict.update({key: reversed_type_dict.get(np.argmax(result_dict.get(key)))})
    return labeled_dict

def get_pred_vs_Y_for_found(result_dict, df_target):
    #Create two array containing the real values and predicted values 
    #Don't take into account values where the prediction is to uncertain
    Y=[]
    pred=[]
    nf=0
    for key in result_dict.keys():
        val=result_dict.get(key)
        if val!='not found':
            Y.append(df_target[df_target['id']==key]['page_type'].values[0])
            pred.append(val)
        else:
            nf+=1
    return Y, pred,nf
    
def get_different_random_integer_values(low, high, size):
    #Create random integer value with no repetition
    if size<(high-low)/2:
        random_values=[]
        while len(random_values)<size:
            integer=np.random.randint(low, high)
            if integer not in random_values:
                random_values.append(integer)
    else:
        out_values=[]
        while len(out_values)<high-low-size:
            integer=np.random.randint(low, high)
            if integer not in out_values:
                out_values.append(integer)
        random_values=[i for i in range(low, high) if i not in out_values]
    return random_values

def get_start_points(df_train_target, pct_point, unique=True, nb_point=0):
    #Give different start point in the training part in order to predict the test part
    #Give a percentage of the total start point in the training part with no repetition
    if unique:
        starts_points=get_different_random_integer_values(0, len(df_train_target), int(len(df_train_target)*pct_point))
        starts_points_id=df_train_target[df_train_target.reset_index(drop=True).index.map(lambda x: True if x in starts_points else False)]['id'].values
    else:
        starts_points=np.random.randint(0, len(df_train_target),nb_point )
        df=df_train_target.reset_index(drop=True)
        starts_points_id=[df[df.index==i]['id'].values[0] for i in starts_points]
    
    return starts_points_id
    
def get_random_point_but_not_last(edges, last_point):
    val=edges[np.random.randint(low=0, high=len(edges))]
    while val==last_point and len(edges)>1:
        val=edges[np.random.randint(low=0, high=len(edges))]
    return val

def assign_point_type(df_test_target,df_train_target, df_edges, edges_dict_test, passage_matrix,pct_point, nb_node_step, nb_random_restart):
    #Main predicition function
    #In a first part use the starts point from the training part to calculate probability on nodes in the test part
    #In a second time use the nodes inside the test part to recalculate probabilities over the other by propagation
    result_dict=initialyze_dict(df_test_target, len(passage_matrix))
    
    starts_points_id=get_start_points(df_train_target, pct_point)
    type_dict, nb_type=get_type_dict(df_train_target)
    edges_dict=add_start_point_to_edge_dict(edges_dict_test, df_edges, starts_points_id)
    a=0

    for start in range(len(starts_points_id)):
        try:
            start_point_id=starts_points_id[start]
            start_type=[0 for i in range(nb_type)]
            start_type[type_dict.get(df_train_target[df_train_target['id']==start_point_id]['page_type'].values[0])]=1
            last_type=-1
            last_point=-1
            for reach_point_id in edges_dict.get(start_point_id):
                reach_type=calculate_type_probabilities(last_type,start_type, passage_matrix, result_dict.get(reach_point_id))
                result_dict.update({reach_point_id: reach_type})
                start_point_id_2=reach_point_id
                start_type_2=reach_type
                for step in range(nb_node_step):
                    start_edges=edges_dict.get(start_point_id_2)
                    for reach_point_id_2 in start_edges:
                        reach_type_2=calculate_type_probabilities(last_type,start_type_2, passage_matrix, result_dict.get(reach_point_id_2))
                        result_dict.update({reach_point_id_2: reach_type_2})
                    last_point=start_point_id_2
                    start_point_id_2=get_random_point_but_not_last(start_edges, last_point)
                    last_type=copy.deepcopy(start_type_2)
                    start_type_2= result_dict.get(start_point_id)
        except:
            a+=1
    starts_points_id=get_start_points(df_test_target, 0, False, nb_random_restart)
    for start in range(nb_random_restart):
        try:
            start_point_id=starts_points_id[start]
            start_type=result_dict.get(start_point_id)
            last_type=-1
            last_point=-1
            for step in range(nb_node_step):
                start_edges=edges_dict.get(start_point_id)
                for reach_point_id in start_edges:
                    reach_type=calculate_type_probabilities(last_type,start_type, passage_matrix, result_dict.get(reach_point_id))
                    result_dict.update({reach_point_id: reach_type})
                last_point=start_point_id
                start_point_id=get_random_point_but_not_last(start_edges, last_point)
                last_type=copy.deepcopy(start_type)
                start_type= result_dict.get(start_point_id)
        except:
            a+=1        

    labeled_dict=relabel_results(result_dict, type_dict)
    return result_dict,labeled_dict


        
def parameters_research(df_train_target, df_edges,pct_point_use, nb_node_step_limits, nb_random_restart_limits, params_precision_step_iter, params_step_number):
    #Research best parameter by crossvalidation over the training part
    Indexes=df_train_target['id']
    kf=KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(Indexes)
    
    best_accuracy=[]
    best_score=[]
    nb_node_step_down,nb_node_step_up, nb_node_step_step=nb_node_step_limits[0],nb_node_step_limits[1], int((nb_node_step_limits[1]-nb_node_step_limits[0])/params_step_number)+1
    nb_random_restart_down,nb_random_restart_up,nb_random_restart_step=nb_random_restart_limits[0],nb_random_restart_limits[1], int((nb_random_restart_limits[1]-nb_random_restart_limits[0])/params_step_number)+1
    best_nb_node_step=[]
    best_nb_random_restart=[]
    for train_fold, test_fold in kf.split(Indexes):
        best_accuracy.append(0)
        best_score.append(0)
        df_train_train_target=df_train_target[df_train_target['id'].map(lambda x: True if x in train_fold else False)==True]
        df_train_test_target=df_train_target[df_train_target['id'].map(lambda x: True if x in test_fold else False)==True]
        passage_matrix=get_passage_probability(df_train_train_target, df_edges, 10,6)
        test_target=df_train_test_target[['id','page_type']].values
        edges_dict_test=get_edges_dict(df_edges,test_target[:,0], True )
        step=1
        for precision_step in range(params_precision_step_iter):
            
            for nb_node_step in range(nb_node_step_down,nb_node_step_up, nb_node_step_step):
                for nb_random_restart in range(nb_random_restart_down,nb_random_restart_up, nb_random_restart_step):
                    raw_dict, predict_dict=assign_point_type(df_train_test_target, df_train_train_target, df_edges,edges_dict_test, passage_matrix,pct_point_use, nb_node_step,nb_random_restart)
                    Y_real, Y_pred, nf=get_pred_vs_Y_for_found(predict_dict, df_target)
                    accuracy=metrics.accuracy_score(Y_real,Y_pred)
                    score=accuracy/nf
                    if score>best_score[-1]:
                        best_nb_node_step_fold=nb_node_step
                        best_nb_random_restart_fold=nb_random_restart
                        best_score[-1]=score
                        best_accuracy[-1]=accuracy
            
            nb_node_step_down,nb_node_step_up=max(1,int(best_nb_node_step_fold-(nb_node_step_limits[1]-nb_node_step_limits[0])/params_step_number/step)),int(best_nb_node_step_fold+(nb_node_step_limits[1]-nb_node_step_limits[0])/params_step_number/step)
            nb_node_step_step=int((nb_node_step_up-nb_node_step_down)/params_step_number)+1
            nb_random_restart_down,nb_random_restart_up=max(1,int(nb_random_restart-(nb_random_restart_limits[1]-nb_random_restart_limits[0])/params_step_number/step)),int(nb_random_restart+(nb_random_restart_limits[1]-nb_random_restart_limits[0])/params_step_number/step)
            nb_random_restart_step=int((nb_random_restart_up-nb_random_restart_down)/params_step_number)+1
            step*=2
        best_nb_node_step.append(best_nb_node_step_fold)
        best_nb_random_restart.append(best_nb_random_restart_fold)
    print("mean accuracy on train set"+ str(np.mean(best_accuracy)))
    return int(np.round(np.mean(best_nb_node_step))), int(np.round(np.mean(best_nb_random_restart)))
    
df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")

nb_pages=df_target.shape[0] 
page_types=df_target['page_type'].unique()

Indexes=df_target['id']
train, test= train_test_split(Indexes, test_size=0.4)

df_train_target=df_target[df_target['id'].map(lambda x: True if x in train else False)==True]
df_test_target=df_target[df_target['id'].map(lambda x: True if x in test else False)==True]


pm=get_passage_probability(df_train_target, df_edges, 10,6)
for pct_used in [0.05,0.1,0.2,0.5,0.6]:
    node_step, nb_random_start=parameters_research(df_train_target, df_edges,pct_used, [2,50], [50,10000], 3, 4)
    #node_step, nb_random_start=10,5000
    
    passage_matrix=get_passage_probability(df_train_target, df_edges, 10,6)
    best_accuracy=0
    test_target=df_test_target[['id','page_type']].values
    edges_dict_test=get_edges_dict(df_edges,test_target[:,0], True )
    
    
    raw_dict, predict_dict=assign_point_type(df_test_target, df_train_target, df_edges,edges_dict_test, passage_matrix,pct_used, node_step,nb_random_start)
    Y_real, Y_pred, nf=get_pred_vs_Y_for_found(predict_dict, df_target)
    accuracy=metrics.accuracy_score(Y_real,Y_pred)
    
    
    print("result using v2step.2 "+str(pct_used*100)+"% of the dataset")
    print(metrics.classification_report(Y_real, Y_pred))
    print(str(nf/len(raw_dict)*100)+"% of the values where not classified")