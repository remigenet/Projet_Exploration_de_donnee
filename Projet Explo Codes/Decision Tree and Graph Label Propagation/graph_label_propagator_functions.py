#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:28:17 2020

@author: rgenet
"""



import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import copy
import itertools
import multiprocessing

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


def get_edges_dict(df_edges, target=None, Use_target=False, full_dict=False):
    #create a dict containing for each id the list of all the one it is connect to
    edges=df_edges.values
    edges_dict={}
    for edge in edges:
        if not (Use_target) or (edge[0] in target and edge[1] in target) or full_dict:
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

def push_in_list_and_remove_last(my_list, new_val):
    new_list=[]
    new_list=[item for item in my_list[1:]]
    new_list.append(new_val)
    return new_list

def initialize_passage_matrix_and_counter_matrix(depth, nb_type):
    return np.zeros((nb_type,) * (depth+1)),np.zeros((nb_type,) * depth)

def fill_passage_count_matrix(passage_sum,passage_count, lasts_points_type, reach_type):
    passage_sum[tuple(lasts_points_type)][reach_type]+=1
    passage_count[tuple(lasts_points_type)]+=1
    return passage_sum,passage_count

def transform_sum_count_in_prob_matrix(passage_sum,passage_count):
    probability_matrix=copy.deepcopy(passage_sum)
    for item_place in itertools.product([i for i in range(len(passage_sum))],repeat=passage_count.ndim):
        for reach_type in range(len(passage_sum)):
            if passage_count[item_place]==0:
                probability_matrix[item_place][reach_type]=0
            else:
                probability_matrix[item_place][reach_type]=passage_sum[item_place][reach_type]/passage_count[item_place]
    return probability_matrix

def get_passage_probabilities(df_train_target, df_egdes, nb_random_start, nb_node_step, n_depth):
    #First training on the set giving the probability of going from one node to another
    train_target=df_train_target[['id','page_type']].values
    nb_type=len(df_train_target['page_type'].unique())
    nb_nodes=len(train_target)
    
    type_dict, nb_type=get_type_dict(df_train_target)
    edges_dict=get_edges_dict(df_egdes,train_target[:,0], True )
    target_dict=get_target_dict(train_target)
    passage_matrix=[]
    for depth in range(1,n_depth+1):
        passage_matrix.append(get_passage_prob_for_depth(df_train_target,target_dict, edges_dict,type_dict, nb_type,nb_nodes,nb_node_step,nb_random_start, depth))
    return passage_matrix

def get_passage_prob_for_depth(df_train_target,target_dict, edges_dict,type_dict, nb_type,nb_nodes,nb_node_step,nb_random_start, depth):
    passage_sum,passage_count=initialize_passage_matrix_and_counter_matrix(depth, nb_type)
    a=0
    for start in range(nb_random_start):
        try:
            lasts_points_type=[]
            start_point_id=get_start_points(df_train_target, 0, False, 1)[0]
            start_type=type_dict.get(target_dict.get(start_point_id))
            lasts_points_type.append(start_type)
            last_points_id=-1
            for diving in range(depth-1):
                reach_point_id=get_random_point_but_not_last(edges_dict.get(start_point_id), last_points_id)
                last_points_id=start_point_id
                start_type=type_dict.get(target_dict.get(reach_point_id))
                start_point_id=reach_point_id
                lasts_points_type.append(start_type)
                
            for step in range(nb_nodes*nb_node_step):
                reach_point_id=get_random_point_but_not_last(edges_dict.get(start_point_id), last_points_id)
                reach_type=type_dict.get(target_dict.get(reach_point_id))
                passage_sum,passage_count=fill_passage_count_matrix(passage_sum,passage_count, lasts_points_type, reach_type)
                last_points_id=start_point_id
                start_point_id=reach_point_id
                lasts_points_type=push_in_list_and_remove_last(lasts_points_type,reach_type)
        except:
            a+=1

    return transform_sum_count_in_prob_matrix(passage_sum, passage_count)
    
def initialyze_dict(ids, nb_type, df_train_target, starts_points_id, type_dict):
    #Affect to each point in the test part an equivalent probability of being one type
    result_dict={}
    #initial_value=[1/nb_type for i in range(nb_type)]
    initial_value=[0 for i in range(nb_type)]
    for id in ids:
        result_dict.update({id: initial_value})
    for id in starts_points_id:
        start_type=[0 for i in range(nb_type)]
        #start_type[type_dict.get(df_train_target[df_train_target['id']==start_point_id]['page_type'].values[0])]=1
        start_type[type_dict.get(df_train_target[df_train_target['id']==id]['page_type'].values[0])]=1
        result_dict.update({id: start_type})
        
    return result_dict
        

def calculate_type_probabilities(lasts_types, passage_matrix, current_guess=[], use_guess=False, additionate=False):
    #Calculate a probability to be of a type 
    #Use in the same time the probability implied by the last point but also the current type on the point
    reach_type=[]
    count_test=0
    for end_type in range(len(passage_matrix)):
        end_tmp=0
        for path_to_end_type in itertools.product([i for i in range(len(passage_matrix))],repeat=len(lasts_types)):
            path_tmp=1
            for points in range(len(path_to_end_type)):
                path_tmp*=lasts_types[points][path_to_end_type[points]]
                if np.isnan(path_tmp):
                    print(lasts_types)
                    print(lasts_types[points])
                    print(lasts_types[points][path_to_end_type[points]])
                    print(path_to_end_type[points])
            end_tmp+=passage_matrix[path_to_end_type][end_type]*path_tmp
        count_test+=end_tmp
        if use_guess and current_guess[end_type]>0:
            end_tmp=(end_tmp+current_guess[end_type])/2
        if additionate:
            end_tmp=end_tmp+current_guess[end_type]
        reach_type.append(end_tmp)

    return reach_type

def get_sum_back_to_prob(result_dict, nb_type):
    for key in result_dict.keys():
        types=result_dict.get(key)
        if types!=[0 for i in range(nb_type)]:
            tot=np.sum(types)
            for val in range(len(types)):
                types[val]/=tot
            result_dict.update({key: types})
    return result_dict

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

def spread_information(result_dict,edges_dict,start_point_id, propagation_matrix, start_type, depth, unmovable_type_points_id):
    
    linked_points=from_link_id_list_to_array(get_linked_id_to_point(edges_dict, start_point_id, -1, depth), depth)
    result_dict=update_dict(linked_points, propagation_matrix, result_dict, start_type, unmovable_type_points_id)
    
    return result_dict

def get_linked_id_to_point(edges_dict,start_point_id, last_point, still_iter):
    
    new_last_point=start_point_id  
    lasts_points_type=[]
    if still_iter>0 and edges_dict.get(start_point_id)!=None:
        for reach_point_id in edges_dict.get(start_point_id):
            if reach_point_id!=last_point:
                new_val=get_linked_id_to_point(edges_dict,reach_point_id, new_last_point, still_iter-1)
                if new_val!=[]:
                    for item in new_val:
                        to_add=copy.deepcopy([reach_point_id])
                        for small_item in item:
                            to_add.append(small_item)
                            lasts_points_type.append(to_add)    
                    
                else:
                    lasts_points_type.append([reach_point_id])
           
    return lasts_points_type

def from_link_id_list_to_array(link_list, depth):
    my_array=np.zeros((len(link_list), depth))-1
    count=0
    for item in range(len(link_list)):
        if item>0:
            if link_list[item]!=link_list[item-1]:
                my_array[count,:len(link_list[item])]=link_list[item]
                count+=1
    result_array=my_array[:count,:]
    return result_array

def update_dict(link_array, propagation_matrix, result_dict, start_type,unmovable_type_points_id):
    
    done_id=[-1]
    for col in range(np.shape(link_array)[1]):
        for row in range(np.shape(link_array)[0]):
            id=link_array[row,col]
            if id not in done_id and id not in unmovable_type_points_id:
                cval=result_dict.get(id)
                val=[propagation_matrix[start_type][col][type_]+cval[type_] for type_ in range(len(cval))]
                result_dict.update({id: val})
    
    return result_dict

def calculate_matrix_propagation(passage_matrix):
    nb_type=len(passage_matrix[0])
    propagation_matrix=[]
    for start_type in range(nb_type):
        current_start=[0 for i in range(nb_type)]
        current_start[start_type]=1
        current_start=[current_start]
        for depth in range(len(passage_matrix)):
            current_start.append(calculate_type_probabilities(current_start, passage_matrix[depth]))
        propagation_matrix.append(current_start[1:])
    return propagation_matrix

def get_infos_back_to_point(result_dict,edges_dict,start_point_id, passage_matrix, depth, still_iter,unmovable_type_points_id, know_points):
    
    still_iter=min(still_iter, depth)
    link_neigbhours=get_linked_type_to_point(result_dict,edges_dict,start_point_id, -1, still_iter, unmovable_type_points_id,know_points)
    new_prob=take_neighbors_information(link_neigbhours, passage_matrix, result_dict.get(start_point_id))
    if (np.max(new_prob)> np.max(result_dict.get(start_point_id))) or (np.argmax(new_prob)!= np.argmax(result_dict.get(start_point_id)) and (np.max(new_prob)>0.9*np.max(result_dict.get(start_point_id)))) :
        result_dict.update({start_point_id: new_prob})

    return result_dict

def get_not_doable_points(result_dict,edges_dict, depth, still_iter, unmovable_type_points_id, know_points):
    
    not_doable_points=[]
    still_iter=min(still_iter, depth)
    for id in result_dict.keys():
        if result_dict.get(id)==[0,0,0,0] and edges_dict.get(id)!=None:
            link_neigbhours=get_linked_type_to_point(result_dict,edges_dict,id, -1, still_iter, unmovable_type_points_id, know_points)
            if link_neigbhours!=[]:
                if np.max([np.min(link_list) for link_list in link_neigbhours])==0:
                    not_doable_points=not_doable_points+[id]
        elif edges_dict.get(id)==None and result_dict.get(id)==[0,0,0,0]:
            not_doable_points=not_doable_points+[id]
    return not_doable_points

def get_linked_type_to_point(result_dict,edges_dict,start_point_id, last_point, still_iter, unmovable_type_points_id,know_points):
    
    new_last_point=start_point_id  
    lasts_points_type=[]
    if still_iter>0 and edges_dict.get(start_point_id)!=None:
        for reach_point_id in edges_dict.get(start_point_id):
            if reach_point_id!=last_point and (reach_point_id not in unmovable_type_points_id or reach_point_id in know_points):
                reach_type=[result_dict.get(reach_point_id)]
                new_val=get_linked_type_to_point(result_dict,edges_dict,reach_point_id, new_last_point, still_iter-1, unmovable_type_points_id,know_points)
                if new_val!=[]:
                    for item in new_val:
                        to_add=copy.deepcopy(reach_type)
                        for small_item in item:
                            to_add.append(small_item)
                            lasts_points_type.append(to_add)    
                    
                else:
                    lasts_points_type.append(reach_type)
           
    return lasts_points_type



def take_neighbors_information(link_neigbhours, passage_matrix, start_type):
    mean_type=np.zeros(len(passage_matrix[0]))
    total_working_link=0
    for link_list in link_neigbhours:
        if [0,0,0,0] not in link_list:
            
            link_good_orders=link_list[::-1]
            prob=calculate_type_probabilities(link_good_orders,passage_matrix[len(link_list)-1])
            #if np.max(prob)>np.max(start_type):
    #        for item in range(len(prob)):
    #            prob[item]=prob[item]**2
            mean_type+=prob
            total_working_link+=1
    if total_working_link!=0:
        mean_type=list(mean_type/total_working_link)
    else:
        mean_type=start_type
    return mean_type

def delete_known_infos(result_dict, unmovable_type_points_id):
    for id in unmovable_type_points_id:
        result_dict.pop(id)
    return result_dict

def get_points_with_less_significant_probabilities(result_dict, nb_points_wanted, edges_dict, not_doable_points):
    list_key=[key for key in result_dict.keys()]
    associated_prob=[np.max(result_dict.get(key)) for key in list_key]
    key_sorted_by_prob=[list_key[key] for key in np.argsort(associated_prob) if edges_dict.get(list_key[key])!=None and list_key[key] not in not_doable_points]
    return key_sorted_by_prob[:min(nb_points_wanted,len(key_sorted_by_prob))]

def assign_point_type(test_ids,df_train_target, df_edges, edges_dict_full, passage_matrix,pct_point, nb_reinforcment, depth,prct_max_reinforced, only_predict=False, verbose=False, used_fixed_know_points=False, fixed_know_points_id=[]):
    #Main predicition function
    #In a first part use the starts point from the training part to calculate probability on nodes in the test part
    #In a second time use the nodes inside the test part to recalculate probabilities over the other by propagation
    if used_fixed_know_points:
        starts_points_id=fixed_know_points_id    
    else:
        starts_points_id=get_start_points(df_train_target, pct_point)
    unmovable_type_points_id=[key for key in edges_dict_full.keys() if key not in test_ids]
    
    type_dict, nb_type=get_type_dict(df_train_target)
    result_dict=initialyze_dict(test_ids, len(passage_matrix[0]), df_train_target, starts_points_id,type_dict)
    print_ifverbose(verbose,"starting spreading")
    result_dict=spread_known_infos(starts_points_id, nb_type,result_dict,edges_dict_full,depth,type_dict,passage_matrix,unmovable_type_points_id )
    result_dict,labeled_dict, not_doable_points=reinforced(edges_dict_full, depth,result_dict, type_dict,passage_matrix,prct_max_reinforced,unmovable_type_points_id,starts_points_id, nb_reinforcment=nb_reinforcment, verbose=verbose)
    if not used_fixed_know_points:
        result_dict=delete_known_infos(result_dict, starts_points_id)
    if only_predict:
        return labeled_dict
    else:
        return result_dict,labeled_dict,not_doable_points
    
def spread_known_infos(starts_points_id, nb_type,result_dict,edges_dict,depth,type_dict,passage_matrix,unmovable_type_points_id ):
   
    propagation_matrix=calculate_matrix_propagation(passage_matrix)
    
    for id in starts_points_id:
        result_dict= spread_information(result_dict,edges_dict,id, propagation_matrix, np.argmax(result_dict.get(id)), depth, unmovable_type_points_id)
    result_dict=get_sum_back_to_prob(result_dict, nb_type)
    return result_dict
    
def reinforced(edges_dict, depth,result_dict, type_dict,passage_matrix,prct_max_reinforced,unmovable_type_points_id, know_points, nb_reinforcment=6, verbose=False):
    for reinforcement in range(nb_reinforcment):
        print_ifverbose(verbose, "doing reinforcment number: "+str(reinforcement))
        not_doable_points=get_not_doable_points(result_dict,edges_dict, depth, depth, unmovable_type_points_id,know_points)
        nb_point_per_reinforcment=int((len(result_dict)-len(not_doable_points))*(reinforcement+1)/nb_reinforcment*prct_max_reinforced)
        starts_points_id=get_points_with_less_significant_probabilities(result_dict, nb_point_per_reinforcment, edges_dict, not_doable_points)
        for id in starts_points_id:
            result_dict=get_infos_back_to_point(result_dict, edges_dict, id, passage_matrix,depth,depth, unmovable_type_points_id,know_points )


    labeled_dict=relabel_results(result_dict, type_dict)
    return result_dict,labeled_dict,not_doable_points


def parameters_research(df_train_target, df_edges,pct_point, depth_list, nb_reinforcment_list,prct_max_reinforced_list, nb_splits,edges_dict_full, max_workers=10, verbose=False ):
    #Research best parameter by crossvalidation over the training part
    Indexes=df_train_target['id']
    kf=KFold(n_splits=nb_splits, shuffle=True)
    kf.get_n_splits(Indexes)
    pct_point=pct_point/(nb_splits-1)*nb_splits
    best_accuracy=[]
    best_score=[]
    best_nb_node_depth=[]
    best_nb_reinforcment=[]
    best_prct_max_reinforced=[]
    nb_workers=min(max_workers,len(prct_max_reinforced_list)*len(nb_reinforcment_list))
    count=1
    for train_fold, test_fold in kf.split(Indexes):
        print_ifverbose(verbose,"doing fold number "+str(count))
        best_accuracy.append(0)
        best_score.append(0)
        df_train_train_target=df_train_target[df_train_target['id'].map(lambda x: True if x in train_fold else False)==True]
        df_train_test_target=df_train_target[df_train_target['id'].map(lambda x: True if x in test_fold else False)==True]
        test_ids=df_train_test_target[['id','page_type']].values[:,0]
        type_dict, nb_type=get_type_dict(df_train_train_target)
        for depth in depth_list:
            print_ifverbose(verbose, "using depth "+str(depth))
            passage_matrix=get_passage_probabilities(df_train_target, df_edges, 10,10,depth)
            args=[]
            for nb_reinforcment in nb_reinforcment_list:  
                for prct_max_reinforced in prct_max_reinforced_list:
                    args.append((test_ids,df_train_train_target, df_edges, edges_dict_full, passage_matrix,pct_point, nb_reinforcment, depth,prct_max_reinforced,True))
                    
            with multiprocessing.Pool(processes=nb_workers) as pool:
                results = pool.starmap(assign_point_type, args)
                
            for result_number in range(len(results)):
                result=results[result_number]
                Y_real, Y_pred, nf=get_pred_vs_Y_for_found(result, df_train_target)
                accuracy=metrics.accuracy_score(Y_real,Y_pred)
                score=accuracy*(1-nf/len(df_train_test_target))
                if score>best_score[-1]:
                    best_nb_node_depth_fold=depth
                    best_nb_reinforcment_fold=args[result_number][6]
                    best_prct_max_reinforced_fold=args[result_number][8]
                    best_score[-1]=score
                    best_accuracy[-1]=accuracy
            
        count+=1
        best_prct_max_reinforced.append(best_prct_max_reinforced_fold)
        best_nb_reinforcment.append(best_nb_reinforcment_fold)
        best_nb_node_depth.append(best_nb_node_depth_fold)
    print("mean accuracy on crossvalidation folds "+ str(np.mean(best_accuracy)))
    return int(np.round(np.mean(best_nb_node_depth))),int(np.round(np.mean(best_nb_reinforcment))),np.mean(best_prct_max_reinforced)
    
def print_ifverbose(verbose, toprint):
    if verbose: print(toprint)