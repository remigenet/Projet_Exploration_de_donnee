# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:40:56 2020

@author: remi
"""

import graph_label_propagator_functions as glp_func 
from sklearn import metrics

class graph_label_propagator:
    
    def __init__(self, df_edges, df_train_target, test_ids,pct_used=0.05, verbose=False, used_fixed_known_points=False, fixed_known_points=[]):
        self.df_edges=df_edges
        self.df_edges=df_edges
        self.df_train_target=df_train_target
        self.verbose=verbose
        self.pct_point=pct_used
        self.test_ids=test_ids
        self.print_verbose("Creating the edges dictionnary")
        self.edges_dict_full=glp_func.get_edges_dict(self.df_edges,full_dict=True )
        self.used_fixed_known_points=used_fixed_known_points
        self.fixed_known_points=fixed_known_points
        
    def fit(self):       
        self.print_verbose("creating the passage matrix")
        self.passage_matrix=glp_func.get_passage_probabilities(self.df_train_target, self.df_edges, 10,10,self.depth)

        
    def predict(self,test_ids):   
        self.test_ids=test_ids
        self.print_verbose("launching prediction, it can take some time for big graph")         
        result_dict, labeled_dict, not_doable_points=glp_func.assign_point_type(test_ids,self.df_train_target, self.df_edges, self.edges_dict_full, self.passage_matrix,self.pct_point, self.nb_reinforcment, self.depth,self.prct_max_reinforced, verbose=self.verbose, used_fixed_know_points=self.used_fixed_known_points, fixed_know_points_id=self.fixed_known_points)
        self.result_dict=result_dict
        self.labeled_dict=labeled_dict
        self.not_doable_points=not_doable_points
        self.print_verbose(str(len(not_doable_points)/len(test_ids)*100)+"% of the points can't be classified at all due to edges and parameters selected")
        return labeled_dict,result_dict
        
    def set_parameters_manually(self,nb_reinforcment=3,depth=2,prct_max_reinforced=0.5):
        self.nb_reinforcment=nb_reinforcment
        self.depth=depth
        self.prct_max_reinforced=prct_max_reinforced
        
    def set_parameters_by_crossvalidation(self,nb_reinforcment_list=[2,4,6], prct_max_reinforced_list=[0.2,0.4,0.6,0.8], depth_list=[1,2,3],max_workers=10, nb_splits=5):
        self.print_verbose("launch crossvalidation")
        best_nb_node_depth,best_nb_reinforcment,best_prct_max_reinforced=glp_func.parameters_research(self.df_train_target, self.df_edges,self.pct_point, depth_list, nb_reinforcment_list,prct_max_reinforced_list,nb_splits,self.edges_dict_full, max_workers=max_workers, verbose=self.verbose )
        self.nb_reinforcment=best_nb_reinforcment
        self.depth=best_nb_node_depth
        self.prct_max_reinforced=best_prct_max_reinforced
        self.print_verbose("depth selected: "+str(best_nb_node_depth)+" , number of reinforcement selected: "+str(best_nb_reinforcment)+" and percentage reinforced: "+str(best_prct_max_reinforced))
        
    def display_classification_report(self, df_target):
        Y_real, Y_pred, nf=glp_func.get_pred_vs_Y_for_found(self.labeled_dict, df_target, self.test_ids)
        self.print_verbose(str((len(self.not_doable_points)/(nf+len(Y_real)))*100)+"% of points totaly unclassifiable")
        print(metrics.classification_report(Y_real, Y_pred))

    def print_verbose(self,to_print):
        if self.verbose: print(to_print)