import numpy as np
import scipy.sparse as sparse
import pandas as pd
import json
import time
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import copy
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


t1=time.time()
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

def create_X(df_target,page_types, df_know_infos, df_edges, features,X_ids ):
    
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*4+9))
    for iter in range(nb_ids):
        id=X_ids[iter]
        direct_link=get_number_link(id, df_know_infos, df_edges, page_types, 0)
        second_link=get_number_link(id, df_know_infos, df_edges, page_types, 1)
        third_link=get_number_link(id, df_know_infos, df_edges, page_types, 2)
        fourth_link=get_number_link(id, df_know_infos, df_edges, page_types, 3)
        id_features=get_features_ratios(id, features)
        row_values=list(direct_link.values())+list(second_link.values())+list(third_link.values())+list(fourth_link.values())+list(id_features.values())
        X[iter,:]=row_values
    return X

parameters = {'max_depth':range(1,23), 'criterion' :['gini', 'entropy'],'splitter': ['best','random'], 'min_samples_split' : range(3,6),'min_samples_leaf': range(1,5), 'min_impurity_decrease': [i/10 for i in range(40)] }


df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")
features=json.load(open("musae_facebook_features.json"))
nb_pages=df_target.shape[0] 
page_types=df_target['page_type'].unique()

df_know_infos=df_target[df_target['id']<nb_pages*0.05][['id','page_type']]

df_train_edges=df_edges[df_edges['id_1']<nb_pages*0.7]
df_train_target=df_target[df_target['id'].map(lambda x: True if x<0.6*nb_pages and x>0.05*nb_pages else False)==True]

df_test_edges=df_edges
df_test_target=df_target[df_target['id'].map(lambda x: True if x>0.6*nb_pages else False)==True]

Y_train=df_train_target['page_type'].values
Y_test=df_test_target['page_type'].values






nb_iter=4
initial_known_informations=copy.deepcopy(df_know_infos)
my_decision_trees=[]
for iter in range(nb_iter):
    print("starting iteration "+str(iter)+" on training part")
    X_ids_train=df_train_target['id'].values
    

    X_train=create_X(df_train_target, page_types, df_know_infos, df_edges,features,X_ids_train)
    
    X_train_resampled, Y_train_resampled=SMOTE().fit_resample(X_train,Y_train)
    
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    clf.fit(X=X_train, y=Y_train)
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_) 
    
    new_informations=tree_model.predict(X_train)
    df_know_infos=initial_known_informations.append(pd.DataFrame({'id':X_ids_train, 'page_type':new_informations}))
    my_decision_trees.append(tree_model)
    
    
df_know_infos=copy.deepcopy(initial_known_informations)
print("starting on test set")
for decision_tree in my_decision_trees:
    X_ids_test=df_test_target['id'].values
    
    X_test=create_X(df_test_target, page_types, df_know_infos, df_edges,features,X_ids_test)
    
    new_informations=decision_tree.predict(X_test)
    df_know_infos=initial_known_informations.append(pd.DataFrame({'id':X_ids_test, 'page_type':new_informations}))
    
    print("current classification report:")
    classification_report=metrics.classification_report(Y_test,new_informations)
    print(classification_report)
    
for decision_tree in my_decision_trees:
    
    new_informations=decision_tree.predict(X_test)
    accuracy=metrics.accuracy_score(Y_test,new_informations)
    classification_report=metrics.classification_report(Y_test,new_informations)
    print(classification_report)
    
confusion_matrix=metrics.confusion_matrix(Y_test, new_informations)
print(confusion_matrix)

print("elapsed time "+str(time.time()-t1)+" seconds")

