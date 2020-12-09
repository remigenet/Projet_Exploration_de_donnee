# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:08:22 2020

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
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth,DBSCAN
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering
import matplotlib.pyplot as plt


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
    return my_dict

def create_X(df_target,page_types, df_know_infos, df_edges, features,X_ids ):
    
    nb_page_type=len(page_types)
    nb_ids=len(X_ids)
    X=np.zeros((nb_ids, nb_page_type*3+6))
    for iter in range(nb_ids):
        id=X_ids[iter]
        direct_link=get_number_link(id, df_know_infos, df_edges, page_types, 0)
        second_link=get_number_link(id, df_know_infos, df_edges, page_types, 1)
        third_link=get_number_link(id, df_know_infos, df_edges, page_types, 2)
        id_features=get_features_ratios(id, features)
        row_values=list(direct_link.values())+list(second_link.values())+list(third_link.values())+list(id_features.values())
        X[iter,:]=row_values
    return X




###### Create variables #######
    

df_edges=pd.read_csv("musae_facebook_edges.csv")
df_target=pd.read_csv("musae_facebook_target.csv")
features=json.load(open("musae_facebook_features.json"))
nb_pages=df_target.shape[0] 
page_types=df_target['page_type'].unique()

df_know_infos=df_target[['id','page_type']]
df_know_partial_infos=df_target[df_target['id']<nb_pages*0.05][['id','page_type']]
X_ids=df_target['id'].values
Y=df_target['page_type'].values
X=create_X(df_target, page_types, df_know_infos, df_edges,features,X_ids)
X_partial_info=create_X(df_target, page_types, df_know_partial_infos, df_edges,features,X_ids)


X_centered=X/X.mean(axis=0)
X_partial_info_centered=X_partial_info/X_partial_info.mean(axis=0)

####Create cluster ########

kmeans=KMeans(n_clusters=4, random_state=0).fit(X_centered)
y_kmeans=kmeans.labels_

kmeans_partial_info=KMeans(n_clusters=4, random_state=0).fit(X_partial_info_centered)
y_kmeans_partial_info=kmeans_partial_info.labels_

Agglomerative_clusters = AgglomerativeClustering(n_clusters=4).fit(X_centered)
y_aglomerative=Agglomerative_clusters.labels_

Agglomerative_clusters_partial_info = AgglomerativeClustering(n_clusters=4).fit(X_partial_info_centered)
y_aglomerative_partial_info=Agglomerative_clusters_partial_info.labels_


##### link clusters detected with real datas throught tree classifier ####

parameters = {'max_depth':range(1,4), 'criterion' :['gini', 'entropy'],'splitter': ['best','random'], 'min_samples_split' : range(2,4),'min_samples_leaf': range(1,5), 'min_impurity_decrease': [i/10 for i in range(10)] }

def f(x):
    if x=='government':
        x=0
    elif x=='politician':
        x=3
    elif x=='company':
        x=2
    else:
        x=1
    return x
    

y_real_kmeans = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4).fit(X=np.array(list(map(f,Y))).reshape(-1, 1), y=y_kmeans).best_estimator_.predict(np.array(list(map(f,Y))).reshape(-1, 1))
y_real_kmeans_partial_info = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4).fit(X=np.array(list(map(f,Y))).reshape(-1, 1), y=y_kmeans_partial_info).best_estimator_.predict(np.array(list(map(f,Y))).reshape(-1, 1))
y_real_aglomerative = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4).fit(X=np.array(list(map(f,Y))).reshape(-1, 1), y=y_aglomerative).best_estimator_.predict(np.array(list(map(f,Y))).reshape(-1, 1))
y_real_aglomerative_partial_info = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4).fit(X=np.array(list(map(f,Y))).reshape(-1, 1), y=y_aglomerative_partial_info).best_estimator_.predict(np.array(list(map(f,Y))).reshape(-1, 1))

classification_report_kmeans=metrics.classification_report(y_real_kmeans,y_kmeans)
classification_report_kmeans_partial_info=metrics.classification_report(y_real_kmeans_partial_info,y_kmeans_partial_info)
classification_report_aglomerative=metrics.classification_report(y_real_aglomerative,y_aglomerative)
classification_report_aglomerative_partial_info=metrics.classification_report(y_real_aglomerative_partial_info,y_aglomerative_partial_info)


##### print the results of the classifications #####
print('classification_report_kmeans')
print(classification_report_kmeans)
print('classification_report_kmeans_partial_info')
print(classification_report_kmeans_partial_info)
print('classification_report_aglomerative')
print(classification_report_aglomerative)
print('classification_report_aglomerative_partial_info')
print(classification_report_aglomerative_partial_info)

#######   plot scatter plot of differents cluster methods vs real clusters #######

fig, axs = plt.subplots(2, 3)
fig.suptitle('direct link, Y known')
axs[0, 0].scatter(X[:,0],X[:,1], c=y_kmeans, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,1],X[:,2], c=y_kmeans, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,0],X[:,1], c=y_aglomerative, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,1],X[:,2], c=y_aglomerative, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,0],X[:,1], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,1],X[:,2], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on direct links all Y known")
plt.show()

fig, axs = plt.subplots(2, 3)
fig.suptitle('second links, Y known')
axs[0, 0].scatter(X[:,4],X[:,5], c=y_kmeans, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,6],X[:,7], c=y_kmeans, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,4],X[:,5], c=y_aglomerative, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,6],X[:,7], c=y_aglomerative, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,4],X[:,5], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,6],X[:,7], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on seconds links all Y known")
plt.show()

fig, axs = plt.subplots(2, 3)
fig.suptitle('third links, Y known')
axs[0, 0].scatter(X[:,8],X[:,9], c=y_kmeans, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,10],X[:,11], c=y_kmeans, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,8],X[:,9], c=y_aglomerative, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,10],X[:,11], c=y_aglomerative, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,8],X[:,9], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,10],X[:,11], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on third links all Y known")
plt.show()







fig, axs = plt.subplots(2, 3)
fig.suptitle('direct link, 5% Y known')
axs[0, 0].scatter(X[:,0],X[:,1], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,2],X[:,3], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,0],X[:,1], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,0],X[:,1], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,0],X[:,1], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,0],X[:,1], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on direct links all Y 10 percent known")
plt.show()

fig, axs = plt.subplots(2, 3)
fig.suptitle('second links, 5% Y known')
axs[0, 0].scatter(X[:,4],X[:,5], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,6],X[:,7], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,4],X[:,5], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,6],X[:,7], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,4],X[:,5], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,6],X[:,7], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on seconds links all Y 10 percent known")
plt.show()

fig, axs = plt.subplots(2, 3)
fig.suptitle('third links, 5% Y known')
axs[0, 0].scatter(X[:,8],X[:,9], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[0, 0].set_title("Kmeans")
axs[0, 0].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 0].scatter(X[:,10],X[:,11], c=y_kmeans_partial_info, s=1, cmap='viridis')
axs[1, 0].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 1].set_title("AgglomerativeClustering")
axs[0, 1].scatter(X[:,8],X[:,9], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[0, 1].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 1].scatter(X[:,10],X[:,11], c=y_aglomerative_partial_info, s=1, cmap='viridis')
axs[1, 1].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
axs[0, 2].set_title("Real Clusters")
axs[0, 2].scatter(X[:,8],X[:,9], c=list(map(f,Y)), s=1, cmap='viridis')
axs[0, 2].set(xlabel='% '+page_types[0], ylabel='% '+page_types[1])
axs[1, 2].scatter(X[:,10],X[:,11], c=list(map(f,Y)), s=1, cmap='viridis')
axs[1, 2].set(xlabel='% '+page_types[2], ylabel='% '+page_types[3])
fig.tight_layout()
plt.savefig("Kmeans vs AgglomerativeClustering on third links all Y 10 percent known")
plt.show()