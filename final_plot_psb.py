import re
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
dirname='/Users/aorlenko/Downloads/metphormin/'

data=pd.DataFrame.from_csv(dirname+'metphormin_metabolites_3.csv', sep=',',index_col=None)
data.columns=[x.lower() for x in data.columns]
features=pd.DataFrame.from_csv(dirname+'rank3.csv', sep=',')
tpot_data = data.to_records()
X_train, X_test, Y_train, Y_test = train_test_split(data, tpot_data['class'], random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tpot.builtins import ZeroCount

exported_pipeline0= make_pipeline(
    Normalizer(norm="max"),
    ExtraTreesClassifier(bootstrap=False, max_features=0.7500000000000001, min_samples_leaf=2, min_samples_split=8, n_estimators=100)

)

exported_pipeline1= make_pipeline(
        GradientBoostingClassifier(learning_rate=0.5, max_depth=1, max_features=0.8500000000000001, min_samples_leaf=18, min_samples_split=17, subsample=0.8)

)

exported_pipeline2= make_pipeline(
ExtraTreesClassifier(max_features=0.9500000000000001, min_samples_leaf=3, min_samples_split=12, n_estimators=100)
)

exported_pipeline3= make_pipeline(
GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=0.2, min_samples_leaf=9, min_samples_split=9, n_estimators=100, subsample=1.0)
)

exported_pipeline4= make_pipeline(
    RobustScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=20, subsample=0.4)
)


def plot_accuracy_rfe(data, pipeline, features, X_train, Y_train, X_test,Y_test):  
    train_acc=[]
    test_acc=[]
    
    #importance, features=zip(*sorted(zip(features.summ_r, features.index.tolist()),reverse=True))
    for i in range(1,len(features)+1):        
        learn=pipeline.fit(X_train[list(features.index[0:i])], Y_train) #clf0
        #learn.get_support(data.columns.tolist())
        #learn2=pipeline.steps[-2][1].fit(X_train[list(features[0:i])], Y_train) #clf0
        #Y_pred0= learn2.predict(X_test[list(features.index[0:i])])
        train_acc.append(learn.score(X_train[list(features.index[0:i])], Y_train))
        test_acc.append(learn.score(X_test[list(features.index[0:i])], Y_test))
    df=[features.index,features['summ_r'],train_acc, test_acc]
    df=pd.DataFrame.from_records(df).transpose()
    df.columns=[['Metabolites','Ranks','Train_R^2', 'Test_R^2']]
    return df
#for debugging 
#plot_accuracy_rfe(data,exported_pipeline3, features,X_train, Y_train, X_test, Y_test)

#a.drop(a.index[len(a)-1])
#a.drop(a.index[len(a)-1])
features.reset_index()
features.index
#features

pipelines=[exported_pipeline0, exported_pipeline1, exported_pipeline2, exported_pipeline3, exported_pipeline4]
#pipelines=[exported_pipeline0]
for p in pipelines:
    
    a=plot_accuracy_rfe(data,exported_pipeline3, features,X_train, Y_train, X_test, Y_test)
    print(a)
    #a=a.drop(a.index[len(a)-1])
    #a=a.drop(a.index[len(a)-1])
    fig, ax = plt.subplots()
    plt.bar(range(a['Metabolites'].values.size),a['Ranks'].values)
    plt.xticks(range(a['Metabolites'].values.size), a['Metabolites'],rotation =90) 
    plt.ylabel('Ranks coefficients')
    ax2 =ax.twinx()
    #sns.pointplot(x='Metabolites', y='Train_R^2', data=a, color='red')
    #sns.pointplot(x='Metabolites', y='Test_R^2', data=a, color='red')
    sns.tsplot(data=a['Train_R^2'], color='darkmagenta',condition = 'Training accuracy')
    sns.tsplot(data=a['Test_R^2'], color='m',condition = 'Testing accuracy',linestyle='--')
    ax2.set_ylim(0,1.01)
    ax2.set_xlim(-0.5,41.5)
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
