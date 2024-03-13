import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm


data_train = pd.read_csv('data/train_df.csv')
data_test = pd.read_csv('data/test_df.csv')

data_train.drop(['search_id'], axis=1, inplace=True)
data_test.drop(['search_id'], axis=1, inplace=True)


list_col = list(data_train.columns)
list_col_bool = []

for i in list_col:
    if data_train[i].unique().size == 2 and np.array([0, 1]).all() in data_train[i].unique():
        list_col_bool.append(i)

for var in list_col_bool:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data_train[var], prefix=var)
    data1=data_train.join(cat_list)
    data_train=data1
cat_vars=['feature_3', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15']
data_vars=data_train.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data_train[to_keep]
data_final.columns.values

X = data_final.loc[:, data_final.columns.str.startswith('feature')]
y = data_final.loc[:, data_final.columns.str.fullmatch('target')]
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['target'])
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['target']==0]))
print("Number of subscription",len(os_data_y[os_data_y['target']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['target']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['target']==1])/len(os_data_X))


data_final_vars=data_train.columns.values.tolist()
y=['target']
X=[i for i in data_final_vars if i not in y]

logreg = LogisticRegression(max_iter=30000)
rfe = RFE(logreg, n_features_to_select=20, step=1)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
rfe.get_feature_names_out()
cols = ['feature_27', 'feature_28', 'feature_33', 
        'feature_70', 'feature_72', 'feature_3_0', 
        'feature_3_1', 'feature_9_0', 'feature_9_1', 
        'feature_10_0', 'feature_10_1', 'feature_11_0', 
        'feature_11_1', 'feature_12_0', 'feature_12_1', 
        'feature_13_0', 'feature_13_1', 'feature_14_0', 
        'feature_14_1']
X=os_data_X[cols]
y=os_data_y['target']

# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary2())

X_test = data_test.drop('target', axis=1)
y_test = data_test['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=30000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(ndcg_score([y_test], [y_pred]))
