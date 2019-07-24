# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd

pd.set_option('display.width', 1000)

# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# Import Sci-Kit Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold

# Ensemble Models


# pip install xgboost

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Package for stacking models
from vecstack import stacking

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# print(os.listdir("../input"))

from IPython.display import display, HTML
display(HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
"""))

        
os.getcwd()        
os.chdir("C:\\Users\\hp\\Downloads\\housing_price_kaggle")
                
train = pd.read_csv("train.csv", index_col='Id')
test = pd.read_csv("test.csv", index_col='Id')


####################################################################
####################################################################
train1 = pd.read_csv("train.csv", index_col='Id')

count=0
count1=0
for col in train1.columns:
    if train1[col].dtype == 'object':
        print(col, "\n", train1[col].value_counts())
        count=count + len((train1[col].unique()).tolist()) -2
    else:
        print(col, "\n")
        count1 += 1
 
    
df= train1    
    
label_encoders = {}
for col in cat_columns:
    print("Encoding {}".format(col))
    new_le = LabelEncoder()
    df_processed[col] = new_le.fit_transform(df[col])
    label_encoders[col] = new_le    
    
    
    
    
    
ohe = OneHotEncoder(categorical_features=cat_columns_idx, 
                    sparse=False, handle_unknown="ignore")
df_processed_np = ohe.fit_transform(df_processed)    



cat_columns = ["city", "transport"]
df_processed = pd.get_dummies(df, prefix_sep="__",
                              columns=cat_columns)






train2=pd.DataFrame()
train3 = pd.concat([train2, train1]




for col in train1.columns:
    if train1[col].dtype == 'object':
        print(col, "\n")          
        
        labelencoder_X = LabelEncoder()
        X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
        onehotencoder = OneHotEncoder(categorical_features = [0])
        X = onehotencoder.fit_transform(X).toarray()
        # Encoding the Dependent Variable
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)
        
        label_enc = LabelEncoder()    
        onehot_encoder = OneHotEncoder(sparse=False)

        integer_encoded = label_enc.fit_transform(train1.loc[:,col])    
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(integer_encoded))
        onehot_encoded = onehot_encoded.iloc[:,1:]
        #print(onehot_encoded)
        train2 = pd.concat([train2, onehot_encoded], axis=1,ignore_index=True)

    else:
        print(col, "\n")





        
for col in train1.columns:
    if train1[col].dtype == 'object':
        print(col, "\n")          

        label_enc = LabelEncoder()    
        onehot_encoder = OneHotEncoder(sparse=False)

        integer_encoded = label_enc.fit_transform(train1.loc[:,col])    
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(integer_encoded))
        onehot_encoded = onehot_encoded.iloc[:,1:]
        #print(onehot_encoded)
        train2 = pd.concat([train2, onehot_encoded], axis=1,ignore_index=True)

    else:
        print(col, "\n")
        
        
        
object_cols = list(train1.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    # Encode the categorical columns with numbers    
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
    


####################################################################
####################################################################






def show_all(df):
    #This fuction lets us view the full dataframe
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):
        display(df)



show_all(train.head())

train.info()

desc = train.describe()


len(train.columns)
type(train.columns)
train.shape

ar1 = np.array(train.columns)
missing = train.isnull().sum()
varnames = (train.isnull().sum() < 200)
ar1[varnames]
varnames.shape
ar1[varnames].shape
ar1[varnames].tolist()

ls1 = ar1[varnames].tolist()
train.loc[:,ls1].shape

train = train.loc[:,ls1]



missing = test.isnull().sum()
print(missing)
    
not_missing = missing[missing < 200]
print(not_missing.index)
ls1 = not_missing.index.tolist()
test.loc[:,ls1].shape

test = test.loc[:,ls1]
    




# Plot missing values 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    print(missing)
    
    missing = missing[missing > 0]
    print(missing)
    #missing.sort_values(inplace=True)
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(12,8))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    
    # search for missing data
#    import missingno as msno
#    msno.matrix(df=df, figsize=(16,8), color=(0,0.2,1))
    
plot_missing(train)
plot_missing(test)


print(train.columns)
varlist = ["Alley",Fire]
(train)
train<-train[]

# # IMPUTING MISSING VALUES
def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns 
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)


fill_missing_values(train)
train.isnull().sum().max()

# Using the function written above to visualize missing values
plot_missing(test)

fill_missing_values(test)
test.isnull().sum().max()




def impute_cats(df):
    '''This function converts categorical and non-numeric 
       columns into numeric columns to feed into a ML algorithm'''
    # Find the columns of object type along with their column index
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))

    # Encode the categorical columns with numbers    
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
        onehotencoder = OneHotEncoder(categorical_features = [i])
        df = onehotencoder.fit_transform(df).toarray()




# Impute the missing values
impute_cats(train)
impute_cats(test)
print("Train Dtype counts: \n{}".format(train.dtypes.value_counts()))
print("Test Dtype counts: \n{}".format(test.dtypes.value_counts()))




corr_mat = train[["SalePrice","MSSubClass","MSZoning","LotFrontage","LotArea", "BldgType",
                       "OverallQual", "OverallCond","YearBuilt", "BedroomAbvGr", "PoolArea", "GarageArea",
                       "SaleType", "MoSold"]].corr()
# corr_mat = train.corr()
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(corr_mat, vmax=1 , square=True)

f, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='YearBuilt', y='SalePrice', data=train)

f, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='OverallQual', y='SalePrice', color='green',data=train)


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train['SalePrice'])

sns.catplot(x='SaleType', y='SalePrice', data=train, kind='bar', palette='muted')


X = train.drop('SalePrice', axis=1)
y = np.ravel(np.array(train[['SalePrice']]))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))

# Initialize the model
random_forest = RandomForestRegressor(n_estimators=50,
                                      max_depth=15,
                                      min_samples_split=5,
                                      min_samples_leaf=5,
                                      max_features=9,
                                      random_state=42,
                                      oob_score=True
                                     )
# Fit the model to our data
random_forest.fit(X_train, y_train)

# Make predictions on test data
rf_pred = random_forest.predict(X_test)

#cross validation k fold
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=random_forest, X=X_train, y=y_train, cv=5)
print(all_accuracies.mean())



#tuning grid with cross validation
from sklearn.model_selection import GridSearchCV

help(RandomForestRegressor)

grid_param = {
    'n_estimators': [100, 500, 1000],
    'bootstrap': [True, False]
}

help(GridSearchCV)
gd_sr = GridSearchCV(estimator=RandomForestRegressor(),
                     param_grid=grid_param,
                     cv=5,
                     n_jobs=-1)

help(gd_sr.fit)
gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)





# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(random_forest, X, y, cv=kf, n_jobs=-1)
y_pred.mean()



# Initialize our model
xg_boost = XGBRegressor( learning_rate=0.01,
                         n_estimators=6000,
                         max_depth=4, min_child_weight=1,
                         gamma=0.6, subsample=0.7,
                         colsample_bytree=0.2,
                         objective='reg:linear', nthread=-1,
                         scale_pos_weight=1, seed=27,
                         reg_alpha=0.00006
                       )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(xg_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()


# Fit our model to the training data
xg_boost.fit(X,y)

# Make predictions on the test data
xgb_pred = xg_boost.predict(test)




g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,
                                     max_depth=5, max_features='sqrt',
                                     min_samples_leaf=15, min_samples_split=10,
                                     loss='ls', random_state =42
                                   )

# Perform cross-validation to see how well our model does 
kf = KFold(n_splits=5)
y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)
y_pred.mean()

# Fit our model to the training data
g_boost.fit(X,y)

# Make predictions on test data
gbm_pred = g_boost.predict(test)



# Initialize our model
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=6,
                                       learning_rate=0.01, 
                                       n_estimators=6400,
                                       verbose=-1,
                                       bagging_fraction=0.80,
                                       bagging_freq=4, 
                                       bagging_seed=6,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                    )

# Perform cross-validation to see how well our model does
kf = KFold(n_splits=5)
y_pred = cross_val_score(lightgbm, X, y, cv=kf)
print(y_pred.mean())


# Fit our model to the training data
lightgbm.fit(X,y)

# Make predictions on test data
lgb_pred = lightgbm.predict(test)







# List of the models to be stacked
models = [g_boost, xg_boost, lightgbm, random_forest]
# Perform Stacking
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,
                           regression=True,
                           mode='oof_pred_bag',
                           metric=rmse,
                           n_folds=5,
                           random_state=25,
                           verbose=2
                          )


# Initialize 2nd level model
xgb_lev2 = XGBRegressor(learning_rate=0.1, 
                        n_estimators=500,
                        max_depth=3,
                        n_jobs=-1,
                        random_state=17
                       )

# Fit the 2nd level model on the output of level 1
xgb_lev2.fit(S_train, y_train)



# Make predictions on the localized test set
stacked_pred = xgb_lev2.predict(S_test)
print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))

g_boost_pred = g_boost.predict(X_test)
print("RMSE of g_boost Model: {}".format(rmse(y_test,g_boost_pred)))

xg_boost_pred = xg_boost.predict(X_test)
print("RMSE of xg_boost Model: {}".format(rmse(y_test,xg_boost_pred)))

lightgbm_pred = lightgbm.predict(X_test)
print("RMSE of lightgbm Model: {}".format(rmse(y_test,lightgbm_pred)))

random_forest_pred = random_forest.predict(X_test)
print("RMSE of random_forest Model: {}".format(rmse(y_test,random_forest_pred)))




y1_pred_L1 = models[0].predict(test)
y2_pred_L1 = models[1].predict(test)
y3_pred_L1 = models[2].predict(test)
y4_pred_L1 = models[3].predict(test)
S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1, y4_pred_L1]


test_stacked_pred = xgb_lev2.predict(S_test_L1)

# Save the predictions in form of a dataframe
submission = pd.DataFrame()

submission['Id'] = np.array(test.index)
submission['SalePrice'] = test_stacked_pred

submission.to_csv('submission.csv', index=False)






g_boost_pred = g_boost.predict(test)

submission = pd.DataFrame()

submission['Id'] = np.array(test.index)
submission['SalePrice'] = g_boost_pred

submission.to_csv('submission.csv', index=False)

























































