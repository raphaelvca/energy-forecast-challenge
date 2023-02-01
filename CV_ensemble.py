import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import os
from datetime import datetime
import random
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import requests
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import regularizers

ft = pd.read_feather("data_import/feiertag.ftr")
ferien = pd.read_feather("data_import/ferien.ftr")
covid = pd.read_feather("data_import/covid.ftr")

def is_in_ferien(datum):
    
    res = False 
    for index, row in ferien.iterrows():
        # checking for date in range
        if datum >= row["date_from"] and datum <= row["date_to"]:
            res = True
    return res
    

TRAIN_DATA_PATH = 'data_import/train.csv'
TEST_DATA_PATH = 'data_import/test.csv'

## train_data
train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=['time'], parse_dates=['time'])
time = train_data.index
date_time_train = pd.to_datetime(train_data.index, format='%Y.%m.%d %H:%M:%S')
timestamp_s = date_time_train.map(pd.Timestamp.timestamp)
train_data['timestamp_s'] = timestamp_s
day = 24*60*60
year = (365.2425)*day
train_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
train_data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
train_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
train_data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
train_data['hour'] = train_data.index.hour
train_data['weekday'] = train_data.index.weekday
train_data['week'] = train_data.index.weekofyear
train_data['month'] = train_data.index.month
train_data['year'] = train_data.index.year
train_data['weekend'] = (train_data.weekday == 5) | (train_data.weekday == 6)
train_data['working_h'] = (train_data.hour >= 8) & (train_data.hour <= 18)
train_data['winter'] = (train_data.month >= 12) & (train_data.month <= 2)
train_data['transition'] = (train_data.month >= 3) & (train_data.month <= 5) | (train_data.month >= 8) & (train_data.month <= 11)

train_data["datum"] = train_data.index.strftime("%Y-%m-%d")
train_data = train_data.merge(ft, how="left", left_on="datum", right_on="datum")
train_data = train_data.merge(covid, how="left", left_on="datum", right_on="datum")

train_data = train_data.fillna(0)
train_data.index = time

train_data["is_ferien"] = list(map(is_in_ferien, date_time_train))

test_data = pd.read_csv(TEST_DATA_PATH, index_col=['time'], parse_dates=['time'])
index_test = test_data.index
date_time_test = pd.to_datetime(test_data.index, format='%Y.%m.%d %H:%M:%S')
timestamp_s_test = date_time_test.map(pd.Timestamp.timestamp)
test_data['timestamp_s'] = timestamp_s_test

test_data['hour'] = test_data.index.hour
test_data['weekday'] = test_data.index.weekday
test_data['week'] = test_data.index.week
test_data['month'] =  test_data.index.month
test_data['year'] = test_data.index.year
test_data['weekend'] = (test_data.weekday == 5) | (test_data.weekday == 6)
test_data['working_h'] = (test_data.hour >= 8) & (test_data.hour <= 18)
test_data['Day sin'] = np.sin(timestamp_s_test * (2 * np.pi / day))
test_data['Year sin'] = np.sin(timestamp_s_test * (2 * np.pi / year))
test_data['Day cos'] = np.cos(timestamp_s_test * (2 * np.pi / day))
test_data['Year cos'] = np.cos(timestamp_s_test * (2 * np.pi / year))
test_data['winter'] = (test_data.month >= 12) & (test_data.month <= 2)
test_data['transition'] = (test_data.month >= 3) & (test_data.month <= 5) | (test_data.month >= 8) & (test_data.month <= 11)


test_data["datum"] = test_data.index.strftime("%Y-%m-%d")
test_data = test_data.merge(ft, how="left", left_on="datum", right_on="datum")
test_data = test_data.merge(covid, how="left", left_on="datum", right_on="datum")

test_data = test_data.fillna(0)
test_data.index = index_test

test_data["is_ferien"] = list(map(is_in_ferien, date_time_test))



#p
predictors = ['Gb(i)', 'Gd(i)', 'H_sun', 'T2m', 'WS10m', 'hour', 'month', 'year', 'week']

"""
train_df = train_data[:int(len(train_data)*0.8)]
val_df = train_data[int(len(train_data)*0.8):]

#residual load

y_train = np.array(train_df["P"])
y_valid = np.array(val_df["P"])
X_train = np.array(train_df[predictors])
X_valid = np.array(val_df[predictors])

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))
# define tensorflow model
# define tensorflow model
model_tf = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(640, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])
# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
)


# compile model
model_tf.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit model
history = model_tf.fit(
    X_train, y_train,
    epochs=250,
    validation_data=(X_valid, y_valid),
    verbose=0, batch_size=6000, callbacks=[early_stopping])


plt.figure(figsize=(10, 6))
plt.plot(y_valid[:700], label='actual')
plt.plot(model_tf.predict(X_valid)[:1000], label='prediction')
plt.legend()
plt.show()
"""

params = {'num_leaves': 30,
          'n_estimators': 400,
          'max_depth': 8,
          'min_child_samples': 200,
          'learning_rate': 0.1,
          'subsample': 0.50,
          'colsample_bytree': 0.75
         }


params_rf = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 10}

#model = lgb.LGBMRegressor(**params)

#model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
#                                   max_depth=4, max_features='sqrt',
#                                   min_samples_leaf=15, min_samples_split=10, 
#                                   loss='huber', random_state =5)



#model = RandomForestRegressor()




#model = keras.models.Sequential([
#    keras.layers.Flatten(input_shape=[len(predictors), 1]),
#    keras.layers.Dense(1)
#])

#model.compile(loss="mse", optimizer="adam")

#model = ElasticNet(random_state=0)

#model = xgb.XGBRegressor()
"""
kf = KFold(n_splits=10, shuffle=True, random_state=123)

fold_metrics = []
for train_index, test_index in kf.split(train_data):
    cv_train, cv_test = train_data.iloc[train_index], train_data.iloc[test_index]
    # Train a model    
    model.fit(cv_train[predictors], cv_train['P'])
    # Make predictions    
    predictions = model.predict(cv_test[predictors])
    # Calculate the metric    
    metric = mean_squared_error(cv_test['P'], predictions, squared=False)    
    fold_metrics.append(metric)

np.mean(fold_metrics)
"""

n = len(train_data)
train_data_shuffled = train_data.sample(frac=1)
train_part1 = train_data_shuffled[0:int(n*0.5)]
train_part2 = train_data_shuffled[int(n*0.5):]

m_xgb = xgb.XGBRegressor()
m_xgb.fit(train_part1[predictors], train_part1['P'])

m_lgb = lgb.LGBMRegressor(**params)
m_lgb.fit(train_part1[predictors], train_part1['P'])


m_rf = RandomForestRegressor()
m_rf.fit(train_part1[predictors], train_part1['P'])

m_gb = GradientBoostingRegressor()
m_gb.fit(train_part1[predictors], train_part1['P'])



train_part2['m_xgb'] = m_xgb.predict(train_part2[predictors])
train_part2['m_lgb'] = m_lgb.predict(train_part2[predictors])
train_part2['m_rf'] = m_rf.predict(train_part2[predictors])
train_part2['m_gb'] = m_gb.predict(train_part2[predictors])
#train_part2['m_tf'] = model_tf.predict(train_part2[predictors])


predictors_2 = ['m_xgb', 'm_lgb', 'm_rf', #'m_tf', 'm_gb', 
                #'Gb(i)', 'Gd(i)', 'H_sun', 'T2m', 'WS10m', 
                #'hour', 'month', 'year', 'week'
                ]
    

# Create linear regression model without the intercept
#model = lgb.LGBMRegressor(**params)
model = LinearRegression(fit_intercept=False)




kf = KFold(n_splits=5, shuffle=True, random_state=123)

fold_metrics = []
for train_index, test_index in kf.split(train_part2):
    cv_train, cv_test = train_part2.iloc[train_index], train_part2.iloc[test_index]
    # Train a model    
    model.fit(cv_train[predictors_2], cv_train['P'])
    # Make predictions    
    predictions = model.predict(cv_test[predictors_2])
    # Calculate the metric    
    metric = mean_squared_error(cv_test['P'], predictions, squared=False)    
    fold_metrics.append(metric)

np.mean(fold_metrics)

train_part2['second_level_P'] = model.predict(train_part2[predictors_2])
#train_part2['second_level_P'] = train_part2[["m_xgb","m_lgb", "m_rf"]].mean(axis=1)

train_part2['second_level_P'][train_part2['H_sun'] == 0] = 0



#test
test_data['m_xgb'] = m_xgb.predict(test_data[predictors])
test_data['m_lgb'] = m_lgb.predict(test_data[predictors])
test_data['m_rf'] = m_rf.predict(test_data[predictors])
test_data['m_gb'] = m_gb.predict(test_data[predictors])
#test_data['m_tf'] = m_gb.predict(test_data[predictors])

test_data['P_Predict'] = model.predict(test_data[predictors_2])
#test_data['P_Predict'] = test_data[["m_xgb","m_lgb", "m_rf"]].mean(axis=1)



test_data['P_Predict'][test_data['H_sun'] == 0] = 0

df_compare = test_data[['m_xgb', 'm_lgb', 'm_rf','m_gb', 'P_Predict']]
df_compare.to_csv("df_compare_P_CV.csv")

######## load

predictors = ['Gb(i)', 'Gd(i)', 'H_sun', 'T2m', 'WS10m', 'hour', 'month', 'year', 'week',
              'weekday', 'weekend', 'working_h', 'is_feiertag', 'is_ferien', 'covcases']

"""
train_data.is_feiertag = train_data.is_feiertag.astype(int)
train_data.is_ferien = train_data.is_ferien.astype(int)
train_df = train_data[:int(len(train_data)*0.8)]
val_df = train_data[int(len(train_data)*0.8):]

#residual load

y_train = np.array(train_df["load"])
y_valid = np.array(val_df["load"])
X_train = np.array(train_df[predictors])
X_valid = np.array(val_df[predictors])

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train).astype('float32'))
# define tensorflow model
# define tensorflow model
model_tf = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(640, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])
# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
)


# compile model
model_tf.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit model
history = model_tf.fit(
    X_train.astype('float32'), y_train.astype('float32'),
    epochs=250,
    validation_data=(X_valid.astype('float32'), y_valid.astype('float32')),
    verbose=1, batch_size=6000, callbacks=[early_stopping])


plt.figure(figsize=(10, 6))
plt.plot(y_valid[:700], label='actual')
plt.plot(model_tf.predict(X_valid.astype('float32'))[:1000], label='prediction')
plt.legend()
plt.show()

""" 
n = len(train_data)
train_data_shuffled = train_data.sample(frac=1)
train_part1 = train_data_shuffled[0:int(n*0.5)]
train_part2 = train_data_shuffled[int(n*0.5):]

m_xgb = xgb.XGBRegressor()
m_xgb.fit(train_part1[predictors], train_part1['load'])

m_lgb = lgb.LGBMRegressor(**params)
m_lgb.fit(train_part1[predictors], train_part1['load'])


m_rf = RandomForestRegressor()
m_rf.fit(train_part1[predictors], train_part1['load'])

m_gb = GradientBoostingRegressor()
m_gb.fit(train_part1[predictors], train_part1['load'])



train_part2['m_xgb'] = m_xgb.predict(train_part2[predictors])
train_part2['m_lgb'] = m_lgb.predict(train_part2[predictors])
train_part2['m_rf'] = m_rf.predict(train_part2[predictors])
train_part2['m_gb'] = m_gb.predict(train_part2[predictors])
#train_part2['m_tf'] = model_tf.predict(train_part2[predictors].astype('float32'))


predictors_2 = ['m_xgb', 'm_lgb', 'm_rf', #'m_tf', 'm_gb', 
                #'Gb(i)', 'Gd(i)', 'H_sun', 'T2m', 'WS10m', 
                #'hour', 'month', 'year', 'week'
                ]
  

# Create linear regression model without the intercept
#model = lgb.LGBMRegressor(**params)
model = LinearRegression(fit_intercept=False)




kf = KFold(n_splits=5, shuffle=True, random_state=123)

fold_metrics = []
for train_index, test_index in kf.split(train_part2):
    cv_train, cv_test = train_part2.iloc[train_index], train_part2.iloc[test_index]
    # Train a model    
    model.fit(cv_train[predictors_2], cv_train['load'])
    # Make predictions    
    predictions = model.predict(cv_test[predictors_2])
    # Calculate the metric    
    metric = mean_squared_error(cv_test['load'], predictions, squared=False)    
    fold_metrics.append(metric)

np.mean(fold_metrics)


train_part2['second_level_load'] = model.predict(train_part2[predictors_2])

train_part2['residual_load_pred'] = train_part2['second_level_load'] - train_part2['second_level_P']
rmse_total = mean_squared_error(train_part2['residual_load'], train_part2['residual_load_pred'], squared=False)
print(rmse_total)



#test
test_data['m_xgb'] = m_xgb.predict(test_data[predictors])
test_data['m_lgb'] = m_lgb.predict(test_data[predictors])
test_data['m_rf'] = m_rf.predict(test_data[predictors])
test_data['m_gb'] = m_gb.predict(test_data[predictors])
#test_data['m_tf'] = m_gb.predict(test_data[predictors])

test_data['Load_Predict'] = model.predict(test_data[predictors_2])



df_compare = test_data[['m_xgb', 'm_lgb', 'm_rf','m_gb', 'Load_Predict']]
df_compare.to_csv("df_compare_load_CV.csv")




test_data['residual_load'] = test_data['Load_Predict'] - test_data['P_Predict']
test_submission = test_data['residual_load']
test_submission.to_csv('data_export/CV_ensemble.csv')





# hyperparam tuning
# blended with less features
# lag
# mean?
# bigger dataset
# week raus, covid raus
# sin/cos



"""

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(train_data[predictors], train_data['P'])

rf_random.best_params_

# predictions with linear regression in 2level
# laged predictions
# 2level with predictors


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

# create the parameter grid
param_grid = {
    'n_estimators': [30, 50, 100, 200, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# create the model to tune
rf = RandomForestRegressor()

# set up the grid search
grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter = 10, cv=3)

# fit the grid search to the data
grid_search.fit(train_data[predictors], train_data['P'])

# print the best hyperparameters
print(grid_search.best_params_)
"""