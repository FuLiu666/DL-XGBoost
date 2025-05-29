#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The optimal model
from sklearn import model_selection
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import pandas as pd
import numpy as np

df1 = pd.read_excel("C:\\Users\\ASUS\\Desktop\\clusters-StandardScaler.xlsx", sheet_name=0)
x1 = df1.iloc[:,7:]
y1 = df1["experimental  log Koc (L/kgOC)"]  
df2 = pd.read_excel("C:\\Users\\ASUS\\Desktop\\clusters-StandardScaler.xlsx", sheet_name=1)
x2 = df2.iloc[:,7:]
y2 = df2["experimental  log Koc (L/kgOC)"]  



x_train1,x_temp1 , y_train1,y_temp1 = train_test_split(x1, y1, test_size=0.2, random_state=99)
x_train2,x_temp2 , y_train2,y_temp2 = train_test_split(x2, y2, test_size=0.2, random_state=99)

x_temp = pd.concat([x_temp1, x_temp2], ignore_index=True)
y_temp = pd.concat([y_temp1, y_temp2], ignore_index=True)

x_train = pd.concat([x_train1, x_train2], ignore_index=True)
y_train= pd.concat([y_train1, y_train2], ignore_index=True)

x_test1, x_val1, y_test1, y_val1, = train_test_split(x_temp1, y_temp1, test_size=0.5, random_state=34)
x_test2, x_val2, y_test2, y_val2, = train_test_split(x_temp2, y_temp2, test_size=0.5, random_state=47)
    
x_test = pd.concat([x_test1, x_test2], ignore_index=True)
y_test= pd.concat([y_test1, y_test2], ignore_index=True)
x_val = pd.concat([x_val1, x_val2], ignore_index=True)
y_val= pd.concat([y_val1, y_val2], ignore_index=True)


params = {'tree_method': 'gpu_hist',  
    "objective":'reg:squarederror', "max_depth":6,
                             "min_child_weight":1,
                             "gamma":0,
                            "colsample_bytree":1,
                            "subsample":1,
                            "reg_alpha":0,
                            "reg_lambda":1,
                            "learning_rate":0.28,
                            "n_estimators":80
                        }
xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(x_train, y_train)
    
y_pred = xgb_model.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
print(f"RMSEtraining：{rmse}")
r2 = r2_score(y_train, y_pred)
print(f"R²training：{r2}")
mae=mean_absolute_error(y_train, y_pred)
print(f"MAEtraining：{mae}") 

y_pred = xgb_model.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"RMSEvalidation：{rmse}")
r2 = r2_score(y_val, y_pred)
print(f"R²validation：{r2}")
mae=mean_absolute_error(y_val, y_pred)
print(f"MAEvalidation：{mae}") 

y_pred = xgb_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSEtest：{rmse}")
r2 = r2_score(y_test, y_pred)
print(f"R²test：{r2}")
mae=mean_absolute_error(y_test, y_pred)
print(f"MAEtest：{mae}") 

