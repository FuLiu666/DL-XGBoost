#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import joblib

f = open('lf_model.pickle','rb')
lf_model = pickle.load(f)
f.close()


from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
data= pd.read_excel("C:\\Users\\ASUS\\Desktop\\pre.xlsx",sheet_name=0)
selected_columns = ["log KOW","V","B"]

df = data[selected_columns]

original_scaler = joblib.load('lf_original_scaler.pkl')
standard_scaler = original_scaler.transform(df)
standard_scaler_df = pd.DataFrame(standard_scaler, columns=df.columns)

pre_chemicals=standard_scaler_df.iloc[:,:]
chemicals_pre_values=lf_model.predict(pre_chemicals)
chemicals_pre_values1=pd.DataFrame(chemicals_pre_values)

results_scaler = joblib.load('lf_pre_scaler.pkl')
pre_values = results_scaler.inverse_transform(chemicals_pre_values1)
pre_values=pd.DataFrame(pre_values)
pre_values.to_excel(r'C:\Users\ASUS\Desktop\predicted-Koc.xlsx')

print(pre_values)
print("Done")

