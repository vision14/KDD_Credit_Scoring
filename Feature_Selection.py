# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv("Credit Dataset.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = np.append(arr=np.ones((1000,1)).astype(int), values=X, axis=1)
X_opt = X
coef_labels = ['CONSTANT', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
               'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
               'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

while(True):
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary(xname=coef_labels))
    print()
    p_values = list(regressor_OLS.pvalues)
    
    p_max = max(p_values)
    p_max_idx = p_values.index(p_max)
    
    if(p_max > 0.05):
        X_opt = np.delete(X_opt, [p_max_idx], axis=1)
        del coef_labels[p_max_idx]
    else:
        break
    
