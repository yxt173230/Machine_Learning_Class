# Assignment_2


import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('compustat_annual_2000_2017_with link information.csv')

# Clean data
missing_percentage = data.isnull().sum()/data.shape[0]
data1 = data.loc[:, missing_percentage < 0.7]
data2 = data1._get_numeric_data()
data3 = data2.fillna(data2.median())

X = data3.loc[:, data3.columns != 'oiadp']
y = data3['oiadp']


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.06, 
                       verbose=True):
# X - candidate predictor variables
# y - Response
# threshold_in - include a feature if its p-value < threshold_in
# threshold_out - exclude a feature if its p-value > threshold_out
# verbose - whether to print the sequence of inclusions and exclusions

    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting predictors:')
print(result)