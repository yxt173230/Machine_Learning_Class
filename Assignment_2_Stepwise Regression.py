### Assignment_2: Stepwise Regression
### Student Name: Yu-Hsuan Tseng


import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('compustat_annual_2000_2017_with link information.csv')

## Clean data and impute the missing values with median
missing_percentage = data.isnull().sum()/data.shape[0]
data1 = data.loc[:, missing_percentage < 0.7]
data2 = data1._get_numeric_data()
data3 = data2.fillna(data2.median())


## Make the column "Operating Income After Depreciation" as dependent variables Y
X = data3.loc[:, data3.columns != 'oiadp']
Y = data3['oiadp']


## Stepwise procedure
def stepwise_model(X, Y):
	added = []
    while True:
        changed=False
        # forward step: add the variables which has the lowest p-value and p-value is lower than 0.05 into the list
        candidates = list(set(X.columns)-set(added))
        added_pvalue = pd.Series(index=candidates) # it has index but no value
        for column in candidates:
            model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[added+[column]]))).fit()
            added_pvalue[column] = model.pvalues[column]
        min_pvalue = added_pvalue.min()
        if min_pvalue < 0.05:
            best_candidates = added_pvalue.argmin() # Returns the indice of the minimum values
            added.append(best_candidates)
            changed=True

	# backward step: step back to see if all the variables in the "added" list are all significant, if not, remove that variable
        model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[added]))).fit()
        # use all coefs except intercept
        p_values = model.pvalues.iloc[1:]
        max_pvalue = p_values.max() # null if pvalues is empty
        if max_pvalue > 0.050001:
            changed=True
            worst_candidates = p_values.argmax()
            added.remove(worst_candidates)

        if not changed:
            break
			
    return added

result = stepwise_model(X, Y)

print('Selected columns by doing Stepwise Regression: ', result)
