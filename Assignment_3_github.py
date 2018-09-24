# Assignment_2: Stepwise Regression
# Student Name: Yu-Hsuan Tseng

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

data = pd.read_csv('compustat_annual_2000_2017_with link information.csv')
data1 = data._get_numeric_data()
data2 = data1.dropna(subset=['oiadp'])

X = data2.loc[:, data2.columns != 'oiadp']
y = data2['oiadp']


# Fit regression model
regr_max4 = DecisionTreeRegressor(max_depth=4)
regr_max5 = DecisionTreeRegressor(max_depth=5)
regr_max4 = regr_max4.fit(X, y)
regr_max5 = regr_max5.fit(X, y)


# Plot the decision tree
tree.export_graphviz(regr_max4, out_file='Tree with layer 4.dot')
tree.export_graphviz(regr_max5, out_file='Tree with layer 5.dot')

# Plot the decision tree:
# Create DOT data
dot_data1 = tree.export_graphviz(regr_max4, out_file=None)
dot_data2 = tree.export_graphviz(regr_max5, out_file=None)

# Draw graph
graph1 = pydotplus.graph_from_dot_data(dot_data1)  
graph2 = pydotplus.graph_from_dot_data(dot_data1)

# Show graph
Image(graph1.create_png())
Image(graph2.create_png())