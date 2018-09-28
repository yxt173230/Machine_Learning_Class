# Assignment_3: Decision tree for continuous variable
# Student Name: Yu-Hsuan Tseng
#################################################################################
# There are two parts: (Part 1) Revise the in-class code to regression tree.
#                      (Part 2) Use Sklearn.
#################################################################################

import pandas as pd
import numpy as np

from __future__ import print_function

data = pd.read_csv('compustat_annual_2000_2017_with link information.csv')
data1 = data._get_numeric_data()
data2 = data1.dropna(subset=['oiadp'])

X = data2.loc[:, data2.columns != 'oiadp']
y = data2['oiadp']



############################################################################################
###### Part.1 Revise the in-class code to regression tree by using variance reduction ######

training_data = X.join(y)
columnlist = training_data.columns
training_data = training_data.values.tolist()


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

	
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row_in_yourlist):
        val = row_in_yourlist[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (columnlist[self.column], condition, str(self.value))
		

def partition(yourlist, question):
    true_rows, false_rows = [], []
    for row in yourlist:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


	
def Parent_Variance(yourlist):
    """Computes variance for split."""
    targetcolumn = []
    for row in yourlist:
        targetcolumn.append(row[-1])
    variance = np.var(targetcolumn)
    return variance

	
	
def Weighted_Variance(left, right):
    w = float(len(left)) / (len(left) + len(right))
    return w * Parent_Variance(left) + (1 - w) * Parent_Variance(right)	
	


def find_best_split(yourlist)
    best_var = Parent_Variance(yourlist)  
    best_question = None  
    parent_var = Parent_Variance(yourlist)
    n_features = len(yourlist[0]) - 1  # number of columns

    for col in range(n_features):  

        values = set([row[col] for row in yourlist])  

        for val in values:  

            question = Question(col, val)
            
            true_rows, false_rows = partition(yourlist, question)
            
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            
            weighted_var = Weighted_Variance(true_rows, false_rows)
            if parent_var > weighted_var:
                if best_var >= weighted_var:
                    best_var, best_question = weighted_var, question


    return best_var, best_question	

	
	
	
###################################################
###### Part.2 Use Sklearn #########################

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

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
