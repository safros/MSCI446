import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import imageio,io
import pydotplus
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing, tree
from dtreeviz.trees import *

df = pd.read_excel('DATAFILE.xlsx',skiprows=[0,1,2,3,4,5])
df.head()
#df[['Latitude']]
#df[['Location','Latitude','Longitude']][600:655]
df.dtypes

df['HasAssociatedDisaster'] = df['Associated Dis'].notnull()*1
df[['HasAssociatedDisaster','Associated Dis']]

subsetdf=df[['AvgPercipitationMonth','AvgMaxTempMonth','AvgMinTempMonth','DisasterSubgroupCoding','Disaster TypeCoding','CPI', 'Total Deaths']]
subsetdf

feature_names = ['HasAssociatedDisaster','AvgPercipitationMonth','AvgMaxTempMonth','AvgMinTempMonth','DisasterSubgroupCoding','Disaster TypeCoding','CPI', 'Total Deaths']
c = DecisionTreeClassifier(min_samples_split=100)
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], # X (you may want to include other features as well)
                                                    df.Target,         # y
                                                    test_size=0.1,
                                                    random_state = 42)    # consistent split
c.fit(X_train,y_train)

#def show_tree(tree, features, path):
tree=c
features=feature_names
  # make file stream to read/write
f = io.StringIO()
  # export the graph into dot format and save it to the io stream
export_graphviz(tree, out_file=f, feature_names=feature_names, class_names=['0','1','2','3','4','5'],
                  filled=True, rounded=True) # for nicer visualization
  # read the dot data and trnasform it into a png, then save it to path
pydotplus.graph_from_dot_data(f.getvalue()).write_png('./results')
  # read the png image saved at path
img = imageio.imread('./results')
  # plot the png image in the notebook
plt.rcParams['figure.figsize'] = (20,20)
plt.imshow(img)
plt.show()

# predict target of testing set
y_pred = c.predict(X_test)

from sklearn.metrics import accuracy_score

# get the accurancy
accuracy  = accuracy_score(y_test,y_pred)

print("Accuracy using Decision Tree: ",accuracy*100 )

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# create instance of random forest classifier
# set min_samples_split as 20 which maximizes the prediction accuracy in our case
# set criterion to entropy
# set max depth to 6
# set random state to 0 to get some result everytime
rf = RandomForestClassifier(min_samples_split=70, criterion="entropy",
                            max_depth=4, random_state=0)

# train model
rf.fit(X_train,y_train)

# predict testing data using model
rf_pred = rf.predict(X_test)

# get the accurancy
rf_accuracy  = accuracy_score(y_test,rf_pred)

print("Accuracy with parameter tuning: ", rf_accuracy*100 )

#def show_tree(tree, features, path):
tree=rf.estimators_[0]
features=feature_names
  # make file stream to read/write
f = io.StringIO()
  # export the graph into dot format and save it to the io stream
export_graphviz(tree, out_file=f, feature_names=feature_names, class_names=['0','1','2','3','4','5'],
                  filled=True, rounded=True) # for nicer visualization
  # read the dot data and trnasform it into a png, then save it to path
pydotplus.graph_from_dot_data(f.getvalue()).write_png('/content/drive/MyDrive/446/randomforest')
  # read the png image saved at path
img = imageio.imread('/content/drive/MyDrive/446/randomforest')
  # plot the png image in the notebook
plt.rcParams['figure.figsize'] = (20,20)
plt.imshow(img)
plt.show()


clf = DecisionTreeClassifier(min_samples_split=100,
                             criterion='entropy',
                             max_depth=5)

# train the model
clf.fit(X_train,y_train)

# predict the values of the testing
clf_pred = clf.predict(X_test)

# get the accurancy
clf_accuracy  = accuracy_score(y_test,clf_pred)

print("Accuracy with parameter tuning: ", clf_accuracy*100 )

X, y = make_hastie_10_2(random_state=6)
X_train, X_test = subsetdf[:600], subsetdf[600:]
y_train, y_test = df['Target'][:600], df['Target'][600:]
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)

# predict testing data using model
clf_pred = clf.predict(X_test)

# get the accurancy
clf_accuracy  = accuracy_score(y_test,clf_pred)

print("Accuracy with parameter tuning: ", clf_accuracy*100 )
