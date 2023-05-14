import arff, numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import math
import random
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# Age: Any ages in years when a women during pregnant.
# SystolicBP: Upper value of Blood Pressure in mmHg, another significant attribute during pregnancy.
# DiastolicBP: Lower value of Blood Pressure in mmHg, another significant attribute during pregnancy.
# BS: Blood glucose levels is in terms of a molar concentration, mmol/L.
# HeartRate: A normal resting heart rate in beats per minute.
# Risk Level: Predicted Risk Intensity Level during pregnancy considering the previous attribute.

df = pd.read_csv("Maternal Health Risk Data Set.csv")
print(df)

# percentile list
perc =[.20, .40, .60, .80]
  
# list of dtypes to include
include =['object', 'float', 'int']
  
# calling describe method
desc = df.describe(percentiles = perc, include = include)
  
# display
print(desc)
#print(df)

X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    df['RiskLevel'], random_state=0)
NB = GaussianNB()  
NB.fit(X_train, y_train)   
y_predict = NB.predict(X_test)  
print("Accuracy Test Set NB: {:.2f}".format(NB.score(X_test, y_test)))
print("Accuracy Train Set NB: {:.2f}".format(NB.score(X_train, y_train)))

print('Classification report \n', classification_report(y_test, y_predict))


cm = confusion_matrix(y_test, y_predict)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[2,1] + cm[2,2] + cm[1,2] + cm[1,1])

print('\nFalse Positives(FP) = ', cm[1,0] + cm[2,0])

print('\nFalse Negatives(FN) = ', cm[0,1] + cm[0,2])

##################################################################################
#Trying to redo the Naive Bayes but from scratch without packages

#prior = P('high risk')
#P(high risk) = how many high risk appears / total observations
#P(risk levels) for low high and mid risk
prior = df.groupby(by = "RiskLevel").size().div(len(df))
#prob values grouped by risk level
#print(prior)

#next calculate likelihood for each of the features in the dataset
#P(Age|risk level)
#P(SystolicBP|risk level)
#P(DiastolicBP| risk level)
#P(BS | risk level)
#P(HeartRate | risk level)

likelihood = {}
likelihood['BodyTemp'] = df.groupby(['RiskLevel', 'BodyTemp']).size().div(len(df)).div(prior)
likelihood['HeartRate'] = df.groupby(['RiskLevel', 'HeartRate']).size().div(len(df)).div(prior)
likelihood['BS'] = df.groupby(['RiskLevel', 'BS']).size().div(len(df)).div(prior)
likelihood['DiastolicBP'] = df.groupby(['RiskLevel', 'DiastolicBP']).size().div(len(df)).div(prior)
likelihood['SystolicBP'] = df.groupby(['RiskLevel', 'SystolicBP']).size().div(len(df)).div(prior)
likelihood['Age'] = df.groupby(['RiskLevel', 'Age']).size().div(len(df)).div(prior)

#print(likelihood)


#sorting everything by values lowest to highest
example = df.sort_values(by=['Age','SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
#print(example)

#sorting the data for ages <= 25 and more than 25
less_than25 = example[example['Age'] <= 25]
#print(less_than25)

more_than25 = example[example['Age'] > 25]
#print(more_than25)

#prob of low high and midrisk based on data
# p_highrisk = likelihood['Age']['']
# p_lowrisk = 
# p_midrisk = 

prior_lt25 = less_than25.groupby(by = "RiskLevel").size().div(len(less_than25))
#prob values grouped by risk level
print("RISK LEVEL PROBABILITY FOR UNDER 25 GROUP\n")
print(prior_lt25)

prior_mt25 = more_than25.groupby(by = "RiskLevel").size().div(len(more_than25))
#prob values grouped by risk level
print("RISK LEVEL PROBABILITY FOR OVER 25 GROUP\n")
print(prior_mt25)

print("As you can see, there is a higher probability that the over 25 group has high or mid risk people compared to the less than 25 group")

likelihood2 = {}
likelihood2['BodyTemp'] = less_than25.groupby(['RiskLevel', 'BodyTemp']).size().div(len(less_than25)).div(prior_lt25)
likelihood2['HeartRate'] = less_than25.groupby(['RiskLevel', 'HeartRate']).size().div(len(less_than25)).div(prior_lt25)
likelihood2['BS'] = less_than25.groupby(['RiskLevel', 'BS']).size().div(len(less_than25)).div(prior_lt25)
likelihood2['DiastolicBP'] = less_than25.groupby(['RiskLevel', 'DiastolicBP']).size().div(len(less_than25)).div(prior_lt25)
likelihood2['SystolicBP'] = less_than25.groupby(['RiskLevel', 'SystolicBP']).size().div(len(less_than25)).div(prior_lt25)
likelihood2['Age'] = less_than25.groupby(['RiskLevel', 'Age']).size().div(len(less_than25)).div(prior_lt25)

likelihood3 = {}
likelihood3['BodyTemp'] = more_than25.groupby(['RiskLevel', 'BodyTemp']).size().div(len(more_than25)).div(prior_mt25)
likelihood3['HeartRate'] = more_than25.groupby(['RiskLevel', 'HeartRate']).size().div(len(more_than25)).div(prior_mt25)
likelihood3['BS'] = more_than25.groupby(['RiskLevel', 'BS']).size().div(len(more_than25)).div(prior_mt25)
likelihood3['DiastolicBP'] = more_than25.groupby(['RiskLevel', 'DiastolicBP']).size().div(len(more_than25)).div(prior_mt25)
likelihood3['SystolicBP'] = more_than25.groupby(['RiskLevel', 'SystolicBP']).size().div(len(more_than25)).div(prior_mt25)
likelihood3['Age'] = more_than25.groupby(['RiskLevel', 'Age']).size().div(len(more_than25)).div(prior_mt25)

#print(likelihood3)
#the likelihoods can be printed if you want, Likelihood1 is before sorting everything 
#like2 is for <=25 and like3 is for >25 

###############################################################
#Trying to retest the NB to see if using the split up data is better than before
X_train, X_test, y_train, y_test = train_test_split(less_than25[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    less_than25['RiskLevel'], random_state=0)
NB = GaussianNB()  
NB.fit(X_train, y_train)   
y_predict = NB.predict(X_test)  
print("Accuracy Test Set NB: {:.2f}".format(NB.score(X_test, y_test)))
print("Accuracy Train Set NB: {:.2f}".format(NB.score(X_train, y_train)))

print('Classification report for Less-Than 25 Group \n', classification_report(y_test, y_predict))

X_train, X_test, y_train, y_test = train_test_split(more_than25[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    more_than25['RiskLevel'], random_state=0)
NB = GaussianNB()  
NB.fit(X_train, y_train)   
y_predict = NB.predict(X_test)  
print("Accuracy Test Set NB: {:.2f}".format(NB.score(X_test, y_test)))
print("Accuracy Train Set NB: {:.2f}".format(NB.score(X_train, y_train)))

print('Classification report for More-Than 25 Group \n', classification_report(y_test, y_predict))
print("If I split up the data into two separate data tables, one for people ages less than 25 and the other for ages more than 25, it gives a higher accuracy of classifying the risk levels of the groups.")

###########################################################################
#Create a Decision Tree for the data
#Going to rename the high mid and low risk values to: 
# Low Risk = 0
# Mid Risk = 1
# High Risk = 2

d = {'low risk':0, 'mid risk':1, 'high risk':2}
df['RiskLevel'] = df['RiskLevel'].map(d)
print(df)

#Then separate the feature and target columns 
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
x = df[features]
y = df['RiskLevel']

# print(x)
# print(y)

dtree = DecisionTreeClassifier(max_depth = 5, min_samples_leaf=5)
dtree = dtree.fit(x,y)
axe = plt.subplots(figsize=(20,10))
tree.plot_tree(dtree, feature_names = features)

plt.savefig('out.pdf')


#Using train test split to make the decision tree, splitting up the data into less than 25 
X_train, X_test, y_train, y_test = train_test_split(less_than25[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    less_than25['RiskLevel'], random_state=0)
tree_lt25 = dtree.fit(X_train, y_train)
axe = plt.subplots(figsize=(20,10))
tree.plot_tree(tree_lt25, feature_names = features)
plt.savefig('outlt25.pdf')
#VALUE in the results can be a low, mid, or high risk. 
#value = [low risk, mid risk, high risk]

#using the train test split to make dec. tree using the split data for people more than 25
X_train, X_test, y_train, y_test = train_test_split(more_than25[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    more_than25['RiskLevel'], random_state=0)
tree_MT25 = dtree.fit(X_train, y_train)
axe = plt.subplots(figsize=(20,10))
tree.plot_tree(tree_MT25, feature_names = features)
plt.savefig('outMT25.pdf')


X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']], 
                                                    df['RiskLevel'], random_state=0)
tree_ALLtts = dtree.fit(X_train, y_train)
axe = plt.subplots(figsize=(20,10))
tree.plot_tree(tree_ALLtts, feature_names = features)
plt.savefig('outALL.pdf')
# print(df)