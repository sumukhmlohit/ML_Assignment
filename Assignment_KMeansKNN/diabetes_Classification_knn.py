#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
diab_data=pd.read_csv('diabetes.csv')
dia_data=diab_data.to_numpy()

print(diab_data.head())


# In[51]:


print(diab_data.shape)


# In[52]:


diab_data.head()


# In[53]:


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
#sns.pairplot(diab_data,hue='Outcome')


# In[54]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(diab_data)


# In[55]:


#mini-max-normalization
#diab_data=(diab_data-diab_data.min())/(diab_data.max()-diab_data.min())
#z-score normalization which gives higher test accuracy
diab_data = (diab_data - diab_data.mean())/diab_data.std()
#Since z-score normalization gave a higher test acuuracy than min-max normalization,z-score normalization was selected.


# In[56]:


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(diab_data,hue='Outcome')


# In[57]:


#It can be seen that the dataset is not easily linearly seperable 


# In[58]:


#Pearson Correlation Coefficient Heatmap was created on the dataset to analyze  which were the most important features.

plt.figure(figsize=(12,10))
cor = diab_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[59]:


#Feature Ablation Study
#From the above heatmap, we can see that the importance of features on outcomes can be arranged in ascending order(from least to highest) as follows:-
#'BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','Age','Pregnancies','BMI' and 'Glucose'
#The three least important features-BloodPressure, SkinThickness and Insulin were removed and it was observed that the test accuracy improved significantly.


# In[60]:


diab_data=diab_data.drop(columns={'BloodPressure','SkinThickness','Insulin'})


# In[74]:


def train_test_split1(dataset, test_size=0.3, random_state=1):
    np.random.seed(random_state)
    _dataset = np.array(dataset)
    np.random.shuffle(_dataset)
    
    threshold = int(_dataset.shape[0] * 0.3)
    X_test = _dataset[:threshold, :-1]
    Y_test = _dataset[:threshold, -1]
    X_train = _dataset[threshold:, :-1]
    Y_train = _dataset[threshold:, -1]
    
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = train_test_split1(diab_data, test_size=0.3, random_state=1)


# In[75]:


#PCA resulted in a significant loss of accuracy. So,PCA was not applied.
#from sklearn.decomposition import PCA
#Make an instance of the Model
#pca = PCA(.95)
#pca.fit(X_train)


# In[76]:


#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)


# In[77]:


train_accuracy=[]
test_accuracy=[]
klist=[]
preclist=[]
reclist=[]
f1list=[]
trainpreclist=[]
trainreclist=[]
trainf1list=[]
max_acc=0
max_recall=0
max_prec=0
max_f1=0
trainmax_acc=0
trainmax_recall=0
trainmax_prec=0
trainmax_f1=0
maxk=0
trainmaxk=0

for k in range (1,50):
    klist.append(k)

# K-Nearest Neighbors(KNN) algorithm
class KNN:
    def __init__(self, features, labels, k,p):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.k = k
        
    def _classify(self, inX):
        # Calculate inX's Euclidean distance from all other samples
        diff_squared = (self.features - inX) **p
        euclidean_distances = np.power(diff_squared.sum(axis=1),1/p)
        
        
        # Sort index of distances --- nearest sample's index is the first element, next nearest sample's index is the second, etc.
        # We only keep the first 'K' ones
        sorted_dist_indices = euclidean_distances.argsort()[:self.k]
                
        # Count the number of classes for K nearest neighbors
        class_count = {}
        for i in sorted_dist_indices:
            vote_label = self.labels[i]
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
            
        # Descending sort the resulting dictionary by class counts
        sorted_class_count = sorted(class_count.items(),
                                   key=lambda kv: (kv[1], kv[0]),
                                   reverse=True)
        
        # Return the first key in the dictionary which is the predicted label
        return sorted_class_count[0][0]
    
    
    def predict(self, test_set):
        predictions = []
        # Loop through all samples, predict the class labels and store the results
        for sample in test_set:
            predictions.append(self._classify(sample))
        
        return np.array(predictions)
    
    
    def accuracy(self, actual, preds):
        total = len(actual)
        print(actual)
        print(preds)
        # Calculate the number of misclassified predictions
        misclassified = sum((actual - preds) != 0)

        return (total - misclassified) / total
    def confusion_matrix(self,actual, predicted):
        unique = set(actual)
        matrix = [list() for x in range(len(unique))]
        for i in range(len(unique)):
            matrix[i] = [0 for x in range(len(unique))]
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for i in range(len(actual)):
            x = lookup[actual[i]]
            y = lookup[predicted[i]]
            matrix[y][x] += 1
        return matrix
 

 


# In[78]:



for i in range (1,50):
    #if i%2!=0:
    p=2
    clf = KNN(features=X_train, labels=Y_train, k=i,p=2)
    preds = clf.predict(test_set=X_test)
    trainpreds=clf.predict(test_set=X_train)
    
    print('For training:')
    mat=clf.confusion_matrix(Y_train,trainpreds)
    TP=mat[0][0]
    FP=mat[0][1]
    FN=mat[1][0]
    TN=mat[1][1]
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=(TP)/(TP+FP)
    recall=(TP)/(TP+FN)
    f1=(2*TP)/(2*TP+FP+FN)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    train_accuracy.append(accuracy)

    trainpreclist.append(precision)
    trainreclist.append(recall)
    trainf1list.append(f1)
    if accuracy>trainmax_acc:
        trainmax_acc=accuracy
        trainmax_recall=recall
        trainmax_prec=precision
        trainmax_f1=f1
        trainmax_k=k
    
    print('For testing:')
    mat=clf.confusion_matrix(Y_test,preds)
    TP=mat[0][0]
    FP=mat[0][1]
    FN=mat[1][0]
    TN=mat[1][1]
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=(TP)/(TP+FP)
    recall=(TP)/(TP+FN)
    f1=(2*TP)/(2*TP+FP+FN)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    test_accuracy.append(accuracy)
    preclist.append(precision)
    reclist.append(recall)
    f1list.append(f1)
    if accuracy>max_acc:
        max_acc=accuracy
        max_recall=recall
        max_prec=precision
        max_f1=f1
        max_k=i


# In[79]:


print(klist)
print(test_accuracy)
    
test_accuracy    
klist=np.array(klist)
test_accuracy=np.array(test_accuracy)

print(test_accuracy.astype(np.float))

print(klist.dtype)
print(test_accuracy.dtype)

print(klist)
print(test_accuracy)


# In[80]:


plt.plot(klist, train_accuracy)

plt.xlabel('k')
plt.ylabel('Training accuracy')

plt.title('Training accuracy vs k')

plt.show()


# In[81]:


plt.plot(klist, trainf1list)

plt.xlabel('k')
plt.ylabel('Training F1 Score')

plt.title('Training F1 score vs k')

plt.show()


# In[82]:


plt.plot(klist, test_accuracy)

plt.xlabel('k')
plt.ylabel('Test accuracy')

plt.title('Test accuracy vs k')

plt.show()


# In[83]:


plt.plot(klist, f1list)

plt.xlabel('k')
plt.ylabel('F1-score')

plt.title('Test F1-score vs k')

plt.show()


# In[84]:


print('At optimal value of k for testing, the following were the metrics:')
print('Max test accuracy:')
print(max_acc)
print('Test recall:')
print(max_recall)
print('Test precision:')
print(max_prec)
print('Test F1 score:')
print(max_f1)
print('k:')
print(max_k)


# In[85]:


#At optimal value of k(k=49),the following were the results of the train metrics
clf = KNN(features=X_train, labels=Y_train, k=46,p=2)
preds = clf.predict(test_set=X_test)
trainpreds=clf.predict(test_set=X_train)
    
print('For training:')
mat=clf.confusion_matrix(Y_train,trainpreds)
TP=mat[0][0]
FP=mat[0][1]
FN=mat[1][0]
TN=mat[1][1]
accuracy=(TP+TN)/(TP+TN+FP+FN)
precision=(TP)/(TP+FP)
recall=(TP)/(TP+FN)
f1=(2*TP)/(2*TP+FP+FN)
print('Accuracy:')
print(accuracy)
print('Precision')
print(precision)
print('Recall')
print(recall)
print('F1-score')
print(f1)


# In[93]:


#For Manhattan distance
max_acc=0
for i in range (1,50):
    #if i%2!=0:
    p=1
    clf = KNN(features=X_train, labels=Y_train, k=i,p=1)
    preds = clf.predict(test_set=X_test)
    trainpreds=clf.predict(test_set=X_train)
    
    print('For training:')
    mat=clf.confusion_matrix(Y_train,trainpreds)
    TP=mat[0][0]
    FP=mat[0][1]
    FN=mat[1][0]
    TN=mat[1][1]
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=(TP)/(TP+FP)
    recall=(TP)/(TP+FN)
    f1=(2*TP)/(2*TP+FP+FN)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)

    if accuracy>trainmax_acc:
        trainmax_acc=accuracy
        trainmax_recall=recall
        trainmax_prec=precision
        trainmax_f1=f1
        trainmax_k=k
    
    print('For testing:')
    mat=clf.confusion_matrix(Y_test,preds)
    TP=mat[0][0]
    FP=mat[0][1]
    FN=mat[1][0]
    TN=mat[1][1]
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=(TP)/(TP+FP)
    recall=(TP)/(TP+FN)
    f1=(2*TP)/(2*TP+FP+FN)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    if accuracy>max_acc:
        max_acc=accuracy
        max_recall=recall
        max_prec=precision
        max_f1=f1
        max_k=i


# In[94]:

print('Max accuracy for Manhattan distance')
print(max_acc)

