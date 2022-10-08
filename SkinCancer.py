#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
csv_1 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/Skin/GSE13355_series_matrix.txt/150MRMR.csv")
csv_2 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/Skin/GSE13355_series_matrix.txt/MutualInfo.csv")
csv_3 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/Skin/GSE13355_series_matrix.txt/F_class.csv")
intersection = list(set(csv_1) & set(csv_2) & set(csv_3))


# In[3]:


data11=csv_1[intersection]


# In[4]:


# importing necessary libraries  
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
# loading the iris dataset 
# X -> features, y -> label 
X = data11.drop('Class', axis=1)
y =data11['Class']
i=0
sum=0
for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =42) 
  
        # training a DescisionTreeClassifier 
        from sklearn.tree import DecisionTreeClassifier 
        dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train) 
        dtree_predictions = dtree_model.predict(X_test) 
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, dtree_predictions) 
        from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
        print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
        sum=sum+metrics.accuracy_score(y_test, dtree_predictions)
        from sklearn.metrics import classification_report
        print(classification_report(y_test, dtree_predictions))
        if i==19:
            print("OverAll accuracy=",sum/20)


# In[5]:


# importing necessary libraries 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
X3 = data11.drop('Class', axis=1)
y3 =data11['Class']
i=0
sum=0
for i in range(20):
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3,random_state = 42) 

        # training a Naive Bayes classifier 
        from sklearn.naive_bayes import GaussianNB 
        gnb = GaussianNB().fit(X_train3, y_train3) 
        gnb_predictions = gnb.predict(X_test3) 
        # accuracy on X_test 
        accuracy = gnb.score(X_test3, y_test3) 
        sum=sum+accuracy
        # creating a confusion matrix 
        cm = confusion_matrix(y_test3, gnb_predictions) 
        from sklearn.metrics import classification_report
        print(classification_report(y_test3, gnb_predictions))
        if i==19:
            print("Overall accuracy=", sum/20)


# In[6]:


# importing necessary libraries 

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
X2 = data11.drop('Class', axis=1)
y2 =data11['Class']
i=0
sum=0
for i in range(20):
    # dividing X, y into train and test data 
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2,test_size=0.3, random_state = 42) 

    # training a KNN classifier 
    from sklearn.neighbors import KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train2, y_train2) 

    # accuracy on X_test 
    accuracy = knn.score(X_test2, y_test2) 
    sum=sum+accuracy 

    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test2)  
    cm = confusion_matrix(y_test2, knn_predictions) 
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    print("Accuracy:",metrics.accuracy_score(y_test2, knn_predictions))
    from sklearn.metrics import classification_report
    print(classification_report(y_test2 ,knn_predictions))
    if i==19:
        print("Overall accuracy",sum/20)


# In[7]:


# importing necessary libraries 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split  
X1 = data11.drop('Class', axis=1)
y1 =data11['Class']
i=0
sum=0
for i in range (20):
    # dividing X, y into train and test data 
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1,test_size=0.3, random_state = 42) 

    # training a linear SVM classifier 
    from sklearn.svm import SVC 
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train1, y_train1) 
    svm_predictions = svm_model_linear.predict(X_test1) 
    accuracy = svm_model_linear.score(X_test1, y_test1)  
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test1, svm_predictions) 
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    print("Accuracy:",metrics.accuracy_score(y_test1, svm_predictions))
    sum=sum+metrics.accuracy_score(y_test1, svm_predictions)
    from sklearn.metrics import classification_report
    print(classification_report(y_test1, svm_predictions))
    if i==19:
        print("Overall accuracy=",sum/20)
    


# In[8]:


from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
X = data11.drop('Class', axis=1)
X = (X - X.min())/ (X.max() - X.min())
y =data11['Class']
i=0
sum=0
# ensure all data are floating point values
X = X.astype('float32')
for i in range (20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42) # 70% training and 30% test

        mlp = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                            max_iter = 200, solver = 'adam')
        # Train the classifier with the traning data
        mlp.fit(X_train,y_train)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Test set score: %f" % mlp.score(X_test, y_test))
        sum=sum+mlp.score(X_test, y_test)
        y_pred = mlp.predict(X_test)
        print(classification_report(y_test, y_pred))
        if i==19:
            print("Overall accuracy =",sum/20)


# In[9]:


import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
X3 = data11.drop('Class', axis=1)
y3 =data11['Class'] 

# dividing X, y into train and test data 
i=0
sum=0
for i in range(20):
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3,random_state = 42) 
        fanr = RandomForestClassifier().fit(X_train3, y_train3) 
        fanr_predictions = fanr.predict(X_test3) 
        # accuracy on X_test 
        accuracy = fanr.score(X_test3, y_test3) 
        sum=sum+ accuracy
        # creating a confusion matrix 
        cm = confusion_matrix(y_test3, fanr_predictions) 
        print(classification_report(y_test3, fanr_predictions))
        if i==19:
            print("overall Accuracy=",sum/20)
    


# In[ ]:




