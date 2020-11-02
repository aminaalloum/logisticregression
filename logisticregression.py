#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
df=pd.read_csv("./Downloads/titanic-passengers.csv",sep=";")
df


# In[42]:


df.head(20)


# In[43]:


df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Survived"]=df["Survived"].map({"Yes":1,"No":0})


# In[44]:


df


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg=LogisticRegression()
x=df["Survived"].values.reshape(-1,1)
y=df["Sex"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[46]:


logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[47]:


df


# In[48]:


print("Accruacy={:.2f}".format(logreg.score(x_test,y_test)))


# In[49]:


import seaborn as sns
sns.regplot(x="Survived",y="Sex",data=df,logistic=True)


# In[50]:


confusion_matrix=pd.crosstab(y_test,y_pred,rownames=["actual"],colnames=["predicted"])
print(confusion_matrix)


# In[51]:


from sklearn.metrics import roc_curve
from sklearn.metrics import  roc_auc_score


# In[58]:



fpr,tpr,thresh=roc_curve(y_test,y_pred,pos_label=1)
random_prob=[0 for i in range(len(y_test))]
p_fpr,p_tpr,_=roc_curve(y_test,random_prob,pos_label=1)


# In[60]:


auc_score=roc_auc_score(y_test,y_pred)
print(auc_score)


# In[68]:


import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.plot(fpr,tpr,linestyle='-',color="red",label="KNN")
plt.plot(p_fpr,p_tpr,linestyle='--',color="blue",label="KNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC curve')
plt.show()


# In[72]:





# In[ ]:


The ROC curve help to visualize how well our machine learning classiffier is 
performing , works only for binary classification problems

AUC It tells how much model is capableof distinguishing between classes.Higher the AUC, better
the model is at predicting (max 1 , lower 0)



TPR:Sensitivity :tells us what proportion of the positive class got correctly classified.
TNR:Specificity: tells us what proportion of the negative class got correctly classified.
The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.
Sensitivity and Specificity are inversely proportional to each other. So when we increase 
Sensitivity, Specificity decreases and vice versa.


