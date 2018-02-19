
# coding: utf-8

# In[3]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().magic(u'matplotlib inline')
from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np


# # Kernel map approximation and selective learning

# In[33]:

# 1. Use SVM to predict implied volatility


# In[3]:

cisc=pd.read_csv('cisc5352.quiz.6.option_data.csv')


# In[4]:

option=pd.read_csv('NBoption.csv')


# In[6]:

from sklearn.preprocessing import StandardScaler
scaler_cisc=StandardScaler()
scaler_option=StandardScaler()


# In[10]:

cisc.head()


# In[13]:

cisc_label=cisc['Implied Volatility']
cisc_data=cisc[['Time_to_Maturity','Interest_Rate']]


# In[11]:

option.head()


# In[14]:

option_label=option['ImpliedVolatility']
option_data=option[['time_to_maturity','LastPrice']]


# In[15]:

cisc_data=scaler_cisc.fit_transform(cisc_data)
option_data=scaler_option.fit_transform(option_data)


# In[16]:

test_percent=0.2
cisc_training_data,cisc_test_data,cisc_training_data_label, cisc_test_data_label=train_test_split(cisc_data,cisc_label, test_size=test_percent, random_state=42)

option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=42)


# In[17]:

# CISC dataset


# In[30]:

kernel_list=['linear','rbf','poly','sigmoid']
print "CISC dataset"
for kernel in kernel_list:
    clf=svm.SVR(kernel=kernel,tol=0.0001,gamma='auto')
    clf.fit(cisc_training_data,cisc_training_data_label)
    result=clf.predict(cisc_test_data)
    
    print "kernel is "+str(kernel)
    print "MSE for kernel "+str(kernel)+" is "+str(((cisc_test_data_label-result)**2).mean())
    print "\n"


# In[31]:

# Option dataset


# In[32]:

kernel_list=['linear','rbf','poly','sigmoid']
print "Option dataset"
for kernel in kernel_list:
    clf=svm.SVR(kernel=kernel,tol=0.0001,gamma='auto')
    clf.fit(option_training_data,option_training_data_label)
    result=clf.predict(option_test_data)
    
    print "kernel is "+str(kernel)
    print "MSE for kernel "+str(kernel)+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"


# In[26]:

# 2. Compare ists running time and running results of SVM, GB and RF.
# use Option data set


# In[48]:

# SVM

time_sum_10=0.0
error=0
x=10

for i in range(x):
    option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=i)
    
    start_time=time.time()
    
    clf=svm.SVR()
    clf.fit(option_training_data,option_training_data_label)
    result=clf.predict(option_test_data)
    
    time_sum_10 += (time.time()-start_time)
    print "Count time total is "+ str(time_sum_10) 
    error += ((option_test_data_label-result)**2).mean()
    print "MSE for SVM "+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"

print "Average time for one SVM is "+str (time_sum_10/x)
print "MSE for SVM is "+str(error/x)


# In[41]:

from sklearn.ensemble import GradientBoostingRegressor


# In[46]:

# GB

time_sum_10=0.0
error=0
x=10

for i in range(x):
    option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=i)
    
    start_time=time.time()
    
    GB=GradientBoostingRegressor()
    GB.fit(option_training_data,option_training_data_label)
    result=GB.predict(option_test_data)
    
    time_sum_10 += (time.time()-start_time)
    print "Count time total is "+ str(time_sum_10) 
    error += ((option_test_data_label-result)**2).mean()
    print "MSE for GB "+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"

print "Average time for one GB is "+str (time_sum_10/x)
print "MSE for GB is "+str(error/x)


# In[44]:

from sklearn.ensemble import RandomForestRegressor


# In[47]:

# Random Forest Regressor
time_sum_10=0.0
error=0
x=10

for i in range(x):
    option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=i)
    
    start_time=time.time()
    
    RF=RandomForestRegressor()
    RF.fit(option_training_data,option_training_data_label)
    result=RF.predict(option_test_data)
    
    time_sum_10 += (time.time()-start_time)
    print "Count time total is "+ str(time_sum_10) 
    error += ((option_test_data_label-result)**2).mean()
    print "MSE for RFR "+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"

print "Average time for one RFR is "+str (time_sum_10/x)
print "MSE for RFR is "+str(error/x)


# After comparing three method with default setting . We can notice that Gradient Boost achieve a best MES with least time. While SVM performed better in MSE and worse in runing time than Random Forest Regressor.
# 

# In[49]:

# 3. to find best performance by learning to tunning parameter for Gradient Bosst


# In[50]:

# by setting different parameters, we could get different result.


# In[58]:

import numpy as np


# In[51]:


option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=42)


# In[77]:

#  we change number of estimators to get different result
k_range=np.arange(0.01,1,0.01)


# In[78]:

k_range


# In[80]:

Error=[]
for k in k_range:
    GB=GradientBoostingRegressor(learning_rate=k)
    GB.fit(option_training_data,option_training_data_label)
    result=GB.predict(option_test_data)
    
    Error.append(((option_test_data_label-result)**2).mean())
    


# In[89]:

plt.figure()

plt.plot(k_range,Error)
plt.title("Relationship between learning_rate and MSE")
plt.xlabel("Learning Rate")
plt.ylabel("MSE")


minimum=np.array(Error).min()
index=k_range[Error.index(minimum)]

plt.plot(index,minimum,'g*')
plt.text(index,minimum+0.005, '('+str(index)+ " , "+str(minimum)+')')


# In[90]:

# so we can notice that if we choose 0.1 as our learning_rate, we can achieve a
# better MSE


# In[91]:

# Do kernel map approximation by using RBF kernel and compare runing time and running result with 
# SVM


# In[92]:

from sklearn.kernel_approximation import RBFSampler


# In[94]:

n_d=50
rbf_feature=RBFSampler(n_components=n_d,gamma=1/2,random_state=1)


# In[95]:

X_features=rbf_feature.fit_transform(option_training_data)
test_features=rbf_feature.fit_transform(option_test_data)


# In[96]:

# SVM

time_sum_10=0.0
error=0
x=10

for i in range(x):
    option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=i)
    
    start_time=time.time()
    
    clf=svm.SVR()
    clf.fit(option_training_data,option_training_data_label)
    result=clf.predict(option_test_data)
    
    time_sum_10 += (time.time()-start_time)
    print "Count time total is "+ str(time_sum_10) 
    error += ((option_test_data_label-result)**2).mean()
    print "MSE for SVM "+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"

print "Average time for one SVM is "+str (time_sum_10/x)
print "MSE for SVM is "+str(error/x)


# In[97]:

# SVM with map approximation


# In[102]:

from sklearn.linear_model import SGDRegressor


# In[103]:

time_sum_10=0.0
error=0
x=10

for i in range(x):
    option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=i)
    X_features=rbf_feature.fit_transform(option_training_data)
    test_features=rbf_feature.fit_transform(option_test_data)
    
    start_time=time.time()
    
    clf=SGDRegressor()
    clf.fit(X_features,option_training_data_label)
    result=clf.predict(test_features)
    
    time_sum_10 += (time.time()-start_time)
    
    print "Count time total is "+ str(time_sum_10) 
    error += ((option_test_data_label-result)**2).mean()
    print "MSE for SVM with map approximation "+" is "+str(((option_test_data_label-result)**2).mean())
    print "\n"

print "Average time for one SVM with map approximation is "+str (time_sum_10/x)
print "MSE for SVM with approximation is "+str(error/x)


# In[104]:

# After the map approximation with SGDRegressor, we can notice that this method is
# faster than SVM and achieve better MSE.


# In[105]:

# 5. Apply selective learning to two datasets and use GB


# In[106]:

#cisc_training_data,cisc_test_data,cisc_training_data_label, cisc_test_data_label


# In[114]:

# CISC dataset


# In[132]:

cisc_data=pd.DataFrame(cisc_data,columns=['LastPrice','time_to_maturity'])
cisc_training_data,cisc_test_data,cisc_training_data_label, cisc_test_data_label=train_test_split(cisc_data,cisc_label, test_size=test_percent, random_state=42)


# In[133]:

# GB with original cisc data
GB=GradientBoostingRegressor()
GB.fit(cisc_training_data,cisc_training_data_label)
result=GB.predict(cisc_test_data)
print "MSE for GB "+" is "+str(((cisc_test_data_label-result)**2).mean())



# In[134]:

len(cisc_training_data)


# In[135]:

test_percent=0.2
train_train_data,train_test_data,train_train_label,train_test_label=train_test_split(cisc_training_data,cisc_training_data_label,test_size=test_percent,random_state=42)


# In[136]:

len(train_train_data)


# In[137]:

GB1=GradientBoostingRegressor()
GB1.fit(train_train_data,train_train_label)


# In[138]:

GB1_result=GB1.predict(train_test_data)


# In[139]:

bottom_GB=(abs(GB1_result-train_test_label)).sort_values(ascending=False).quantile(0.9)


# In[140]:

GB_keys=(((abs(GB1_result-train_test_label))[(abs(GB1_result-train_test_label))>bottom_GB])).keys()
GB_bad=(((abs(GB1_result-train_test_label))[(abs(GB1_result-train_test_label))>bottom_GB]))


# In[141]:

from sklearn.neighbors import NearestNeighbors


# In[142]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(train_train_data) 


# In[146]:

len(GB_keys)


# In[143]:

# bad neighbors in train_train_data
bad_neighbors=[]
for key in GB_keys:
    print key
    bad_neighbors.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[147]:

len(bad_neighbors)


# In[ ]:

unique_bad_neighbors=set(bad_neighbors)


# In[145]:

len(unique_bad_neighbors)


# In[ ]:




# In[152]:

clean_train_test_data=train_test_data.drop(GB_keys)
clean_train_test_label=train_test_label.drop(GB_keys)


# In[153]:

clean_train_train_data=train_train_data.drop(train_train_data.index[list(unique_bad_neighbors)])
clean_train_train_label=train_train_label.drop(train_train_data.index[list(unique_bad_neighbors)])


# In[154]:

# clean the test data set


# In[155]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(cisc_test_data) 


# In[156]:

bad_neighbors_test=[]
for key in GB_keys:
    print key
    bad_neighbors_test.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[157]:

unique_bad_neighbors_test=set(bad_neighbors_test)


# In[158]:

clean_test_data=cisc_test_data.drop(cisc_test_data.index[list(unique_bad_neighbors_test)])
clean_test_label=cisc_test_data_label.drop(cisc_test_data_label.index[list(unique_bad_neighbors_test)])


# In[159]:

frame1=[clean_train_train_data,clean_train_test_data]
frame2=[clean_train_train_label,clean_train_test_label]


# In[ ]:




# In[160]:

clean_train_data=pd.concat(frame1)
clean_train_label=pd.concat(frame2)


# In[165]:

len(clean_train_train_data)


# In[166]:

len(clean_train_test_data)


# In[167]:

len(clean_train_label)


# In[161]:

# GB with selective cisc data
GB=GradientBoostingRegressor()
GB.fit(clean_train_data,clean_train_label)
result=GB.predict(clean_test_data)
print "After selective learning, MSE for GB "+" is "+str(((clean_test_label-result)**2).mean())



# In[171]:

# Option dataset


# In[174]:

option_data=pd.DataFrame(option_data,columns=['LastPrice','time_to_maturity'])
option_training_data,option_test_data,option_training_data_label, option_test_data_label=train_test_split(option_data,option_label, test_size=test_percent, random_state=1)


# In[175]:

# GB with original cisc data
GB=GradientBoostingRegressor()
GB.fit(option_training_data,option_training_data_label)
result=GB.predict(option_test_data)
print "MSE for GB "+" is "+str(((option_test_data_label-result)**2).mean())



# In[176]:

test_percent=0.2
train_train_data,train_test_data,train_train_label,train_test_label=train_test_split(option_training_data,option_training_data_label,test_size=test_percent,random_state=2)


# In[177]:

GB1=GradientBoostingRegressor()
GB1.fit(train_train_data,train_train_label)


# In[178]:

GB1_result=GB1.predict(train_test_data)


# In[179]:

bottom_GB=(abs(GB1_result-train_test_label)).sort_values(ascending=False).quantile(0.9)


# In[180]:

GB_keys=(((abs(GB1_result-train_test_label))[(abs(GB1_result-train_test_label))>bottom_GB])).keys()
GB_bad=(((abs(GB1_result-train_test_label))[(abs(GB1_result-train_test_label))>bottom_GB]))


# In[181]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(train_train_data) 


# In[182]:

# bad neighbors in train_train_data
bad_neighbors=[]
for key in GB_keys:
    print key
    bad_neighbors.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[187]:

clean_train_test_data=train_test_data.drop(GB_keys)
clean_train_test_label=train_test_label.drop(GB_keys)


# In[190]:

unique_bad_neighbors=set(bad_neighbors)


# In[188]:




# In[191]:

clean_train_train_data=train_train_data.drop(train_train_data.index[list(unique_bad_neighbors)])
clean_train_train_label=train_train_label.drop(train_train_data.index[list(unique_bad_neighbors)])


# In[192]:

# clean the test data set


# In[193]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(option_test_data) 


# In[194]:

bad_neighbors_test=[]
for key in GB_keys:
    print key
    bad_neighbors_test.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[195]:

unique_bad_neighbors_test=set(bad_neighbors_test)


# In[196]:

clean_test_data=option_test_data.drop(option_test_data.index[list(unique_bad_neighbors_test)])
clean_test_label=option_test_data_label.drop(option_test_data_label.index[list(unique_bad_neighbors_test)])


# In[197]:

frame1=[clean_train_train_data,clean_train_test_data]
frame2=[clean_train_train_label,clean_train_test_label]


# In[198]:

clean_train_data=pd.concat(frame1)
clean_train_label=pd.concat(frame2)


# In[199]:

# GB with selective option data
GB=GradientBoostingRegressor()
GB.fit(clean_train_data,clean_train_label)
result=GB.predict(clean_test_data)
print "After selective learning, MSE for GB "+" is "+str(((clean_test_label-result)**2).mean())



# # Baby PCA

# In[3]:

X=np.array([[1,2,0],[7.2,5,9],[-3,100,5.8],[1,-90,9.7],[2,88,1.2]])


# In[4]:

X.shape


# In[5]:

X


# In[6]:

X.T


# In[ ]:

# 1. Compute its covariance matrix


# In[7]:

X_cov=np.cov(X.T)


# In[9]:

X_cov


# In[10]:

# 2. compute its PCs and variances


# In[11]:

Variances,PCs=np.linalg.eig(X_cov)


# In[15]:

PCs


# In[14]:

# first PCs
PCs[:,0]


# In[13]:

Variances


# In[18]:

np.matrix(PCs.T)*np.matrix(PCs)


# In[19]:

# new data 


# In[26]:

new_data=np.matrix(X)*np.matrix(PCs)
print new_data


# In[27]:

# retrive the original data 


# In[29]:

X


# In[28]:

np.matrix(new_data)*np.matrix(PCs.T)


# # PCA Application

# In[2]:

# python version of PCA analysis


# In[4]:

vehicle=pd.read_csv("vehicles.csv",index_col='index')


# In[5]:

vehicle.shape


# In[6]:

vehicle.head()


# In[7]:

# only last 11 variables are useful


# In[8]:

data=vehicle.iloc[:,7:18]


# In[9]:

data.head()


# In[10]:

data.describe()


# In[11]:

data=data.dropna()


# In[12]:

# normalize data 


# In[13]:

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[14]:

data=scaler.fit_transform(data)


# In[15]:

name=vehicle.columns[7:18]
data=pd.DataFrame(data,columns=name)


# In[16]:

data.head()


# In[17]:


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(15,8))
parallel_coordinates(data,'Cylinders')
plt.legend(loc=1)


# In[18]:

data_cov=np.cov(data.T)


# In[19]:

data_cov


# In[20]:

Variances,PCs=np.linalg.eig(data_cov)


# In[21]:

PCs


# In[22]:

# first two components
PCs[:,0:2]


# In[23]:

Variances


# In[24]:

plt.plot(Variances)
plt.xlabel('PC')
plt.ylabel('Variances')


# In[25]:

# Scores in the new cordinate system
scores=np.dot(data,PCs)


# In[26]:

index=vehicle.dropna().index
column_name=[  'PC'+str(i)    for i in range(1,12)]


# In[27]:

scores=pd.DataFrame(scores,index=index,columns=column_name)


# In[28]:

scores.head()


# In[29]:

explanedvariance=Variances/sum(Variances)
explanedvariance


# In[30]:

total=0
ratioofsum=np.cumsum(explanedvariance)


# In[31]:

plt.figure()
plt.scatter(range(1,12),ratioofsum)
plt.xticks(range(1,12))

plt.axhline(y=0.95, xmin=0, xmax=1,c='r')
plt.xlabel('sum of first i PCs ')
plt.ylabel('percentage of variance could be covered')


# # selective first 5 PC since the fifth point is above 0.95

# In[32]:

reduced_data=scores.iloc[:,0:5]


# In[33]:

reduced_data.head()


# In[34]:

reduced_data.shape


# In[35]:

xvector = PCs[:,0] # see 'prcomp(my_data)$rotation' in R
yvector = PCs[:,1]

xs = reduced_data['PC1']
ys = reduced_data['PC2']




## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them
plt.figure(figsize=(7,7))

for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(data.columns.values)[i], color='r',size='large')

for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo',alpha=0.01)
    plt.text(xs[i]*1.2, ys[i]*1.2, list(vehicle.index)[i], color='b',alpha=0.5)

plt.show()


# In[38]:

get_ipython().magic(u'matplotlib notebook')


# In[39]:

from mpl_toolkits.mplot3d import Axes3D
xvector = PCs[:,0] # see 'prcomp(my_data)$rotation' in R
yvector = PCs[:,1]
zvector = PCs[:,2]

xs = reduced_data['PC1']
ys = reduced_data['PC2']
zs = reduced_data['PC3']



## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them
fig=plt.figure(figsize=(10,10))
ax=Axes3D(fig)




for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    ax.quiver(0, 0, 0 ,xvector[i], yvector[i], zvector[i],
              color='r',length=5)
    ax.text(xvector[i]*5, yvector[i]*5 , zvector[i]*5,
             list(data.columns.values)[i], color='r',size='large')

for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    ax.scatter(xs[i], ys[i], zs[i] ,'bo',alpha=0.01)
    ax.text(xs[i]*1.2, ys[i]*1.2,zs[i]*1.2, list(vehicle.index)[i], color='b',alpha=0.5)

    
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


# # SP 2010_baby.csv

# In[66]:

stock=pd.read_csv('SP_2010__baby.csv')


# In[82]:

stock


# In[68]:

from sklearn.decomposition import PCA


# In[69]:

name=stock.columns


# In[71]:

name=name[3:]


# In[73]:

data=stock[name]


# In[74]:

data.head()


# In[83]:

pca=PCA(n_components=8)


# In[84]:

pca.fit(data)


# In[88]:

pca.transform(data).shape


# In[89]:

column_name=['PC'+str(i) for i in range(1,9)]


# In[90]:

column_name


# In[111]:

PC_data=pd.DataFrame(pca.transform(data),columns=column_name,index=stock['Name'])


# In[112]:

PC_data.head()


# In[113]:

PC_data['PC1'].sort_values(ascending=False)[:20]


# In[114]:

PC_data['PC2'].sort_values(ascending=False)[:20]


# In[115]:

PC_data['PC3'].sort_values(ascending=False)[:20]


# In[117]:

((PC_data**2).sum(axis=1)**(1/2)).sort_values(ascending=False)[:20]


# In[119]:

top_20=(((PC_data**2).sum(axis=1)**(1/2)).sort_values(ascending=False)[:20]).index


# In[121]:

PC_data.loc[top_20]


# In[ ]:

# They all have a high number of  PC1 

