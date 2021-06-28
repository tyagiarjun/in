#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary libraries
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt


# In[2]:


#loading the data set with the help of pandas
df=pd.read_csv('dataset.csv')
df.head()


# In[3]:


#command will tell about the number of rows and columns that are present in the dataset
df.shape


# In[4]:


#command will give us overview which column has how many null values
df.isnull().sum()


# In[5]:


df.nunique()


# In[6]:


#dropping the are_business column as its value is null for the whole dataset
df1=df.drop(['area_business'],axis=1)


# In[7]:


df1 = df1.dropna(axis=0, subset=['invoice_id'])


# In[8]:


df1.isnull().sum()


# In[9]:


#creating a new dataset which is transpose of the original dataset, we are doing this to remove the duplicate columns 
transposed=df1.T


# In[10]:


transposed.drop_duplicates(keep='first',inplace=True)


# In[11]:


#again taking transpose to convert the transposed data set into original dataset after removing duplicates
df2=transposed.T
df2.head()


# In[12]:


#Converting the amount in USD where the invoice currency is CAD
df2['amount_in_usd']=np.where(df2.invoice_currency=='CAD',df2.total_open_amount*0.81,df2.total_open_amount)


# In[13]:


df2.head(10)


# In[14]:


#splitting the dataset on the basis of Clear date as we have to predict the clearing date only so null values of clearing date are not useful
test=df2[df2['isOpen']==1]
test=test.drop(['isOpen'],axis=1)


# In[15]:


train=df2[df2['isOpen']==0]
train=train.drop(['isOpen'],axis=1)


# In[16]:



plt.boxplot(x=train["amount_in_usd"])              # graphs to detect outliers

plt                                                           


# In[17]:


plt.scatter(train["business_code"], train["amount_in_usd"])      #graph based on amount and business code distribution
plt


# In[18]:


plt.hist(train["amount_in_usd"])                   #to see my data is normally distributed or not
plt


# In[19]:


train=train.sort_values(by='posting_date')   #sorting dataset based on the posting date so that our model can learn some pattern


# In[20]:


train.head()  #show five entries from start


# In[21]:


train.dtypes       #data types of columns in  dataset


# In[22]:


train['clear_date']= pd.to_datetime(train['clear_date'])
train['posting_date']= pd.to_datetime(train['posting_date'])      #converting the data type of our dates


# In[23]:


train.nunique()      #return number of unique elements in the object


# In[24]:


train=train.drop(['document type','posting_id','invoice_currency'],axis=1)  #dropping document type and posting id because the only have one unique value and invoice currency because we have already converted our amount to USD


# In[25]:


train.head()


# In[26]:


train['document_create_date.1'] = pd.to_datetime(train['document_create_date.1'], format='%Y%m%d') #changing the format of date


# In[27]:


train['due_in_date'] = pd.to_datetime(train['due_in_date'], format='%Y%m%d') #changing the format of date


# In[28]:


train.head()


# In[29]:


train.head()


# In[30]:


train=train.drop(['doc_id','document_create_date','baseline_create_date'],axis=1)  #dopping these columns to remove duplicacy as we heave document_create_date already


# In[31]:


train.rename(columns = {"document_create_date.1": "create_date"}, inplace = True)  #keeping document_create_date.1 and renaming it because it is in the standard form


# In[32]:


train.head()


# In[33]:


train['late_days']=train['clear_date']-train['due_in_date']  #calculating the delay in number of days i.e after how many days from due_date the invoice was cleared 
 # Also it is our target variable


# In[34]:


train.isnull().sum()  #checking if we have any null values or not


# In[ ]:





# In[ ]:





# In[35]:


train.head()


# In[36]:


train.tail()


# # SPLITTING DATA ON THE BASIS OF DATE

# In[37]:


x_train=train[train['posting_date'] <= '2019-12-30'].copy()
x_temp=train[train['posting_date'] > '2019-12-31'].copy()


# In[38]:


x_validation=x_temp[x_temp['posting_date'] > '2020-01-01'].copy()


# In[39]:


x_train.shape


# In[40]:


x_validation.shape


# In[41]:


x_train.nunique()


# In[42]:


x_train.dtypes


# # FEATURE ENGINEERING

# In[43]:


x_train["late_days"] = (x_train["late_days"]).dt.days  #changing format of our target variable 


# In[44]:


x_validation["late_days"] = (x_validation["late_days"]).dt.days


# In[45]:


x_train.head()


# In[46]:


x_train["post_weekday"] = x_train["posting_date"].dt.dayofweek                    #train
x_train["post_month"] = x_train["posting_date"].dt.month
x_train["post_year"] = x_train["posting_date"].dt.year                      #Extracting features that will help to learn about some pattern

x_validation["post_weekday"] = x_validation["posting_date"].dt.dayofweek          #validation
x_validation["post_month"] = x_validation["posting_date"].dt.month
x_validation["post_year"] = x_validation["posting_date"].dt.year


# In[47]:


#due_in_date
x_train["due_weekday"] = x_train["due_in_date"].dt.dayofweek                   #train
x_train["due_month"] = x_train["due_in_date"].dt.month
x_train["due_year"] = x_train["due_in_date"].dt.year

x_validation["due_weekday"] = x_validation["due_in_date"].dt.dayofweek         #validation
x_validation["due_month"] = x_validation["due_in_date"].dt.month
x_validation["due_year"] = x_validation["due_in_date"].dt.year


# In[48]:


x_train.head()


# In[49]:


#as we saw our data set the are many companies with same customer number but different name so we are splitting the 
#name and saving the first part
#.copy()is used when we dont want our set to change when we change the original data set
x_train["name_customer"]= x_train.name_customer.str.split(' ').str[0].copy()    
x_validation["name_customer"]= x_validation.name_customer.str.split(' ').str[0].copy() 


# In[50]:


# calculatin difference in due date and posting date so that our model can learn some pattern from them
x_train["diff_duedate_psdate"] = (x_train["due_in_date"]-x_train["posting_date"]).dt.days

x_validation["diff_duedate_psdate"] = (x_validation["due_in_date"] - x_validation["posting_date"]).dt.days


# In[51]:


x_train.head()


# In[52]:


# extracting first five digit of cust_number and storing back to cust_number

x_train["cust_number"] = x_train.cust_number.str[:5]
x_validation["cust_number"] = x_validation.cust_number.str[:5]


# In[53]:


x_train.nunique()


# In[54]:


x_train.info()


# In[55]:


#dropping business year as it will have less impact in training our model because there are only two values
x_train.drop(columns = ["buisness_year"], inplace = True)
x_validation.drop(columns = ["buisness_year"], inplace = True)


# In[56]:


x_train.head()


# In[57]:


x_train.corr()  #.corr() is used to find the pairwise correlation of all columns in the dataframe


# In[58]:


#plotting heat map for our correlation within the data frame
import seaborn as sns
sns.heatmap(x_train.corr(), annot = True)


# In[59]:


x_train.dtypes


# In[60]:


#doing one hot encoding as it will assign distict values corresponding to the cust_number
from sklearn.preprocessing import LabelEncoder
cust_number_encoder = LabelEncoder()
cust_number_encoder.fit(x_train['cust_number'])
x_train['cust_number_enc'] = cust_number_encoder.transform(x_train['cust_number'])


# In[61]:


x_train[['cust_number_enc','cust_number']]


# In[62]:


from sklearn.preprocessing import LabelEncoder
cust_terms_encoder = LabelEncoder()
cust_terms_encoder.fit(x_train['cust_payment_terms'])
x_train['cust_terms_enc'] = cust_terms_encoder.transform(x_train['cust_payment_terms'])


# In[63]:


x_train[['cust_terms_enc','cust_payment_terms']]


# In[64]:


from sklearn.preprocessing import LabelEncoder
business_code_encoder = LabelEncoder()
business_code_encoder.fit(x_train['business_code'])
x_train['business_code_enc'] = business_code_encoder.transform(x_train['business_code'])


# In[65]:


x_train[['business_code_enc','business_code']]


# In[66]:


from sklearn.preprocessing import LabelEncoder
business_code_encoder = LabelEncoder()
business_code_encoder.fit(x_validation['business_code'])
x_validation['business_code_enc'] = business_code_encoder.transform(x_validation['business_code'])


# In[67]:


from sklearn.preprocessing import LabelEncoder
cust_number_encoder = LabelEncoder()
cust_number_encoder.fit(x_validation['cust_number'])
x_validation['cust_number_enc'] = cust_number_encoder.transform(x_validation['cust_number'])


# In[68]:


from sklearn.preprocessing import LabelEncoder
cust_terms_encoder = LabelEncoder()
cust_terms_encoder.fit(x_validation['cust_payment_terms'])
x_validation['cust_terms_enc'] = cust_terms_encoder.transform(x_validation['cust_payment_terms'])


# In[69]:


x_train.head()


# In[70]:


x_train.dtypes


# In[71]:


# splitting our target variable and saving it in y_train also dropping it from the x_train column
y_train = x_train["late_days"]
x_train = x_train.drop(columns = ["late_days"])

y_validation = x_validation["late_days"]
x_validation = x_validation.drop(columns = ["late_days"])


# In[72]:


#converting object amount in float
x_train['amount_in_usd']=pd.to_numeric(x_train['amount_in_usd'])
x_validation['amount_in_usd']=pd.to_numeric(x_validation['amount_in_usd'])

x_train['total_open_amount']=pd.to_numeric(x_train['total_open_amount'])
x_validation['total_open_amount']=pd.to_numeric(x_validation['total_open_amount'])


# In[73]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(x_train.merge(y_train , on = x_train.index ).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[74]:


x_train.dtypes


# In[75]:


# these are the key features I selected for training of my model
features = ['business_code_enc','cust_terms_enc', 'diff_duedate_psdate', 'cust_number_enc', 'due_weekday','amount_in_usd','post_weekday']


# In[76]:


x_train.nunique()


# In[77]:


x_validation.head()


# # MODEL PREPARATION

# In[78]:


from sklearn.linear_model import LinearRegression
base_model = LinearRegression()
base_model.fit(x_train[features], y_train)


# In[79]:


#Saving the predicted value in y_predicted to validate it with real y_validation value and check the error
y_predict = base_model.predict(x_validation[features])


# In[80]:


from sklearn.metrics import mean_squared_error
import math

print("Rmse validation: ",(mean_squared_error(y_validation, y_predict, squared=False)))
print("Mse validation: ",(mean_squared_error(y_validation, y_predict)))


# In[81]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math


# In[82]:


clf = RandomForestRegressor()
clf.fit(x_train[features], y_train)
predicted_validation = clf.predict(x_validation[features])

print("Rmse validation: ",math.sqrt(mean_squared_error(y_validation, predicted_validation)))
print("Mse validation: ",(mean_squared_error(y_validation, predicted_validation)))


# In[ ]:





# In[83]:


test.head()


# In[84]:


test.dtypes


# In[85]:


test['due_in_date'] = pd.to_datetime(test['due_in_date'], format='%Y%m%d') 
test['posting_date']=pd.to_datetime(test['posting_date'])


# In[ ]:





# In[86]:


test["diff_duedate_psdate"] = (test["due_in_date"]-test["posting_date"]).dt.days

test["cust_number"] = test.cust_number.str[:5]
test["post_weekday"] = test["posting_date"].dt.dayofweek


# In[87]:


test.head()


# In[88]:


#Doing one hot encoding for the selected features
from sklearn.preprocessing import LabelEncoder
business_code_encoder = LabelEncoder()
business_code_encoder.fit(test['business_code'])
test['business_code_enc'] = business_code_encoder.transform(test['business_code'])

from sklearn.preprocessing import LabelEncoder
cust_number_encoder = LabelEncoder()
cust_number_encoder.fit(test['cust_number'])
test['cust_number_enc'] = cust_number_encoder.transform(test['cust_number'])

from sklearn.preprocessing import LabelEncoder
cust_terms_encoder = LabelEncoder()
cust_terms_encoder.fit(test['cust_payment_terms'])
test['cust_terms_enc'] = cust_terms_encoder.transform(test['cust_payment_terms'])


# In[89]:


test["due_weekday"] = test["due_in_date"].dt.dayofweek         


# In[90]:


test.head()


# In[91]:


test["late_days"] = base_model.predict(test[features])


# In[92]:


# rounding off the predicted values because days only take integer values

test['late_days'] = test['late_days'].apply(np.ceil)


# In[93]:


#adding the predicted late_days to due_in date to get the predicted_payment_date
test["predicted_payment_date"] = test["due_in_date"] + test['late_days'].apply(lambda x: pd.Timedelta(x, unit='D'))


# In[94]:


test.head()


# # CREATING BUCKET

# In[95]:


def bucketisation(value):
    if value > 60:
        return ">60 days"
    elif value > 45:
        return "46-60 days"
    elif value > 30:
        return "31-45 days"
    elif value > 15:
        return "16-30 days"
    elif value >=0:
        return "0-15 days"
    elif value <0:
        return 'before due date'


# In[96]:


#applying the bucketisation function to our predicted late_days colunn
test["Aging_Bucket"] = test['late_days'].apply(bucketisation)


# In[97]:


test.head()


# In[ ]:





# In[ ]:




