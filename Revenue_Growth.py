#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Business Case Study</h1>
# <h2 align="center">Strategy on Revenue Growth for a Portfolio of 5 Software Products</h2>
# <h3 align="center">Yuan Wang</h3>

# # Loading package

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading data

# <font size="4">**Data Dictionary**<br><br>
# <font size="2">
# >DATE:	           Date payment made<br>
# CLIENT_ID:          Client ID that uniquely identifies a client<br>
# INDUSTRY:	       Industry that client is categorized in<br>
# CLIENT_SIZE:        Size of client as number of employees reported<br>
# STATE:              US state client is located<br>
# PRODUCT: 	       Product ID<br>
# PRICE_PER_LICENSE:  Price per license per year paid by client in USD<br>
# NUM_LICENSE:        Number of licenses purchased<br>
# CSat:	           A customer satisfaction score in range of 1(lowest) to 10(highest) submitted by client<br>

# In[2]:


data = pd.read_csv('Revenue_Growth.csv')
data.head()


# In[3]:


data.shape


# # Exploratory Data Analysis

# ## Data Structure

# In[4]:


data.info()


# <font size="4">Before doing the fowlling analysis, we need to **convert categorical data type to numeric**.

# In[5]:


categorical = ["CLIENT_SIZE", "INDUSTRY", "STATE", "PRODUCT"]

for col in categorical:
    print(data[col].unique())


# In[6]:


from sklearn.preprocessing import LabelEncoder

for col in categorical:
    data['i.' + col] = LabelEncoder().fit_transform(data[col])

data.head()


# ## Missing data

# In[7]:


data.isnull().sum()


# <font size="4">The missing values only appear in satisfaction score. One way to handle these missing values would be simply removing all these lines from the data. However, given the size of dataset, removing more than 1000 data points would be a little wasting. Another fancy way of doing this would be **train a predictive model (kNN)**, so that it can predict the scores according to other features. Or, we can just fill the N/A values by **assigning the mean rating of the corresponding product**.

# ### 1) Assign the mean rating of the corresponding product

# In[8]:


for product in ['A','B','C','D','E']:
    
    # Compute the mean customer rating on each product
    mean = data.groupby('PRODUCT').CSat.mean()[product]
    print(mean)

    # For each missing rating of a certain product, assign its mean rating to be its value
    # data.loc[(data["CSat"].isnull()) & (data["PRODUCT"] == product), "CSat"] = mean


# <font size="4">The Customer satisfaction score (CSat) in the dataset is integer, but our mean value is not. Thus, the replacement by mean is inconsistent with original data.

# ### 2) Predict the rating by k-Nearest Neighbors model (KNN)

# In[11]:


index_null = data.loc[(data["CSat"].isnull())].index
data_null  = data.iloc[index_null]
data_full  = data.drop(data.index[index_null])

X = data_full[['CLIENT_ID','i.CLIENT_SIZE','i.INDUSTRY','i.STATE','i.PRODUCT','NUM_LICENSE','PRICE_PER_LICENSE']]
y = data_full['CSat']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier as KNN
error = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNN(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 


# <font size="4">Choose optimal K=19.

# In[12]:


classifier = KNN(n_neighbors=19)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[13]:


X_null = data_null[['CLIENT_ID','i.CLIENT_SIZE','i.INDUSTRY','i.STATE','i.PRODUCT','NUM_LICENSE','PRICE_PER_LICENSE']]
y_null = classifier.predict(X_null)

data.loc[(data["CSat"].isnull()), "CSat"] = y_null


# In[14]:


print('Check if the ratings are integers')
y_null


# In[15]:


data.isnull().sum()


# <font size="4">Now we have predicted the missing value by KNN model, retruning a full dataset without missing value. That is good start for the following data analysis.

# ## Product v.s. Industry

# In[16]:


df = data.sort_values(by=['PRODUCT', 'INDUSTRY'])

fig, ax = plt.subplots(figsize=(18,6))
plt.subplot(121)
sns.countplot(x='PRODUCT', hue='INDUSTRY', data=df)
plt.subplot(122)
sns.countplot(x='INDUSTRY', hue='PRODUCT', data=df)
plt.show()


# <font size="4">The sale pattern is similar for each product in each industry. Product A and D are more popular; IT and Finance industry is the major clients.

# ## Sale Distribution over Product Price

# In[17]:


plt.subplots(figsize=(12,6))

for product in ['A', 'B', 'C', 'D', 'E']:
    plt.hist(data[data["PRODUCT"] == product].PRICE_PER_LICENSE)

plt.xlabel('Price Per License', fontsize=20)
plt.ylabel('Total Number of Sale', fontsize=20)
plt.legend(["A", "B", "C", "D", "E"],title='PRODUCT',loc=2, prop={'size': 15})
plt.show()


# <font size="4">Product A is sold at the highest price, followed by product D, and product E is the cheapest one.<br>
# And most product is sold at it higher price.

# # Question_1: Which product has had the highest revenue growth rate?

# ## Target variable:  $REVENUE = UnitPrice* NumberSale$

# In[18]:


data["REVENUE"] = data["PRICE_PER_LICENSE"] * data["NUM_LICENSE"]


# ## set datetime (Y/Q/M)

# In[19]:


data['Year'] = pd.to_datetime(data.DATE).dt.to_period('Y')
data['Quarter'] = pd.to_datetime(data.DATE).dt.to_period('Q')
data['Month'] = pd.to_datetime(data.DATE).dt.to_period('M')
data.head()


# ## Plot the Revenue over Month, Quarter, Year

# In[20]:


REVENUE_Per_Month = pd.DataFrame(data.groupby(['PRODUCT', 'Month']).REVENUE.sum()).reset_index()
g = sns.catplot(x = 'Month', y = 'REVENUE', hue = 'PRODUCT', kind="point", data = REVENUE_Per_Month, height=6, aspect=12/6)
g.set_xticklabels(rotation = 70)

REVENUE_Per_Quarter = pd.DataFrame(data.groupby(['PRODUCT', 'Quarter']).REVENUE.sum()).reset_index()
sns.catplot(x = 'Quarter', y = 'REVENUE', hue = 'PRODUCT', kind="point", data = REVENUE_Per_Quarter, height=6, aspect=12/6)

REVENUE_Per_Year = pd.DataFrame(data.groupby(['PRODUCT', 'Year']).REVENUE.sum()).reset_index()
sns.catplot(x = 'Year', y = 'REVENUE', hue = 'PRODUCT', kind="point", data = REVENUE_Per_Year, height=6, aspect=12/6)

plt.show()


# <font size="4"><br>**Answer**:<br><br>
# 
# From the *Revenue Growth Charts*, we notice that<br><br>
# (1) Produat A grows dramatically over the year of 2015 and 2016, but the revenue declined in year of 2017. The revenue **jumped at the beginning of each year**.<br><br>
# (2) Product B and C had **gradually growth** over the year 2014 to 2017, the growth rates are close to constant.<br><br>
# (3) Produat D grows dramatically over the year of 2015 and 2016, but the revenue declined in year of 2017. Moreover, the revenue showed clearly **seasonal patter** in each year. It jumped to highest demand at the beginning of new year, then gradually decline until the end of each year.<br><br>
# (4) Product E entered market on July 2016. Its revenue **grows exponentially** after launched. We expect a much higher growth rate in the year of 2018 and future.

# ## Calculate Growth Rate (yearly )

# <font size="4">We calculate the **annual growth rate** by two methods:

# <font size="4"> 1) Take average over 4 years:

# In[21]:


Growth_Rate = []

for product in ['A','B','C','D']:
    REVENUE = REVENUE_Per_Year[REVENUE_Per_Year["PRODUCT"] == product]["REVENUE"].tolist()
    for year in [0,1,2]: 
        Growth = (REVENUE[year+1] - REVENUE[year]) * 100 / REVENUE[year]
        Growth_Rate.append(Growth)

    
REVENUE = REVENUE_Per_Year[REVENUE_Per_Year["PRODUCT"] == "E"]["REVENUE"].tolist()
Growth_Rate.append(np.nan)
Growth_Rate.append(np.nan)
Growth_Rate.append((REVENUE[1] - REVENUE[0]) *100 / REVENUE[0])

Growth_Rate_A = Growth_Rate[0:3]
Growth_Rate_B = Growth_Rate[3:6]
Growth_Rate_C = Growth_Rate[6:9]
Growth_Rate_D = Growth_Rate[9:12]
Growth_Rate_E = Growth_Rate[12:15]

d = {'A':Growth_Rate_A,'B':Growth_Rate_B,'C':Growth_Rate_C,'D':Growth_Rate_D,'E':Growth_Rate_E,}
GR1 = pd.DataFrame(d,columns=['A','B','C','D','E']).T
GR1.columns = ['2014-2015','2015-2016','2016-2017']
GR1['Average'] = GR1.mean(axis=1)


#  <font size="4"> 2) Fit a line:

# In[22]:


Growth_Rate = []

from sklearn.linear_model import LinearRegression

for product in ['A','B','C','D']:
    
    y = REVENUE_Per_Year[REVENUE_Per_Year['PRODUCT']==product].REVENUE
    y = np.log(list(y))
    X = np.array(range(0,len(y))).reshape(-1, 1)
    
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    coff = reg.coef_[0]
    Growth_Rate.append(coff*100)

Growth_Rate.append(np.nan)
d = {'Line_Slope':Growth_Rate}
GR2 = pd.DataFrame(d,index=['A','B','C','D','E'],columns=['Line_Slope'])


#  <font size="4"> Now Report the annually growth rate:

# In[23]:


Growth_Rate = pd.concat([GR1, GR2],axis=1)
print()
print()
print('\033[1m     Table. Annually Revenue Growth Rate (%)')
Growth_Rate


# <font size="4"><br>**Answer**:<br><br>
# 
# About the revenue growth rate:<br>
# >Product C had the highest revenue growth rate over the last 4 years.<br>
# >Product E lauched for only one year, but it grew fastest in last year.<br>

# # Question_2: Find major contributing factors to revenue growth of product E

# In[24]:


Product_E = data[(data['PRODUCT'] == 'E')]
mean_price_E = pd.DataFrame(Product_E.groupby('Month').PRICE_PER_LICENSE.mean()).reset_index()

fig, ax = plt.subplots(figsize=(18,6))
plt.subplot(121)
sns.countplot(x='PRODUCT', hue='Month', data=Product_E)
plt.subplot(122)
sns.barplot(x='Month', y='PRICE_PER_LICENSE', data=mean_price_E)

fig, ax = plt.subplots(figsize=(18,6))
plt.subplot(121)
sns.countplot(x='CLIENT_SIZE', hue='Month', data=Product_E)
plt.subplot(122)
sns.countplot(x='INDUSTRY', hue='Month', data=Product_E)

fig, ax = plt.subplots(figsize=(18,6))
sns.countplot(x='STATE', hue='Month', data=Product_E)

plt.show()


# <font size="4"><br>**Answer**:<br><br>
# 
# The dramatical increase in number of sale contributes to the fast revenue growth of product E, since the unit price remain constant over a year. We found same growth trend in all clients' size, industry sector, and location. The major contributing factor are remarkable increase of sale in clients with small-size in IT industry.

# # Question_3: How to grow overall revenue?

# ## Revenue Distribution

# In[25]:


# Pie chart
def Pie_Chart(group):
    sizes = data.groupby([group]).REVENUE.sum()
    labels = sizes.index.tolist()
    colors = ['yellow','lightcoral','lightgreen','pink','lightskyblue']
    plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal') 

fig, ax = plt.subplots(figsize=(27,8))
plt.subplot(131)
Pie_Chart('PRODUCT')
plt.subplot(132)
Pie_Chart('INDUSTRY')
plt.subplot(133)
Pie_Chart('CLIENT_SIZE')
plt.show()


# <font size="4">These three Pie charts above show that:<br>
# >Product A and D, IT and Finance industry, and clients with 100+ employees make larger contribution to total revenue.

# ## Correlation matrix

# In[26]:


# Compute the correlations between features except for the DATE and CLIENT_ID
correlation = data.drop(['DATE', 'CLIENT_ID'], axis=1).corr()

sns.set(style= 'white')

mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                 square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title("Correlation Heatmap", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 14)
plt.show()


# <font size="4">From this correlation matrix, we notice that:
# >(1) Revenue per order is positively related to license unit price and number of license sold.<br>
# >(2) A client with more employees needs more copies of software licenses, so as increase the revenue.<br>
# >(3) Clients would like to choose product with high rating, thus customer satisfaction score, client size and revenue are cross-correlated.<br>
# >(4) Price of products has significant difference.

# ## Multivariable Linear Regression

# In[27]:


# Create a dataframe of monthly overall revenue with count of other feathers

tmp0 = pd.DataFrame(data.groupby(['Month']).REVENUE.sum()).reset_index()
tmp1 = pd.DataFrame(pd.crosstab(data['Month'], data['CLIENT_SIZE'])).reset_index()
tmp2 = pd.DataFrame(pd.crosstab(data['Month'], data['INDUSTRY'])).reset_index()
tmp3 = pd.DataFrame(pd.crosstab(data['Month'], data['PRODUCT'])).reset_index()
tmp4 = pd.DataFrame(data.groupby(['Month']).CSat.mean()).reset_index()

tmp = [tmp0, tmp1, tmp2, tmp3, tmp4]

result = pd.concat(tmp, axis=1)
result = result.T.drop_duplicates().T

order = [0,1,2,4,3] + list(range(5,17))
col = result.columns

result = result[col[order]]
result.head()


# In[28]:


import statsmodels.api as sm

result = result[result.columns].apply(pd.to_numeric, errors='coerce', axis=1)
y = result['REVENUE']
X = result.drop(['Month', 'REVENUE'], axis=1)

LM = sm.OLS(y, X)
LM_results = LM.fit()

print (LM_results.summary())


# # Conclusions:

# <font size="4">From the regression results, we recommond that:<br>
# >(1) Introduce the products to large-size clients (employee>100).<br>
# >(2) Develop more client in Finance and Retail industry.<br>
# >(3) Boost the sale amount of product A and D, those are the major source of total revenue.<br>

# ## [The End]

# ###                                                      Yuan Wang,                Nov-15-2018
