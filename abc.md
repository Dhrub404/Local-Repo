```python
# ============================
# PROGRAM 1
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("suv_data.csv")
dataset.head()
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
print (x)
print (y)
print(dataset[dataset.isnull().any(axis=1)])
bool_series=pd.isnull(dataset["Gender"])
dataset[bool_series]
bool_series=pd.notnull(dataset["Gender"])
dataset[bool_series]
dataset[10:25]
new_data=dataset.dropna(axis=0,how='any')
new_data
dataset.replace(to_replace=np.nan, value=-99)
dataset["Gender"].fillna("No Gender")
print("Old data frame length:", len(dataset))
print("New data frame length:", len(new_data))
print("Number of rows with at least 1 NA value:", len(dataset)-len(new_data))
new_df1 = dataset.ffill()
print(new_df1)
new_df3=dataset.dropna(how='all')
new_df3


# ============================
# PROGRAM 2
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Titanic-Dataset.csv')
print(data)
x = data.drop('Survived', axis = 1)
y = data['Survived']
print(x)
print(y)
x.drop(['Name', 'Ticket', 'Cabin'],axis = 1, inplace = True)
print(x)
x['Age'] = x['Age'].fillna(x['Age'].mean())
print(x)
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])
print(x)
x = pd.get_dummies(x, columns = ['Sex', 'Embarked'],prefix = ['Sex', 'Embarked'],drop_first = True)
print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train)
print(y_train)
from sklearn.preprocessing import StandardScaler
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)
print(x_train)


# ============================
# PROGRAM 3
# ============================
from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot

path = 'diabetes.csv'
dataframe = read_csv(path)
print(dataframe.head())

y = dataframe['Outcome']
x = dataframe.drop('Outcome', axis=1)
feature_names = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(x_train, y_train)

x_train_fs = fs.transform(x_train)
x_test_fs = fs.transform(x_test)

for i in range(len(fs.scores_)):
    print('Feature %d (%s): %f' % (i, feature_names[i], fs.scores_[i]))

pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks(range(len(fs.scores_)), feature_names, rotation=45)
pyplot.show()


# ============================
# PROGRAM 4
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('loandata.csv')
print(df.head())

if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

num_cols = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Education', 'Property_Area']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

print(df.head())

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

best_features = SelectKBest(score_func=chi2, k=min(10, len(X.columns)))
fit = best_features.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']

top_features = feature_scores.nlargest(10, 'Score')
print(top_features)

plt.figure(figsize=(12,6))
plt.bar(top_features['Feature'], top_features['Score'])
plt.xticks(rotation=45)
plt.show()


# ============================
# PROGRAM 5
# ============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')
print(iris)

x = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=1)

treemodel = DecisionTreeClassifier()
treemodel.fit(x_train,y_train)

y_pred = treemodel.predict(x_test)

plt.figure(figsize=(20,30))
tree.plot_tree(treemodel,filled=True)
plt.show()

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of Decision Tree Model:",accuracy)


# ============================
# PROGRAM 6
# ============================
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

dataset = pd.read_csv('User_Data.csv')
x= dataset.iloc[:,[2,3]].values
print(x)
y = dataset.iloc[:,4].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train , y_train)

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree Model:", accuracy)


# ============================
# PROGRAM 7
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

from sklearn.preprocessing import LabelEncoder
if 'Gender' in dataset.columns:
    le = LabelEncoder()
    dataset['Gender'] = le.fit_transform(dataset['Gender'])

X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

print(dataset.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

custom_prediction = classifier.predict(sc.transform([[1, 46, 28000]]))
print(custom_prediction)

y_pred = classifier.predict(X_test)

results = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),  axis=1)
print(results)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(cm)
print(accuracy)


# ============================
# PROGRAM 8
# ============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('salary_data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
dataset.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3, random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(y_test)
print(y_pred)

print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1))

from sklearn.metrics import mean_squared_error
mean=mean_squared_error(y_test,y_pred)
print(mean)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.show()


# ============================
# PROGRAM 9
# ============================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('50_Startups.csv')
df = pd.DataFrame(data)

print(df.head())

X = df.drop(columns=['Profit'])
y = df['Profit']

column_transformer = ColumnTransformer(
transformers=[('encoder', OneHotEncoder(), ['State'])], remainder='passthrough')
X = column_transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# ============================
# PROGRAM 10
# ============================
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
dataset.head()

import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_pred = hc.fit_predict(x)

mtp.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'blue')
mtp.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green')
mtp.scatter(x[y_pred== 2, 0], x[y_pred == 2, 1], s = 100, c = 'red')
mtp.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan')
mtp.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta')
mtp.show()


# ============================
# PROGRAM 11
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

x_train = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)

dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(x_scaled)

df['Cluster'] = y_dbscan

print(df['Cluster'].value_counts())

plt.scatter(x_train['Annual Income (k$)'], x_train['Spending Score (1-100)'], c=y_dbscan, cmap='rainbow')
plt.show()
```
