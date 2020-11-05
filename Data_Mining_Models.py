# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Credit Dataset.csv')
# X = dataset.iloc[:, :-1].values
X = dataset.iloc[:, [4, 5, 9, 10, 17]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Logistic Regression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Support Vector Classifier
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Calculation of Confusion Matrices
lr_cm = confusion_matrix(y_test, lr_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
svc_cm = confusion_matrix(y_test, svc_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
