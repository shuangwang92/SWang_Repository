#### Predicting walk disabilities in the next 6 month from disabilities patients ###
### Using a subset of the original dataset: 5k data
## 2D plot

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('uWalk6M_Cases.csv')

dataset = dataset.sample(frac = 0.005, random_state = 42)

# separate independent variable X and indep var y
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Split dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

############################## Applying the PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
# explained_variance = pca.explained_variance_ratio_ Principal 2 components: 0.56

############################## Applying the kernel PCA
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

######## Model1: Fitting the logistic regression ############################## Several Seconds
###############################################################################
from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression(random_state = 42)
classifier_log.fit(X_train, y_train)

# Predict the test set
y_pred_log = classifier_log.predict(X_test)
y_prob_log = classifier_log.predict_proba(X_test)[:, 1]

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm_log = confusion_matrix(y_test, y_pred_log)
(192+623)/1048 # Accuracy_score = 0.778;

F1_score_log = f1_score(y_test, y_pred_log) #All indep: F1 score = 0.84;

######## Model2: Fitting the Naive Bayes ######################################
###############################################################################
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

# Predict the test set
y_pred_NB = classifier_NB.predict(X_test)
y_prob_NB = classifier_NB.predict_proba(X_test)[:, 1]

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm_NB = confusion_matrix(y_test, y_pred_NB)
(592+217)/1048 # include all indep vars: Accuracy_score = 0.77;

F1_score_NB = f1_score(y_test, y_pred_NB) #All indep: F1 score = 0.832; 

# Visualising test results with Logistic Regression (1k)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_log.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('uWalk Predicting within the next 6 months by Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising test results with Naive Bayes (1k)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_NB.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('uWalk Predicting within the next 6 months by Naive Bayes (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
