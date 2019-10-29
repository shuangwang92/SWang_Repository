#### Predicting walk disabilities in the next 6 month from disabilities patients ###
### Using the original dataset: 1 Million

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('uWalk6M_Cases.csv')

dataset.info()
dataset.isnull().sum()
dataset.isnull().any()
dataset['uWalk'].value_counts()
dataset['Unable_Walk6M'].value_counts()
dataset.dtypes
dataset.head()

#subset = dataset.sample(frac = 0.001, random_state = 42)

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
a = np.unique(X_train)

############################## Applying the PCA
from sklearn.decomposition import PCA
pca = KernelPCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_

######## Model1: Fitting the SVR to the training set ##########################
###############################################################################
#from sklearn.svm import SVC
#classifier_SVM = SVC(kernel = 'linear', probability = True, random_state = 42)
#classifier_SVM.fit(X_train, y_train)

# Applying Grid Search to find the best model and best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]

#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_


# Predict the test set
#y_pred_SVM = classifier_SVM.predict(X_test)
#y_prob_SVM = classifier_SVM.predict_proba(X_test)[:, 1]

# Making the confusion matrix
#from sklearn.metrics import confusion_matrix, f1_score
#cm_SVM = confusion_matrix(y_test, y_pred_SVM)
#(6120 + 12171)/20943 # include all indep vars:Accuracy_score = 0.8734; 

# PCA D-reduct: 

#F1_score_SVM = f1_score(y_test, y_pred_SVM) #All indep: F1 score = 0.9018; After PCA: 0.877

# Applying K-Fold Cross Validation to evaluate the Logistic Regression's performance
from sklearn.model_selection import cross_val_score
accuracies_log = cross_val_score(estimator = classifier_log,
                                 X = X_train,
                                 y = y_train,
                                 cv = 10)

accuracies_log.mean() #0.8730
accuracies_log.std() #0.00089

# Applying K-Fold Cross Validation to evaluate the Naive Bayes's performance
from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator = classifier_NB,
                                X = X_train,
                                y = y_train,
                                cv = 10)

accuracies_NB.mean() #0.8711
accuracies_NB.std() #0.0007

# Applying K-Fold Cross Validation to evaluate the Random Forest's performance
from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = classifier_RF,
                                X = X_train,
                                y = y_train,
                                cv = 10)

accuracies_RF.mean() #0.8711
accuracies_RF.std() #0.0007

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
(63970+119076)/209426 # Accuracy_score = 0.8740;

F1_score_log = f1_score(y_test, y_pred_log) #All indep: F1 score = 0.9003;

######## Model2: Fitting the Naive Bayes ###################################### Type 2 Error High
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
(62720+119872)/209426 # include all indep vars: Accuracy_score = 0.8719;

F1_score_NB = f1_score(y_test, y_pred_NB) #All indep: F1 score = 0.8993; 


######## Model3: Fitting to the Random Forest ################################# 6 mins
###############################################################################
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 42)
classifier_RF.fit(X_train, y_train)

# Predict the test set
y_pred_RF = classifier_RF.predict(X_test)
y_prob_RF = classifier_RF.predict_proba(X_test)[:, 1]

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm_RF = confusion_matrix(y_test, y_pred_RF)
(53727+126824)/209426 # include all indep vars: Accuracy_score = 0.8621; 


F1_score_RF = f1_score(y_test, y_pred_RF) #All indep: F1 score = 0.8978; LDA, 1 indep: 0.8959

# Visualising training results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_NB.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('uWalk Predicting in the next 6 months by Kernel SVM (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# Visualising test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_NB.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('uWalk Predicting in the next 6 months by Kernel SVM (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

####### Plotting the ROC curve and culculating AUC #########
from sklearn.metrics import roc_curve, roc_auc_score
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_prob_log)
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, y_prob_NB)
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_prob_RF)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_log, tpr_log, label = 'Logistic Regression')
plt.plot(fpr_NB, tpr_NB, label = 'Naive Bayes')
plt.plot(fpr_RF, tpr_RF, label = 'Random Forest')
plt.xlabel('1-Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC curve')
plt.legend()
plt.show()

auc_NB = roc_auc_score(y_test, y_prob_NB) # ROC_AUC_SVM: 0.9288
auc_log = roc_auc_score(y_test, y_prob_log) # ROC_AUC_Log: 0.9304
auc_RF = roc_auc_score(y_test, y_prob_RF)   # ROC_AUC_RF: 0.9304



