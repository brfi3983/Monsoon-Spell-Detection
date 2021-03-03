import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN

#====================================================================================================================
'''
PIPELINE TUNING:---
Hyper-parameters are parameters that are manually tuned by a human operator to maximize the model performance
against a validation set through a grid search.GridSearchCV module helps in this.
'''

n_components_to_test = [2, 4, 8, 16, 24, 32, 48, 64]    # number of reduced features

#SVC classifier's parameters..

C_to_test = [0.1, 1, 10, 50, 100]

kernel_to_test = ['linear','rbf']#,'sigmoid']

gamma_to_test = ['scale'] #1e-1, 1e-2, 1e-3, 1e-4, 'auto']

n_estimators_to_test = [3, 5, 10, 30, 50, 80, 100, 200]



#====================================================================================================================

# function with evaluation measures to evaluate the regression problem
def calculateEvaluationMetrics(y_true, y_pred):

    # Balanced accuracy score
    bas = balanced_accuracy_score(y_true, y_pred)*100

    # Confusion matrix
    conf = confusion_matrix(y_true, y_pred)

    # Accuracy score
    acc = accuracy_score(y_true, y_pred)*100

    # Number of correctly said classes
    noc = accuracy_score(y_true, y_pred, normalize= False)

    # classification report
    cr = classification_report(y_true, y_pred)

    #heat map from confusion matrix
    conf_norm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(conf_norm)

    print("\nBalanced accuracy score (takes care of imbalanced class) is: %.2f" % bas)
    print("Overall Accuracy of classification: %.2f" % acc)
    print("The classifier correctly classify: ", noc," out of total", len(y_true), " samples.")
    print("The overall classification report is as following:\n", cr)
    print("The confusion matrix is following.\n",conf)

    sns.heatmap(df, annot=True, linewidths=0.8, cmap="YlGnBu")
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()

    return

#===================================================================================================================
# classifier with grid search over the parameters: SVM

def classifierSVC(X_train, Y_train, X_test):
    #-----------------------------------------------------------------
    # creating the pipeline: presently have only one classifier
    pipe = Pipeline([('classifier', SVC(probability = False, max_iter=-1, class_weight = 'balanced'))])
    params = {'classifier__C': C_to_test, 'classifier__kernel': kernel_to_test, 'classifier__gamma': gamma_to_test,}

    # optimization is invoked as follows.... (resampled data with SMOTE)
    gridsearch = GridSearchCV(pipe, params, verbose=1, cv=5).fit(X_train, Y_train)
    # get the prediction by the learned model
    Y_pred = gridsearch.predict(X_test)

    return Y_pred

#===================================================================================================================
# classifier with grid search over the parameters: randomForest

def classifierRandomForest(X_train, Y_train, X_test):
    #-----------------------------------------------------------------
    # creating the pipeline: presently have only one classifier
    pipe = Pipeline([('classifier', RandomForestClassifier(class_weight = 'balanced'))])
    params = {'classifier__n_estimators': n_estimators_to_test}

    # optimization is invoked as follows.... (resampled data with SMOTE)
    gridsearch = GridSearchCV(pipe, params, verbose=1, cv=5).fit(X_train, Y_train)
    # get the prediction by the learned model
    Y_pred = gridsearch.predict(X_test)

    return Y_pred

#===================================================================================================================


def main():

    climate_data = pd.read_csv(".././DataSets/Lead_10_Hist.csv")
    climate_data = np.asarray(climate_data)
    end_col = climate_data.shape[1]

    #---------------------------------------------
    #segregating the predictand and predictors
    X = climate_data[:,:end_col-1]
    Y = climate_data[:,end_col-1]

    #----------------------------------------------
    # checking the number of samples for each class
    print("\nSamples of each rainfall class in overall set: ", collections.Counter(Y))

    #------------------------------------------------------------------
    # dividing into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, shuffle=False)
    print("\nSamples in training set: ", collections.Counter(Y_train))

    # ---------------------------------------------------
    # Upsampling the data for increasing the balance between class
    #resampling should be done over the training set and test set should be put away from it

    # #method 1: SMOTE
    X_resampled1, Y_resampled1 = SMOTE().fit_resample(X_train, Y_train)
    print("\nSMOTE:", sorted(collections.Counter(Y_resampled1).items()))

    #method 2: ADASYN
    X_resampled2, Y_resampled2 = ADASYN().fit_resample(X_train, Y_train)
    print("\nADASYN:", sorted(collections.Counter(Y_resampled2).items()))

    #-----------------------------------------------------------------
    # Calling the classifier module
    Y_pred = classifierSVC(X_resampled2, Y_resampled2, X_test)
    Y_true = Y_test
    # evaluating the classification
    print("\nUp-sampled data using ADASYN..................")
    calculateEvaluationMetrics(Y_true, Y_pred)


if __name__ == '__main__':
    main()