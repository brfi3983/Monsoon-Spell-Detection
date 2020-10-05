#===================================================================================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, f_regression

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
#====================================================================================================================
'''
PIPELINE TUNING:---
Hyper-parameters are parameters that are manually tuned by a human operator to maximize the model performance
against a validation set through a grid search.GridSearchCV module helps in this.
'''

# n_components_to_test = [2, 4, 8, 16, 32, 64, 100, 128, 200, 256, 300]   #used in PCA

n_components_to_test = [2, 4, 8, 16, 24, 32, 48, 64]    #used in PCA

alpha_to_test = [2.0 ** np.arange(-15, +3)] #used in Ridge regression where the alpha needs to be optimized

# gamma_to_test = [2.0 ** np.arange(-20, -4)] #for Lasso

gamma_to_test = np.logspace(-9, 3, 13)  # for SVM

C_to_test = np.logspace(-2, 10, 13)  # for SVM

kernel_to_test = ['linear','rbf','poly'] # for SVM

# alpha_to_test = (0.1, 1.0, 10.0) #used in Ridge regression and Lasso where the alpha needs to be optimized

# kernel_to_test = ['linear','rbf'] # used in SVR

cv_to_test = [5]

n_estimators_to_test = [5, 10, 50]  # random forest trees

#====================================================================================================================

# function with evaluation measures to evaluate the regression problem
def calculateEvaluationMetrics(y_true, y_pred):

    # Explained variance regression score function
    # print("\n\nExplained variance score: %.2f" %explained_variance_score(y_true, y_pred))
    # Mean absolute error regression loss
    mae = mean_absolute_error(y_true, y_pred)
    # print("\n\nMean absolute error: %.2f" %mae)
    # Mean squared error regression loss
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    # print("\n\nRoot Mean squared error: %.2f" %rmse)
    # Mean squared logarithmic error regression loss
    # print("\n\nMean squared log error: %.2f" %mean_squared_log_error(y_true, y_pred))
    # # Median absolute error regression loss
    # print("\n\nMedian absolute error: %.2f" %median_absolute_error(y_true, y_pred))
    # R^2 (coefficient of determination) regression score function
    rscore = r2_score(y_true, y_pred)
    # print("\n\nR2-score: %.2f" %rscore)
    #calculate pearson correlation
    # r, p = pearsonr(y_true, y_pred)
    cor = np.corrcoef(y_true, y_pred)
    # print("\n\nPearson correlation prev and present : %.3f %.3f " %(p, cor))

    stddev = np.std(y_true)
    # print("\n\nstandard-deviation", stddev)
    nrmse = (1-(sqrt(mean_squared_error(y_true, y_pred))/stddev))
    # print("\n\nNormalized Root Mean squared error: %.2f\n\n" % nrmse)


    return mae, rmse, rscore, cor[0][1], nrmse

#====================================================================================================================

#Function for writing the regression results to the csv file and text file

def writeActualAndPredictedPipeModel1(filename, y_true, y_pred, y_truelpa, y_predlpa):

    # writing actual vs predicted for each case
    result_data = {'actual': y_true, 'predicted': y_pred, 'actuallpa': y_truelpa, 'predlpa': y_predlpa}
    df = pd.DataFrame(result_data, columns=['actual', 'predicted', 'actuallpa', 'predlpa'])
    df.to_csv(filename)


def writeErrorMeasuresPipeModel(filename, errors):

    errors = np.array(errors)
    comp = errors[:,0]
    mae = errors[:,1]
    rmse = errors[:,2]
    rscore = errors[:,3]
    r = errors[:,4]
    nrmse = errors[:,5]

    # writing the error measures for different number of features
    result_data = {'n_components': comp, 'mae': mae, 'rmse': rmse, 'r2-score': rscore, 'correlation': r, 'normalized-rmse': nrmse}
    df = pd.DataFrame(result_data, columns=['n_components', 'mae','rmse','r2-score','correlation','normalized-rmse'])
    df.to_csv(filename)



#====================================================================================================================

# the pipelined model for optimizing the hyperparameters of feature reduction and regressor model within the pipeline

def pipe_pca_SVM(X_train, y_train, X_test, y_test, filename1, filename2, lpa):

    errors=[]

    for comp in n_components_to_test:
        pipe = Pipeline([('reduce_dim', PCA(n_components=comp)),('clf', SVC())])
        params = {'svm_cv': cv_to_test,'svm_C': C_to_test, 'svm_gamma': gamma_to_test, 'svm_kernel': kernel_to_test}

        #optimization is invoked as follows....
        gridsearch = GridSearchCV(pipe, params, verbose=1, cv=5).fit(X_train, y_train)

        #get the prediction by the learned model
        y_pred = gridsearch.predict(X_test)
        y_true = y_test

        y_predlpa = []
        y_truelpa = []
        for ind in range(len(y_pred)):
            y_predlpa.append(y_pred[ind] * (100.0/lpa))
            y_truelpa.append(y_true[ind] * (100.0/lpa))

        print("\n\nPCA followed by SVM Results for PCA n-comp = ",comp)
        mae, rmse, rscore, r, nrmse = calculateEvaluationMetrics(y_truelpa, y_predlpa)
        errors.append([comp, mae, rmse, rscore, r, nrmse])
        writeActualAndPredictedPipeModel1(filename1 + '_comp' + str(comp) + '.csv', y_true, y_pred, y_truelpa,
                                          y_predlpa)


    writeErrorMeasuresPipeModel(filename2, errors)

#====================================================================================================================
