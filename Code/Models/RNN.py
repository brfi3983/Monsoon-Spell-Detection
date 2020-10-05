# ===================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE #, KMeansSMOTE

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM, Flatten
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score

# ===================================================================================================================

#initialize some knowledge via variables...................................................................
# set classification flag as 1 when you deal as classification problem and apply softmax at the last layer,
# and set the flag as zero when it is regression problem, predicting regression rainfall values
classificationflag = 1

# trying to predict with "present lead" number of days ahead
present_lead = 3

#history considered for climatic variables at lead
num_hist_cd = 3

#history considered for rainfall data at lead
num_hist_rf = 5

# test and validation ratio
test_ratio = 0.15

# path in computer and clusters
path_comp_moumita = "/media/moumita/Research/Files/University_Colorado/Work/work4/Spells_data_results/"
path_comp_brandon = ""
path_cluster = "/projects/mosa2108/spells/"

path = path_cluster

# ===================================================================================================================
# function: add history of days before lead as features
def addHistoryClimateVariables(climate_data, num_hist_cd):

    climate_data_new = climate_data
    for i in range(1, num_hist_cd):
        climate_data_hist = np.roll(climate_data, i, axis=0)
        climate_data_new = np.concatenate((climate_data_new, climate_data_hist), axis=1)
    print(climate_data_new.shape)

    return climate_data_new

# ===================================================================================================================
# function: add history of rainfall before lead as features
def addHistoryRainfall(climate_data_new, rainfall_regress, present_lead, num_hist_rf):

    for i in range(num_hist_rf):
        rainfall_hist = np.roll(rainfall_regress, (i + present_lead), axis=0)
        climate_data_new = np.concatenate((climate_data_new, rainfall_hist), axis=1)
    print(climate_data_new.shape)

    return climate_data_new

# ===================================================================================================================
# function: change the class label to one hot notation
def changeToOneHot(rainfall_class):

    rainfall_onehot = []
    for i in range(rainfall_class.shape[0]):
        if rainfall_class[i]==1:
            rainfall_onehot.append([1, 0, 0])
        elif rainfall_class[i]==2:
            rainfall_onehot.append([0, 1, 0])
        elif rainfall_class[i]==3:
            rainfall_onehot.append([0, 0, 1])
        else:
            continue

    rainfall_onehot = np.asarray(rainfall_onehot)
    # print(rainfall_class.shape, rainfall_onehot.shape)

    return rainfall_onehot

# ===================================================================================================================
# function: assign the predictand as rainfall values for regression problem, or one hot class for classification one
def assignPredictorAndPredictand(climate_data_new, classificationflag, rainfall_class, rainfall_regress):

    x = climate_data_new
    if classificationflag == 1:
        y = changeToOneHot(rainfall_class)
    elif classificationflag == 0:
        y = rainfall_regress

    return [x, y]

# ===================================================================================================================
# function: divide the set into training, validation, and test set
def divideIntoSets(x,y, test_ratio):

    x_train1, x_test, y_train1, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=test_ratio, shuffle=False)

    return [x_train,x_valid,x_test,y_train,y_valid,y_test]

# ===================================================================================================================
# function: add a new axis to the numpy shape to incline it towards the model input
def addNewAxis(x_train,x_valid,x_test):

    x_train_f = x_train[:, newaxis, :]
    x_valid_f = x_valid[:, newaxis, :]
    x_test_f = x_test[:, newaxis, :]

    return [x_train_f,x_valid_f,x_test_f]

# ===================================================================================================================
# function: train the LSTM model for classification problem (one hot rainfall classes)
def trainLSTMModelForClassification(x_train_f, y_train, x_valid_f, y_valid, bs, lr):

    model = Sequential()
    # model.add(CuDNNLSTM(64, input_shape=(X_train_f.shape[1], X_train_f.shape[2]), return_sequences=True))
    # model.add(Dropout(0.2))

    # I changed this to CPU version for my machine
    print(x_train_f.shape)
    model.add(LSTM(64, input_shape=(x_train_f.shape[1], x_train_f.shape[2])))  # return a single vector of dimension 64

    ## the variable return_sequence== true for feeding this layer to another layer of LSTM
    # model.add(LSTM(64, return_sequences=True, input_shape=(X_train_f.shape[1], X_train_f.shape[2]))) # returns a sequence of vectors of dimension 64
    # model.add(LSTM(64))  # return a single vector of dimension 64
    # model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation='softmax'))

    opt = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train_f, y_train, epochs=300, batch_size=bs, validation_data=(x_valid_f, y_valid), verbose=2, shuffle=False)  # , class_weight= 'auto')

    return [model, history]

# ===================================================================================================================
# function:  changing the class probability given by the softmax layer of the model to numeric class label
def predictToNumeric(pred):

    pred_numeric = []

    for i in range(pred.shape[0]):
        if pred[i,0] >= pred[i,1] and pred[i,0] >= pred[i,2]:
            pred_numeric.append(1)
        elif pred[i,1] >= pred[i,0] and pred[i,1] >= pred[i,2]:
            pred_numeric.append(2)
        elif pred[i,2] >= pred[i,0] and pred[i,2] >= pred[i,1]:
            pred_numeric.append(3)
        else:
            continue

    pred_numeric = np.asarray(pred_numeric)

    print(pred.shape, pred_numeric.shape)

    return pred_numeric

# ===================================================================================================================
# function: changing the one hot class notation to numeric class label
def oneHotToNumeric(y_onehot):

    y_num = []
    for i in range(y_onehot.shape[0]):
        if y_onehot[i,0]==1 and y_onehot[i,1]==0 and y_onehot[i,2]==0:
            y_num.append(1)
        elif y_onehot[i,0]==0 and y_onehot[i,1]==1 and y_onehot[i,2]==0:
            y_num.append(2)
        elif y_onehot[i,0]==0 and y_onehot[i,1]==0 and y_onehot[i,2]==1:
            y_num.append(3)
        else:
            continue

    y_num =np.asarray(y_num).transpose()
    y_num = np.reshape(y_num, (y_num.shape[0],1))
    print(y_onehot.shape, y_num.shape)

    return y_num

# ===================================================================================================================
# function: plot the model losses for training and validation period
def plotModelAccuracy(history,filename):

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Var')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='upper right')
    plt.savefig(path+'results/RNN/'+filename+'.png')
    plt.close()

    # plt.title('Model Accuracy')
    # plt.plot(history.history['acc'], label='training')
    # plt.plot(history.history['val_acc'], label='validation')
    # plt.ylabel('Acc')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.show()

# ===================================================================================================================
# function: providing different classification performance measures
def modelPerformanceClassificationMetrics(y_test_nor, y_pred_nor,filename):

    file = open(path+'results/RNN/'+filename+".txt",'w')
    file.write("\nClassification report:\n")
    file.write(classification_report(y_test_nor, y_pred_nor))
    file.write("\n\nConfusion matrix: True class vs Predicted Class\n")
    file.write(np.array2string(confusion_matrix(y_test_nor, y_pred_nor), separator=', '))
    file.write("\n\nBalanced accuracy score:")
    file.write(str(balanced_accuracy_score(y_test_nor,y_pred_nor)))
    file.write("\n\nAccuracy:")
    file.write(str(accuracy_score(y_test_nor,y_pred_nor)))

def countClasses(y):

    c1=0
    c2=0
    c3=0
    for item in y:
        if item[0]==1:
            c1= c1+1
        elif item[1]==1:
            c2=c2+1
        elif item[2]==1:
            c3= c3+1

    print(c1, c2, c3)

# ===================================================================================================================
# function: the main function..........................
def main():

    # read the data variable
    climate_data = np.asarray(pd.read_csv(path+"DataSets/dataset_with_lead_having_five_variables_AIR_SLP_PWTR_UWND_VWND/Lead_"+str(present_lead)+".csv"))

    # noting the number of samples and features (including the last column as class)
    [num_samples, num_features] = climate_data.shape
    print(num_samples,num_features)

    #the last column is class so separate that class column from the rest data features
    rainfall_class = climate_data[:,num_features-1]
    climate_data = climate_data[:,:num_features-1]
    print(climate_data.shape)

    # adding history to the climate features in terms of history of 5 variables we are using at lead
    climate_data_new = climate_data
    climate_data_new = addHistoryClimateVariables(climate_data_new, num_hist_cd)

    # input the raw rainfall values......
    rainfall_regress = np.asarray(pd.read_csv(path+"DataSets/dataset_rainfall/rainfall_data_SC.csv"))
    print(rainfall_regress.shape)

    # adding the past rainfall regress values as features (as we are trying to predict at lead 3, we will consider history rainfall before that)
    climate_data_new = addHistoryRainfall(climate_data_new, rainfall_regress, present_lead, num_hist_rf)

    # assigning the X and Y for further initiating the model
    [x, y] = assignPredictorAndPredictand(climate_data_new, classificationflag, rainfall_class, rainfall_regress)

    # didiving into training, validation, and test sets (with 70% train 15% valid anmd 15% test)
    [x_train, x_valid, x_test, y_train, y_valid, y_test] = divideIntoSets(x,y, test_ratio)

    # oversampling to make them equal numbers
    x_train_new, y_train_new = SMOTE().fit_resample(x_train, y_train)
    x_valid_new, y_valid_new = SMOTE().fit_resample(x_valid, y_valid)
    x_test_new, y_test_new = SMOTE().fit_resample(x_test, y_test)

    countClasses(y_train_new)
    countClasses(y_valid_new)
    countClasses(y_test_new)


    # add new axis to the data for making it compatible to the Sequential model
    [x_train_new_f, x_valid_new_f, x_test_new_f] = addNewAxis(x_train_new,x_valid_new,x_test_new)

    print(x_test_new_f.shape, y_test_new.shape)
    print(x_valid_new_f.shape, y_valid_new.shape)
    print(x_train_new_f.shape, y_train_new.shape)

    #  running the models for a number of gridSearch variables
    # applying grid search
    batch_size = [20, 30, 40, 50, 70, 100, 150, 200]
    # batch_size = [25]
    learn_rate = [0.00001]
    # learn_rate = [0.00001]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    for bs in batch_size:
        for lr in learn_rate:

            # train the LSTM sequential model
            [model, history] = trainLSTMModelForClassification(x_train_new_f, y_train_new, x_valid_new_f, y_valid_new, bs, lr)

            # plot the train and validation loses
            plotModelAccuracy(history,'loss_batch-size_'+str(bs)+'_learn-rate_'+str(lr))

            # Test the model over the explicit test set
            y_pred = model.predict(x_test_new_f)
            y_pred_nor = predictToNumeric(y_pred)
            y_test_nor = oneHotToNumeric(y_test_new)

            # Performance in terms of classification metrices
            modelPerformanceClassificationMetrics(y_test_nor, y_pred_nor,'result_batch-size_'+str(bs)+'_learn-rate_'+str(lr))


# ===================================================================================================================

if __name__ == '__main__':
    main()

# ===================================================================================================================