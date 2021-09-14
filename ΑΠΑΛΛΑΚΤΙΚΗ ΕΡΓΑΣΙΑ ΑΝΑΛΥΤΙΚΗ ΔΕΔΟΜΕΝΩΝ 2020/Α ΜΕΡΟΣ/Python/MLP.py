import csv

import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, r2_score, matthews_corrcoef, confusion_matrix,balanced_accuracy_score,jaccard_score
tf.disable_v2_behavior()


def get_data():
    with open('pre_proccess_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data.pop(0)
    print(data)
    return data

def split_data_X_Y(data):
    X = []
    Y = []
    for i in range(0, len(data)):
        X.append([x for _, x in zip(range(4), data[i])])
        Y.append(data[i][-10:])
    return X, Y

def round_labels(labels):
    labels = np.array(labels)
    round_label = []
    for i in range(0,len(labels)):
        round_label.append(np.argmax(labels[i], axis=0))
    return round_label

# summarize history for accuracy
def accuracy_history_init(his):
    plt.plot(his.history['accuracy'])
    plt.plot(his.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# summarize history for loss
def loss_history_init(his):
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def baseline_model(inp,h1,h2,out):
    # define the keras model
    model = Sequential()
    model.add(Dense(h1, input_dim=inp, activation='relu'))
    model.add(Dense(h2, activation='relu'))
    model.add(Dense(out, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

data = get_data()
print(data)
data_X, data_Y = split_data_X_Y(data)


# Split data into training , validation and testing set with percentage 60% , 10% and 30% each.
records_size = len(data_X)
train_size = int(records_size * .6)
valid_size = int(records_size * .1)
print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (records_size - (train_size + valid_size)))

# Split the data
data_X = np.array(data_X)
data_Y = np.array(data_Y)
X_train = data_X[:train_size,:]
Y_train = data_Y[:train_size]
X_validation = data_X[train_size:train_size + valid_size,:]
Y_validation = data_Y[train_size:train_size + valid_size]
X_test = data_X[train_size + valid_size:,:]
Y_test = data_Y[train_size + valid_size:]
print(Y_train)

# Network Parameters
number_of_hidden_layer_1 = 10 # 1st layer number of features
number_of_hidden_layer_2 = 10 # 2nd layer number of features
number_of_inputs = X_train.shape[1]  #it will be 4
number_of_outputs = data_Y.shape[1]  #10 total classes CYT,NUC,MIT,ME3,ME2,ME1,EXC,VAC,POX,ERL

model = baseline_model(number_of_inputs,number_of_hidden_layer_1,number_of_hidden_layer_2,number_of_outputs)
print(model.summary())

# fit the keras model on the dataset
history = model.fit(X_train, Y_train,validation_data=(X_validation, Y_validation),epochs=100, batch_size=25)

# list all data in history
print(history.history.keys())

#history plots
accuracy_history_init(history)
loss_history_init(history)

# Evaluate and Predict
scores = model.evaluate(X_train, Y_train, verbose=0)

ytr_pred = model.predict_classes(X_train, verbose=0)
print("Train Accuracy by model.predict: %.2f%%" % (100*sum(round_labels(Y_train) ==ytr_pred)/Y_train.shape[0]))
# make class predictions with the model
yva_pred = model.predict_classes(X_validation, verbose=0)
print("Val Accuracy by model.predict: %.2f%%" % (100*sum(round_labels(Y_validation) ==yva_pred)/Y_validation.shape[0]))
# make class predictions with the model
yte_pred = model.predict_classes(X_test,verbose=0)
print("Test Accuracy by model.predict: %.2f%%" % (100*sum(round_labels(Y_test) == yte_pred)/Y_test.shape[0]))



print(confusion_matrix(round_labels(Y_train), ytr_pred))
print(confusion_matrix(round_labels(Y_validation), yva_pred))
print(confusion_matrix(round_labels(Y_test), yte_pred))
print("Train MMC: ", matthews_corrcoef(round_labels(Y_train), ytr_pred))
print("Val MMC: ", matthews_corrcoef(round_labels(Y_validation), yva_pred))
print("Test MMC: ", matthews_corrcoef(round_labels(Y_test), yte_pred))
print("Train JACCARD: ", jaccard_score(round_labels(Y_train), ytr_pred,average='micro'))
print("Val JACCARD: ", jaccard_score(round_labels(Y_validation), yva_pred,average='micro'))
print("Test JACCARD: ", jaccard_score(round_labels(Y_test), yte_pred,average='micro'))
print("Train F1: ", f1_score(round_labels(Y_train), ytr_pred,average='weighted'))
print("Val F1: ", f1_score(round_labels(Y_validation), yva_pred,average='weighted'))
print("Test F1: ", f1_score(round_labels(Y_test), yte_pred,average='weighted'))
print("Train BALANCED ACCURACY: ", balanced_accuracy_score(round_labels(Y_train), ytr_pred))
print("Val BALANCED ACCURACY: ", balanced_accuracy_score(round_labels(Y_validation), yva_pred))
print("Test BALANCED ACCURACY: ", balanced_accuracy_score(round_labels(Y_test), yte_pred))


