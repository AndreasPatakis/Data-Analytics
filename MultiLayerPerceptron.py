import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix,f1_score, jaccard_score


"""na kanei plot kai to test oxi mono to traiun, validation"""

def plot_history_loss(history,metrics,plot=True,save=False,cur_fold=0):
    parent_dir = './Classification_Plots'
    if not os.path.exists(parent_dir+'/NO_FOLDS'):
        os.makedirs(parent_dir+'/NO_FOLDS')

    """Plotting history loss for different metrics"""
    for metric in metrics:
        train_metric = metric
        if train_metric == 'accuracy':
            train_metric = 'loss'
        test_metric = 'val_'+train_metric
        fig, ax = plt.subplots()
        ax.plot(history.history[train_metric])
        ax.plot(history.history[test_metric])
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'validation'], loc='upper left')
        ax.set_title("Model Loss using "+train_metric+" metric")
        if save:
            if cur_fold:
                ax.set_title('Model Loss using fold: '+str(cur_fold)+' and metric: '+train_metric)
                cur_dir = parent_dir+'/FOLDS/'+train_metric+'_metric'
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)
                plt.savefig(cur_dir+'/'+train_metric+'_FOLD_'+str(cur_fold))
            else:
                plt.savefig(parent_dir+'/NO_FOLDS/'+train_metric+'_NO_FOLDS')
    if plot:
        plt.show()
    else:
        plt.close()

def evaluate_predictions(observed_train,predicted_train,observed_test,predicted_test,evaluate_dict):
    np.set_printoptions(threshold=np.inf)
    def get_indices(a_list):
        """Return the index where number 1 was found"""
        #return np.where(a_list==np.amax(a_list,axis=1))
        return a_list.argmax(1)

    observed_train = get_indices(observed_train)
    observed_test = get_indices(observed_test)
    predicted_train = get_indices(predicted_train)
    predicted_test = get_indices(predicted_test)
    """Evaluated the predictions of our model using multiple metrics"""
    print('\nConfusion Matrix of training set:\n',confusion_matrix(observed_train,predicted_train))
    print('\nConfusion Matrix of test set:\n',confusion_matrix(observed_test,predicted_test))
    print("\nTest set evaluation loss: ",int(evaluate_dict['loss']*100),"%")
    print("Test set evaluation accuracy: ",int(evaluate_dict['accuracy']*100),"%")
    print('\nJaccard Score of training set: ',jaccard_score(observed_train,predicted_train,average='micro'))
    print('Jaccard Score of test set: ',jaccard_score(observed_test,predicted_test,average='micro'))
    print('\nF1 Score of training set: ',f1_score(observed_train,predicted_train,average='weighted'))
    print('F1 Score of test set: ',f1_score(observed_test,predicted_test,average='weighted'))



if __name__ == '__main__':

    """Loading our dataset"""
    df = pd.read_csv('./Data/cleanedCTG_NN.csv')
    features = list(df.columns[:21])
    features_y = list(df.columns[21:])

    """Reformatting our DataFrame to numpy array"""
    X = df[features].to_numpy()
    Y = df[features_y].to_numpy()

    """Initializing some parameters for the model"""
    input_layer = len(features)
    layer_1 = 35
    layer_2 = 25
    output_layer = len(features_y)
    metrics = ['accuracy','mae','mse']


    """Intializing our model"""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_layer, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(layer_1, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(layer_2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(output_layer, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=metrics)


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
    history = model.fit(X_train, Y_train, validation_split=0.143, epochs=35, batch_size=32)
    evaluate_dict = model.evaluate(X_test, Y_test,return_dict=True)
    y_test_prediction = model.predict(X_test,verbose=0)
    y_train_prediction = model.predict(X_train,verbose=0)
    evaluate_predictions(Y_train,y_train_prediction,Y_test,y_test_prediction,evaluate_dict)
    plot_history_loss(history,metrics,plot=False,save=True)



















    # """Applying 8-fold cross validation on our dataset"""
    # kfold = KFold(n_splits=8,shuffle=False)
    # cur_fold = 1
    # scores = []
    # for train,test in kfold.split(X,Y):
    #     print("\n","-"*10,"Training for fold: {}".format(cur_fold),"-"*10,'\n')
    #
    #
    #     """Validation_split=0.1 indicates that 10% of the training set will be used as validation set"""
    #     history = model.fit(X[train], Y[train], validation_split=0.1, epochs=40)#batch_size = ??
    #
    #
    #     """Testing our model using 30% of the initial dataset"""
    #     print("-"*10,"Evaluating test set for fold ",cur_fold,"-"*10)
    #     evaluate_dict = model.evaluate(X[test], Y[test])
        # print("\nTest set evaluation loss: ",evaluate_dict['loss']*100,"%")
        # print("\nTest set evaluation accuracy: ",evaluate_dict['accuracy']*100,"%")
    #     #print("Validation loss: {}\nValidation Accuracy: {}\n".format(val_loss,val_accuracy))
    #     #scores.append(val_accuracy)
    #     plot_history_loss(history,metrics,save=True,plot=False,cur_fold=cur_fold)
    #     cur_fold += 1
    # best_fold = scores.index(max(scores))
    #
    # print('\nThe highest accuracy was found on fold {} with:\nValidation Accuracy: {}\n'.format(best_fold+1,scores[best_fold]))
