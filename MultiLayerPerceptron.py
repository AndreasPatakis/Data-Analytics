import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold




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
    layer_1 = 15
    layer_2 = 12
    output_layer = len(features_y)



    """Intializing our model"""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_layer, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(layer_1, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(layer_2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(output_layer, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    """Applying 10-fold cross validation on our dataset"""
    kfold = KFold(n_splits=7,shuffle=False)
    cur_fold = 1
    scores = []
    for train,test in kfold.split(X,Y):

        print("\n","-"*10,"Training for fold: {}".format(cur_fold),"-"*10,'\n')
        """Validation_split=0.1 indicates that 10% of the training set will be used as validation set"""
        model.fit(X[train], Y[train], validation_split=0.1, epochs=5)

        """Testing our model using 30% of the initial dataset"""
        val_loss, val_accuracy = model.evaluate(X[test], Y[test])
        print("Validation loss: {}\nValidation Accuracy: {}\n".format(val_loss,val_accuracy))
        scores.append([val_loss,val_accuracy])
        cur_fold += 1

    best_fold = scores.index(max(scores))
    #print('\nThe best accuracy was found on fold {} with:\nValidation loss: {}\nValidation Accuracy: {}\n'.format(best_fold+1,scores[best_fold][0],scores[best_fold][1]))
