import pandas as pd
import numpy as np
import pickle


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

###############################Reading data from datasets to train the models###########
tictac_final = open('tictac_final.txt', 'r')
tictac_single = open('tictac_single.txt', 'r')
tictac_multi = open('tictac_multi.txt', 'r')

tictac_final_df = pd.DataFrame(data=np.loadtxt(tictac_final),columns=['x0','x1','x2','x3','x4','x5','x6','x7','x8','y'])

tictac_single_df = pd.DataFrame(data=np.loadtxt(tictac_single),columns=['x0','x1','x2','x3','x4','x5','x6','x7','x8','y_optimal'])

tictac_multi_df = pd.DataFrame(data=np.loadtxt(tictac_multi),columns=['x0','x1','x2','x3','x4','x5','x6','x7','x8','y0','y1','y2','y3','y4','y5','y6','y7','y8'])
########################################################################################

action = "Test"
#set action = "Train" for training
#set action = "Test"  for testing

class Classifiers():
    def __init__(self):
        self.mlp_dataset1(action)
        self.mlp_dataset2(action)

        self.svm_dataset1(action)
        self.svm_dataset2(action)
        
        self.knn_dataset1(action)
        self.knn_dataset2(action)
        

    def knn_dataset1(self,action):
        X_train, X_test, y_train, y_test = train_test_split(
            tictac_final_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_final_df['y'],
            test_size=0.2, random_state=4)

        if action == "Train":
            #Getting Optimal K value
            k_range = range(1, 25)
            scores_list = list()

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
                scores_list.append(np.average(scores))

            # Chose the optimal value of K to be 5. Finding the elbow. Using the same for the model.
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            print("Saving the KNN Trained Model ( k = 5) daatset 1")
            #Saving model using pickle
            with open("Trained_KNN_D1.pkl", 'wb') as file:
                pickle.dump(knn, file)

        elif action == "Test":
            #Load trained model using pickle
            with open("Trained_KNN_D1.pkl", 'rb') as file:
                knn = pickle.load(file) 

            y_pred = knn.predict(X_test)
            print("Accuracy score(KNN) for dataset 1 on test set for k = 5",metrics.accuracy_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return

    def knn_dataset2(self,action):
        
        X_train, X_test, y_train, y_test = train_test_split(
            tictac_single_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_single_df['y_optimal'],
            test_size=0.2, random_state=4)

        if action == "Train":
            #Finding best K value
            k_range = range(1, 25)
            scores_list = list()

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
                scores_list.append(np.average(scores))

            # Chose the optimal value of K to be 1. Using the same for the model.
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            print("Saving the KNN Trained Model ( k = 1) dataset2")
            #Saving model using pickle
            with open("Trained_KNN_D2.pkl", 'wb') as file:
                pickle.dump(knn, file)

        elif action == "Test":
            #Load trained model using pickle
            with open("Trained_KNN_D2.pkl", 'rb') as file:
                knn = pickle.load(file)   

            y_pred = knn.predict(X_test)
            print("Accuracy score(KNN) for dataset 2 on test set for k = 1: ", metrics.accuracy_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return


    def mlp_dataset1(self,action):
    
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_final_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_final_df['y'],
            test_size=0.2, random_state=4)

        if action == "Train":
            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(90, 20), random_state=1)
            # need to get k fold
            kf = KFold(n_splits=10)
            scores_list_kfold = list()
            max_score = 0
            clf_out = clf
            #KFOLD Cross Validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.accuracy_score(y_test, y_pred)
                scores_list_kfold.append(score)
                if (score > max_score):
                    max_score = score
                    clf_out = clf
            print("Model Validation(MLP) accuracy on dataset 1 : ", np.average(scores_list_kfold))

            #Saving model using pickle
            with open("Trained_MLP_D1.pkl", 'wb') as file:
                pickle.dump(clf_out, file)

        elif action == "Test":  
            #Load trained model using pickle
            with open("Trained_MLP_D1.pkl", 'rb') as file:
                clf_out = pickle.load(file)  

            # Using the weight that proved to be most successful for predicting on the test set.
            y_actual_pred = clf_out.predict(X_test_pre)
            score = metrics.accuracy_score(y_test_pre, y_actual_pred)
            print("Model Prediction(MLP) accuracy on dataset 1: ", score)
            cm = confusion_matrix(y_test_pre, y_actual_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return

    def mlp_dataset2(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_single_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_single_df['y_optimal'],
            test_size=0.2, random_state=4)

        if action == "Train":
            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(200, 100, 40),
                                random_state=1, max_iter=500)
            # need to get k fold
            kf = KFold(n_splits=10)
            scores_list_kfold = list()
            max_score = 0
            clf_out = clf
            #KFOLD Cross Validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.accuracy_score(y_test, y_pred)
                scores_list_kfold.append(score)
                if (score > max_score):
                    max_score = score
                    clf_out = clf
            print("Model Validation(MLP) accuracy on dataset 2 : ", np.average(scores_list_kfold))

            #Saving model using pickle
            with open("Trained_MLP_D2.pkl", 'wb') as file:
                pickle.dump(clf_out, file)
            
        elif action == "Test":
            #Load trained model using pickle
            with open("Trained_MLP_D2.pkl", 'rb') as file:
                clf_out = pickle.load(file)  

            y_actual_pred = clf_out.predict(X_test_pre)
            score = metrics.accuracy_score(y_test_pre, y_actual_pred)
            print("Model Prediction(MLP) accuracy on dataset 2:  ", score)
            cm = confusion_matrix(y_test_pre, y_actual_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return

    def svm_dataset1(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_final_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_final_df['y'],
            test_size=0.2, random_state=4)
        
        if action == "Train":
            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            clf = svm.SVC(kernel='linear')
            kf = KFold(n_splits=10)
            scores_list_kfold = list()
            max_score = 0
            clf_out = clf
            #KFOLD Cross Validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.accuracy_score(y_test, y_pred)
                scores_list_kfold.append(score)
                if (score > max_score):
                    max_score = score
                    clf_out = clf
            print("Model Validation(SVM) accuracy for dataset 1: ", np.average(scores_list_kfold))

            #Saving model using pickle
            with open("Trained_SVM_D1.pkl", 'wb') as file:
                pickle.dump(clf_out, file)

        elif action == "Test":
            #Load trained model using pickle
            with open("Trained_SVM_D1.pkl", 'rb') as file:
                clf_out = pickle.load(file) 

            # Using the weight that proved to be most successful for predicting on the test set.
            y_actual_pred = clf_out.predict(X_test_pre)
            score = metrics.accuracy_score(y_test_pre, y_actual_pred)
            print("Model Prediction(SVM) accuracy for dataset 1:", score)
            cm = confusion_matrix(y_test_pre, y_actual_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return    

    def svm_dataset2(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_single_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']], tictac_single_df['y_optimal'],
            test_size=0.2, random_state=4)

        if action == "Train":
            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
            kf = KFold(n_splits=5)
            scores_list_kfold = list()
            max_score = 0
            clf_out = clf
            #KFOLD Cross Validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.accuracy_score(y_test, y_pred)
                scores_list_kfold.append(score)
                if (score > max_score):
                    max_score = score
                    clf_out = clf
            print("Model Validation(SVM) accuracy for dataset 2: ", np.average(scores_list_kfold))

            #Saving model using pickle
            with open("Trained_SVM_D2.pkl", 'wb') as file:
                pickle.dump(clf_out, file)

        elif action == "Test":
            #Load trained model using pickle
            with open("Trained_SVM_D2.pkl", 'rb') as file:
                clf_out = pickle.load(file) 

            # Using the weight that proved to be most successful for predicting on the test set.
            y_actual_pred = clf_out.predict(X_test_pre)
            score = metrics.accuracy_score(y_test_pre, y_actual_pred)
            print("Model Prediction(SVM) accuracy for dataset 2: ", score)
            cm = confusion_matrix(y_test_pre, y_actual_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            print("\n")
        return


#########REGRESSORS############    

class Regressors():
    def __init__(self):
        self.mlp(action)
        self.knr(action)
        self.liner_regression(action)

    # metrics.accuracy_score of sklearn works only on vectors, custom defined function for calculating on matrices
    def calculate_accuracy_matrix(self, y_test, y_pred):
        accuracy = 0
        for i in range(0, 9):
            accuracy = accuracy + metrics.accuracy_score(y_test[:, i], y_pred[:, i])
        return accuracy / 9

    def liner_regression(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_multi_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']],
            tictac_multi_df[['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']], test_size=0.2, random_state=4)

        cols = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']
        threshold = 0.5
        X = X_train_pre.to_numpy()
        y = y_train_pre.to_numpy()
        X = np.append(X, np.ones([len(X), 1]), 1)

        X_test_pre = np.append(X_test_pre, np.ones([len(X_test_pre), 1]), 1)

        if action == "Train":
            kf = KFold(n_splits=10)
            max_score = 0
            max_weight = list()
            
            #KFOLD Cross Validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                y_pred = np.empty((y_test.shape[0], y_test.shape[1]))
                weight_list = list()
                for col in range(len(cols)):
                    pseudo_inverse = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)
                    weight = np.dot(pseudo_inverse, y_train[:, col])
                    weight_list.append(weight)
                    y_pred_col = np.dot(X_test, weight)
                    y_pred[:, col] = y_pred_col

                y_pred_threshold = np.where(y_pred < threshold, 0, 1)
                score = self.calculate_accuracy_matrix(y_test, y_pred_threshold)
                if (score > max_score):
                    max_score = score
                    max_weight = weight_list
            print("Saving the weight for the best model for Linear Regression")
            #Save weights with pickle
            with open("Trained_Linearreg_D3.txt", 'wb') as file:
                pickle.dump(max_weight, file)

        elif action == "Test":
            #Load trained weights using pickle
            with open("Trained_Linearreg_D3.txt", 'rb') as file:
                max_weight = pickle.load(file) 
                    
            # Using the weight that proved to be most successful for predicting on the test set.
            y_pred_test = np.empty((y_test_pre.shape[0], y_test_pre.shape[1]))

            for col in range(len(cols)):
                y_pred_test_col = np.dot(X_test_pre, max_weight[col])
                y_pred_test[:, col] = y_pred_test_col

            y_pred_test_threshold = np.zeros_like(y_pred_test)
            y_pred_test_threshold[np.arange(len(y_pred_test)), y_pred_test.argmax(1)] = 1
            score = self.calculate_accuracy_matrix(y_test_pre.to_numpy(), y_pred_test_threshold)

            print("Accuracy Score For Linear Regression: ", score)
            print("\n")
        return max_weight

    def knr(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_multi_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']],
            tictac_multi_df[['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']], test_size=0.2, random_state=4)

        threshold = .5
        if action == "Train":
            k_range = range(1, 25)
            scores = {}
            scores_list = list()

            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            # KFOLD Cross Validation
            for k in k_range:
                knr = KNeighborsRegressor(n_neighbors=k)
                # need to get k fold
                kf = KFold(n_splits=10)
                scores_list_kfold = []

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    knr.fit(X_train, y_train)
                    y_pred = knr.predict(X_test)
                    y_pred_threshold = np.where(y_pred < threshold, 0, 1)
                    score = self.calculate_accuracy_matrix(y_test, y_pred_threshold)
                    scores_list_kfold.append(score)

                avg_score = np.average(scores_list_kfold)
                scores_list.append(avg_score)

            #Choosing K as 1
            knr = KNeighborsRegressor(n_neighbors=1)
            knr.fit(X_train_pre, y_train_pre)


            print("Saving the KNR Trained Model")

            #Saving model using pickle
            with open("Trained_KNR_D3.pkl", 'wb') as file:
                pickle.dump(knr, file)

        
        if action == "Test":
            #Load trained model using pickle
            with open("Trained_KNR_D3.pkl", 'rb') as file:
                knr = pickle.load(file)

            y_pred = knr.predict(X_test_pre)
            y_pred_threshold = np.where(y_pred < threshold, 0, 1)
            score = self.calculate_accuracy_matrix(y_test_pre.to_numpy(), y_pred_threshold)
            print("Accuracy Score For KNR, setting value of K = 1 : ", score)
            print("\n")
        return knr

    def mlp(self,action):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            tictac_multi_df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']],
            tictac_multi_df[['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']], test_size=0.2, random_state=4)
        threshold = 0.4
        if action == "Train":
            X = X_train_pre.to_numpy()
            y = y_train_pre.to_numpy()

            clf = MLPRegressor(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(200, 100, 40),
                            random_state=1, max_iter=500)

            # KFOLD Cross Validation
            kf = KFold(n_splits=10)
            scores_list_kfold = list()
            max_score = 0
            clf_out = clf

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_threshold = np.where(y_pred < threshold, 0, 1)
                score = self.calculate_accuracy_matrix(y_test, y_pred_threshold)
                scores_list_kfold.append(score)
                if (score > max_score):
                    max_score = score
                    clf_out = clf
            print("MLP Model Cross Validation accuracy: ", np.average(scores_list_kfold))
            
            #Saving model using pickle
            with open("Trained_MLP_D3.pkl", 'wb') as file:
                pickle.dump(clf_out, file)


        if action == "Test":
            #Load trained model using pickle
            with open("Trained_MLP_D3.pkl", 'rb') as file:
                clf_out = pickle.load(file)

            y_actual_pred = clf_out.predict(X_test_pre)
            y_pred_threshold = np.where(y_actual_pred < threshold, 0, 1)
            score = self.calculate_accuracy_matrix(y_test_pre.to_numpy(), y_pred_threshold)
            print("MLP Model Prediction accuracy: ", score)
            print("\n")
        return clf_out

if __name__ == "__main__":
    """Extracting the board from the command line input.
       if board provided as input, pass this board.
       else pass empty board for new game
    """
    while(True):
        i = int(input("Enter 1 for Classifiers:\nEnter 2 for Regressors:\nEnter any other key to Stop Program:\nEnter: "))
        if(i == 1):
            # Classification
            classifiers = Classifiers()
        elif(i == 2):
            # Regression
            regressors = Regressors()
        else:
            break