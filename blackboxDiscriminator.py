#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, pickle, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import load_ffri2018_blackboxD


# classify_algo :   Algorithm name
def build_model(classify_algo):
    if classify_algo == 'RF':
        model = RandomForestClassifier(n_estimators=1000)
    elif classify_algo == 'GB':
        model = GradientBoostingClassifier()
    elif classify_algo == 'AB':
        model = AdaBoostClassifier()
    elif classify_algo == 'LR':
        model = LogisticRegression(solver='lbfgs')
    elif classify_algo == 'DT':
        model = DecisionTreeClassifier()
    elif classify_algo == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(64, ))
    elif classify_algo == 'SVM':
        model = SVC()
    elif classify_algo == 'BNB':
        model = BernoulliNB()
    elif classify_algo == 'KNN':
        model = KNeighborsClassifier(n_jobs=10)
    elif classify_algo == 'VOTE':
        clf1 = RandomForestClassifier(n_estimators=1000)
        clf2 = GradientBoostingClassifier()
        clf3 = AdaBoostClassifier()
        clf4 = LogisticRegression(solver='lbfgs')
        clf5 = DecisionTreeClassifier()
        clf6 = MLPClassifier(hidden_layer_sizes=(64, ))
        clf7 = SVC()
        model = VotingClassifier(
            estimators=[
                ('RF',clf1),('GB',clf2),('AB',clf3),('LR',clf4),('DT',clf5),('MLP',clf6),('SVM',clf7)
            ], voting='soft')
    else:
         raise NotImplementedError('Error! No such classify algorithm')
    return model


# folder :   '<Path>'
# classify_algo :   Algorithm name
# api_list_area :   '000*txt', To limit dimensions.
def train(folder, classify_algo, api_list_area):
    # Load data to make api_list
    folder_list = [folder+'/cleanware', folder+'/malware']
    api_list = load_ffri2018_blackboxD.make_api_list(folder_list, api_list_area)
    
    # Save api_list 
    with open('blackboxD_api_list_'+api_list_area, 'w') as f:
        for api in api_list:
            f.write(api+'\n')
    
    # Load train data
    train_area_list = ['0*.txt', '1*.txt']
    used_api = load_ffri2018_blackboxD.make_used_api_dataframe(folder_list, train_area_list, api_list)
    X_train = used_api.drop('label', axis=1)
    y_train = used_api['label']
    
    # Load test data
    test_area_list = ['2*.txt', '3*.txt']
    used_api = load_ffri2018_blackboxD.make_used_api_dataframe(folder_list, test_area_list, api_list)
    X_test = used_api.drop('label', axis=1)
    y_test = used_api['label']
    
    # Train
    model = build_model(classify_algo)
    model.fit(X_train, y_train)
    
    # Evaluate
    print(model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred, [1, 0]))    
    pickle.dump(model, open('blackboxD_model_'+classify_algo+'_'+api_list_area.rstrip('.txt')+'.pkl', 'wb'))
    
    
# folder :   '<Path>'
# classify_algo :   Algorithm name
# api_list_area :   '000*txt', To limit dimensions.
def predict(folder, classify_algo, api_list_area): 
    # Load api_list
    with open('blackboxD_api_list_'+api_list_area) as f:
        api_list = [api.rstrip('\n') for api in f]
    
    # Load test data
    data_area = '*.txt'
    used_api = load_ffri2018_blackboxD.make_used_api_dataframe([folder], [data_area], api_list)
    X_test = used_api.drop('label', axis=1)
    
    # Predict
    model = pickle.load(open('blackboxD_model_'+classify_algo+'_'+api_list_area.rstrip('.txt')+'.pkl', 'rb'))
    y_pred = model.predict(X_test)
    
    # Print
    file_list = [str(file) for file in Path(folder).glob(data_area)]
    file_list = [(int(re.search(r"[0-9]+", file).group()), file) for file in file_list]
    file_list.sort()
    file_list = [x[1] for x in file_list]
    
    for i, result in enumerate(y_pred):
        if result == 0:
            print(file_list[i]+' is cleanware')
        elif result == 1:
            print(file_list[i]+' is malware')
        else:
            print(result)
      
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="select mode: 'train' or 'predict'")
    parser.add_argument("--algo", type=str, help="classify algorithm")
    parser.add_argument("--folder", type=str, help="folder_path")
    args = parser.parse_args()
    return args
               
    
if __name__ == '__main__':
    args = get_args()
    api_list_area = '000*.txt'
    if args.mode == 'train':      
        train(args.folder, args.algo, api_list_area)
    elif args.mode == 'predict':
        predict(args.folder, args.algo, api_list_area)
    else:
        raise NotImplementedError('Error! No such mode')

        