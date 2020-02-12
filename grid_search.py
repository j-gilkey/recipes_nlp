
import pandas as pd
import squirrel_data_access
import resampling
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
import itertools
#import pydotplus


def grid_search(model, param_grid, X_train, X_test, y_train, y_test, scoring = 'accuracy'):
    #naive to the input model but is dependent on global X, Y variables being declared in the script
    grid_model = GridSearchCV(model, param_grid, scoring=scoring, cv=None, n_jobs=1)
    #initialize grid model
    grid_model.fit(X_train, y_train)
    #fit it

    best_parameters = grid_model.best_params_
    print('Grid Search found the following optimal parameters: ')
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))
    #report optimum parameters

    training_preds = grid_model.predict(X_train)
    test_preds = grid_model.predict(X_test)
    #create predictions

    cm = confusion_matrix(y_test, test_preds)
    print(cm)
    #show confusion matrix

    training_classified = classification_report(y_train, training_preds)
    test_classified = classification_report(y_test, test_preds)
    #assess them

    print(training_classified)
    print(test_classified)
    #print results
    return grid_model
