#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Predict customer churn using GBDT, with a training parameters procedure.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation, ensemble, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


if __name__ == "__main__":
    path = 'D:/bak_20170421/data/finance/'
    modelData = pd.read_csv(path + 'modelData.csv', header=0)
    allFeatures = list(modelData.columns)
    allFeatures.remove('CUST_ID')
    allFeatures.remove('CHURN_CUST_IND')

    x_train, x_test, y_train, y_test = train_test_split(modelData[allFeatures]
                                                        , modelData['CHURN_CUST_IND']
                                                        , test_size=0.3, random_state=9)
    # print y_train.value_counts()

    gbm0 = GradientBoostingClassifier(random_state=9)
    # gbm0 = LogisticRegression()
    # gbm0.fit(x_train, y_train)
    # y_pred = gbm0.predict(x_test)
    # y_predprod = gbm0.predict_proba(x_test)[:, 1]
    # print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred)
    # print "AUC Score (Testing Dataset): %f" % metrics.roc_auc_score(y_test, y_predprod)
    #
    # y_pred2 = gbm0.predict(x_train)
    # y_predprod2 = gbm0.predict_proba(x_train)[:, 1]
    # print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred2)
    # print "AUC Score (Training Dataset): %f" % metrics.roc_auc_score(y_train, y_predprod2)

    # tunning the number of estimators
    param_test1 = {'n_estimators': range(20,81,10)}
    gs1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1
                                , min_samples_split=300, min_samples_leaf=20
                                , max_depth=8, max_features='sqrt', subsample=0.8
                                , random_state=10)
                       , param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    # gs1.fit(x_train, y_train)
    # print 'grid_scores_: ', gs1.grid_scores_
    # print 'best_estimator_: ', gs1.best_estimator_
    # print 'best_score_: ', gs1.best_score_


    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70
                                     , min_samples_split=300, min_samples_leaf=20
                                     , max_depth=8, max_features='sqrt', subsample=0.8, random_state=10)
    clf.fit(x_train, y_train)
    importances = clf.feature_importances_
    # print importances
    features_sorted = np.argsort(-importances)
    # print features_sorted
    import_features = [allFeatures[i] for i in features_sorted]
    import_features_value = [importances[i] for i in features_sorted]
    # print import_features

    # draw by bar
    var = pd.DataFrame({'FE': import_features, 'VAL': import_features_value})
    pcnt90 = np.percentile(import_features_value, 90)
    var1 = var.loc[(var['VAL']>pcnt90)]
    var2 = var1.set_index(['FE'])
    print var2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Attributes')
    ax1.set_ylabel('importance')
    ax1.set_title('the top 10% most important features')
    var2.plot(kind='bar', )
    plt.show()
