#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Analyze the input varibles before training model.
Draw the histogram and save as png for every attribute.
'''

import numpy as np
import pandas as pd
import numbers
import math

import numbers
import math
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
import random
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisquare

def numvar_analysis(data, indattr, targetattr, filepath, trunflag):
    '''
    :param data: the complete dataset, both numerical and categorical
    :param indattr: the independent numerical attribute
    :param targetattr: churn flag, class 0/1
    :param filepath: the path to save the analyzed pictures
    :param trunflag: whether need to do some truncation for outliers
    :return none
    '''
    # extract target attr and independent attr
    # filter nan attr, nan==nan is false
    validAttr = data.loc[data[indattr] == data[indattr]][[indattr, targetattr]]
    # calc the percentage of valid records
    validRow = validAttr.shape[0] * 1.0 / data.shape[0]
    validRowFmt = "%.2f%%" % (validRow * 100)

    # show the distribution by target attrs
    x0 = validAttr.loc[validAttr[targetattr] == 0][indattr]
    x1 = validAttr.loc[validAttr[targetattr] == 1][indattr]

    # truncate rows
    if trunflag == True:
        pcnt95 = np.percentile(validAttr[indattr], 95)
        x0 = x0.map(lambda x0: min(x0, pcnt95))
        x1 = x1.map(lambda x1: min(x1, pcnt95))

    xw0 = 100.0 * np.ones_like(x0) / x0.size
    xw1 = 100.0 * np.ones_like(x1) / x1.size

    x = pd.concat([x0, x1])
    # print x.shape
    descStats = x.describe()
    mu = "%.2e" % descStats['mean']
    std = "%.2e" % descStats['std']
    maxVal = "%.2e" % descStats['max']
    minVal = "%.2e" % descStats['min']
    # print validRowFmt, mu

    # plot histgram
    fig, ax = pyplot.subplots()
    ax.hist(x0, weights=xw0, alpha=0.5, label='Retained')
    ax.hist(x1, weights=xw1, alpha=0.5, label='Attrition')
    titleTxt = 'Histogram of ' + indattr + '\n' + 'valid pcnt =' + validRowFmt \
            + ', Mean =' + mu + ', Std=' + std + '\n max=' + maxVal \
            + ', min=' + minVal
    ax.set(title=titleTxt, ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    pyplot.legend(loc="upper right")
    pyplot.subplots_adjust(top=0.8) # adjust top to let title look well

    figSavePath = filepath + str(indattr) + '.png'
    # pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

    # anova test
    anova_results = anova_lm(ols(indattr + '~' + targetattr, validAttr).fit())
    # print 'anova test for ' + indattr + '~' + targetattr + ':'
    # print anova_results


def catvar_analysis(data, indattr, targetattr, filepath, trunflag):
    '''
    :param data: the complete dataset, both numerical and categorical
    :param indattr: the independent numerical attribute
    :param targetattr: churn flag, class 0/1
    :param filepath: the path to save the analyzed pictures
    :param trunflag: whether need to do some truncation for outliers
    :return none
    '''
    # extract target attr and independent attr
    # filter nan attr, nan==nan is false
    validAttr = data.loc[data[indattr] == data[indattr]][[indattr, targetattr]]
    # calc the percentage of valid records
    recdNum = validAttr.shape[0]
    validRow = validAttr.shape[0] * 1.0 / data.shape[0]
    validRowFmt = "%.2f%%" % (validRow * 100)

    freqDict = {}
    churnRateDict = {}
    # calculate the percentage and churn rate for categorical attr
    for v in set(validAttr[indattr]):
        vAttr = validAttr.loc[validAttr[indattr] == v]
        freqDict[v] = vAttr.shape[0] * 1.0 / recdNum
        churnRateDict[v] = (sum(vAttr[targetattr]) * 1.0 / vAttr.shape[0])
    descStats = pd.DataFrame({'percent': freqDict
                                 , 'churn rate': churnRateDict})
    # print descStats

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()    # Create another axes that shares the same x-axis as ax.
    pyplot.title('The percentage and churn rate for '+ indattr
                 + '\n valid pcnt =' + validRowFmt )
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2
                           , width=0.2, position = 1)
    ax.set_ylabel('churn rate', color='r')
    ax2.set_ylabel('percentage', color='b')
    figSavePath = filepath + str(indattr) + '.png'
    # pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

    # chi-squre test
    chisqData = allData[[indattr, targetattr]]
    grouped = chisqData[targetattr].groupby(chisqData[indattr])
    count = list(grouped.count())
    churn = list(grouped.sum())
    chisqTab = pd.DataFrame({'total': count, 'churn': churn})
    churnRatio = 0.101  # general churn ratio
    chisqTab['expected'] = chisqTab['total'].map(lambda x: round(x * churnRatio))
    chisqValList = chisqTab[['churn', 'expected']].apply\
                                (lambda x: (x[0] - x[1]) ** 2 / x[1], axis=1)
    chisqVal = sum(chisqValList)
    print 'chi-squre test for ' + indattr + '~' + targetattr + ':', chisqVal
    # chisqVal2 = chisquare(chisqTab['churn'], chisqTab['expected'])
    # print chisqVal2


if __name__ == "__main__":
    # read the internal and external data
    path = 'D:/bak_20170421/data/finance/'
    interData = pd.read_csv(path + 'bankChurn.csv')
    extData = pd.read_csv(path + 'ExternalData.csv')
    allData = pd.merge(interData, extData, on='CUST_ID')
    # print allData.shape

    # mark the type for every attribute, num or string
    allAttrs = set(list(allData))
    allAttrs.remove('CHURN_CUST_IND')   # CHURN_CUST_IND is the flag of churning
    # print allAttrs

    # extract the numerical and catigorical attrs respectively
    numAttrs = []
    catAttrs = []
    for attr in allAttrs:
        x = allData[attr]
        x = [i for i in x if i==i]  # eliminate the noise: 'nan' and ''
        # print x
        if isinstance(x[0], (int, float, numbers.Real, numbers.Integral)):
            numAttrs.append(attr)
        elif isinstance(x[0], str):
            catAttrs.append(attr)
        else:
            print 'the type of ', attr, ' cannot be determined.'
    # print catAttrs
    # print numAttrs

    # analyze the distribution for every numerical attribute by churn/un-churn
    churnflag = 'CHURN_CUST_IND'
    # filepath = path + 'num_picts/trunc/'
    # for attr in numAttrs:
    #     numvar_analysis(allData, attr, churnflag, filepath, True)

    filepath = path + 'cat_picts/'
    for attr in catAttrs:
        catvar_analysis(allData, attr, churnflag, filepath, False)


    # exit()