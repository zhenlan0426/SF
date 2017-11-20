#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:21:29 2017

@author: will
"""

import numpy as np
import pandas as pd

def RNN_generator(data,batchSize,seqSize,df,key,discreteList,shuffle=True,downSample=1):
    # return bool (if init should be reset) 
    # and a list [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)
    # finetune by use store-item pair in testset only
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    n = data.shape[0]

    for from_ in range(0,n-batchSize,batchSize):
        X = list(data.loc[from_:from_+batchSize-1,'store_nbr':'cluster'].values.astype(np.int32).T)
        weight = np.ones((batchSize,1),dtype=np.float32)
        weight[data.loc[from_:from_+batchSize-1,'perishable']==1] = 1.25
        for j,(y,w,Xdiscrete,Xcontinue) in enumerate(timeGenerator(\
                           data.loc[from_:from_+batchSize-1,'date':'amax'],seqSize,downSample,df,key,discreteList)):
            yield j==0, [y,weight*w,Xcontinue] + Xdiscrete + X
            
def timeGenerator(data,seqSize,downSample,df,key,discreteList):
    # df is time related variables like holiday and seasonality
    # returns y, weight, X_discrete, X_continue
    n = data.shape[0]
    data['curr'] = data.amin 
    sparse = pd.concat([data.apply(lambda x: pd.Series(x.date,name='date'),axis=1)\
                                    .stack().reset_index().drop(['level_1'],1),\
                        data.apply(lambda x: pd.Series(x.unit_sales),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1),\
                        data.apply(lambda x: pd.Series(x.onpromotion),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1)],1)
    sparse.columns = ['level','date','sales','onpromotion']
    while np.all(data.curr + pd.DateOffset(seqSize) <= data['amax']):
        dense = data.apply(lambda x:pd.Series(pd.date_range(x.curr,periods=seqSize+1)),axis=1)\
                            .stack().reset_index().drop(['level_1'],1)
        dense.columns = ['level','date']
        dense = pd.merge(pd.merge(dense, df[np.random.randint(10)], how='left', on=key),
                        sparse,how='left',on=['level','date']).fillna(0)
        dense_continue = dense[['dcoilwtico','sales']].values.astype(np.float32)\
                                  .reshape((n,seqSize+1,2))[:,:seqSize,:]
        dense_discrete = list(np.moveaxis(dense[discreteList].values.astype(np.int32)\
                                                .reshape((n,seqSize+1,len(discreteList)))[:,1:,:]\
                                          ,2,0))
        y = dense['sales'].values.astype(np.float32).reshape((n,seqSize+1))[:,1:]
        weight = np.ones_like(y,dtype=np.float32)
        weight[y==0] = downSample
        yield y, weight, dense_discrete, dense_continue
        data.curr = data.curr + pd.DateOffset(seqSize)
        
def MLP_generator(data,batchSize,seqSize,df,key,discreteList,shuffle=True,downSample=1):
    # return bool (if init should be reset) 
    # and a list [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)
    # finetune by use store-item pair in testset only
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    n = data.shape[0]

    for from_ in range(0,n-batchSize,batchSize):
        X = list(data.loc[from_:from_+batchSize-1,'store_nbr':'cluster'].values.astype(np.int32).T)
        weight = np.ones((batchSize,1),dtype=np.float32)
        weight[data.loc[from_:from_+batchSize-1,'perishable']==1] = 1.25
        for y,w,Xdiscrete,Xcontinue in MLP_timeGenerator(\
                           data.loc[from_:from_+batchSize-1,'date':'amax'],seqSize,downSample,df,key,discreteList):
            yield [y,weight*w,Xcontinue] + Xdiscrete + X
            
def MLP_timeGenerator(data,seqSize,downSample,df,key,discreteList):
    # df is time related variables like holiday and seasonality
    # returns y, weight, X_discrete, X_continue
    n = data.shape[0]
    sparse = pd.concat([data.apply(lambda x: pd.Series(x.date,name='date'),axis=1)\
                                    .stack().reset_index().drop(['level_1'],1),\
                        data.apply(lambda x: pd.Series(x.unit_sales),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1),\
                        data.apply(lambda x: pd.Series(x.onpromotion),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1)],1)
    sparse.columns = ['level','date','sales','onpromotion']

    dense = data.apply(lambda x:pd.Series(pd.date_range(x.amin,x.amax))\
                        .iloc[np.random.choice(x.countDays,seqSize,False)],axis=1)\
                        .stack().reset_index().drop(['level_1'],1)
    dense.columns = ['level','date']
    dense = pd.merge(pd.merge(dense, df[np.random.randint(10)], how='left', on=key),
                    sparse,how='left',on=['level','date']).fillna(0)
    dense_continue = dense['dcoilwtico'].values.astype(np.float32)\
                              .reshape((n,seqSize,1))
    dense_discrete = list(np.moveaxis(dense[discreteList].values.astype(np.int32)\
                                            .reshape((n,seqSize,len(discreteList)))\
                                      ,2,0))
    y = dense['sales'].values.astype(np.float32).reshape((n,seqSize))
    weight = np.ones_like(y,dtype=np.float32)
    weight[y==0] = downSample
    yield y, weight, dense_discrete, dense_continue

def MLP_Test_generator(data,batchSize,seqSize,df,key,discreteList):
    # return bool (if init should be reset) 
    # and a list [id (B,T),Xcontinue of shape (B,T,1)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)
    # finetune by use store-item pair in testset only
    n = data.shape[0]

    for from_ in range(0,n,batchSize):
        X = list(data.loc[from_:from_+batchSize-1,'store_nbr':'cluster'].values.astype(np.int32).T)

        for id_,Xdiscrete,Xcontinue in MLP_Test_timeGenerator(\
                           data.loc[from_:from_+batchSize-1,'id':'amax'],seqSize,df,key,discreteList):
            yield [id_,Xcontinue] + Xdiscrete + X
            
def MLP_Test_timeGenerator(data,seqSize,df,key,discreteList):
    # df is time related variables like holiday and seasonality
    # returns id_, X_discrete, X_continue
    n = data.shape[0]
    sparse = pd.concat([data.apply(lambda x: pd.Series(x.id),axis=1)\
                                    .stack().reset_index().drop(['level_1'],1),\
                        data.apply(lambda x: pd.Series(x.onpromotion),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1)],1)
    sparse.columns = ['level','id','onpromotion']

    dense = data.apply(lambda x:pd.Series(pd.date_range(x.amin,x.amax))\
                        .iloc[np.random.choice(x.countDays,seqSize,False)],axis=1)\
                        .stack().reset_index().drop(['level_1'],1)
    dense.columns = ['level','date']
    dense = pd.merge(pd.merge(dense, df[np.random.randint(10)], how='left', on=key),
                    sparse,how='left',on=['level','date']).fillna(0)
    dense_continue = dense['dcoilwtico'].values.astype(np.float32)\
                              .reshape((n,seqSize,1))
    dense_discrete = list(np.moveaxis(dense[discreteList].values.astype(np.int32)\
                                            .reshape((n,seqSize,len(discreteList)))\
                                      ,2,0))
    y = dense['sales'].values.astype(np.float32).reshape((n,seqSize))
    weight = np.ones_like(y,dtype=np.float32)
    #weight[y==0] = downSample
    yield y, weight, dense_discrete, dense_continue