#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:12:15 2017

@author: will
"""
import numpy as np
import pandas as pd

def isConsecutive(seq):
    # if non-consecutive, need to re-map to consecutive number starting from 1
    uniq = pd.unique(seq)
    return len(uniq) == (uniq.max()-uniq.min() + 1)

def isUniqBigger(seq1,seq2):
    return set(seq1) >= set(seq2)

def reMapDF(df,cols):
    # remap cols in dataframe to consecutive integers starting from one
    for col in cols:
        uniq = pd.unique(df[col])
        dict_ = {item:i+1 for i,item in enumerate(uniq)}
        df = df.replace({col:dict_})
    return df

def dimentionDF(df,cols):
    return {col:len(set(df[col])) for col in cols}

def mergeFillCast(df1,df2,key):
    cols = df2.columns.values
    types = df2.dtypes.values
    dict_ = {col:type_ for col,type_ in zip(cols,types)}
    dfOut = pd.merge(df1, df2, how='left', on=key, 
             suffixes=('', '_y'), copy=True, indicator=False).fillna(0)
    dfOut[cols] = \
        dfOut[cols].astype(dict_)
    return dfOut

def mergeFillCastsss(df0,dfs,keys):
    for df,key in zip(dfs,keys):
        df0 = mergeFillCast(df0,df,key)
    return df0

def createDataMain(MLP):
    types = {'id': 'int32',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': bool}
    train = pd.read_csv('train.csv',usecols=['date','item_nbr','store_nbr','unit_sales','onpromotion'],\
                    parse_dates=['date'],dtype=types, infer_datetime_format=True)
    train = train.fillna(2,axis=1)
    train.onpromotion = train.onpromotion.astype(np.int8)
    train.loc[train.unit_sales<0,'unit_sales'] = .0 # clip negative sales to zero
    val = train[train.date >= '2017-07-31']
    train = train[train.date < '2017-07-31']
    item_uniq = pd.unique(train.item_nbr)
    item_dict = {item:i+1 for i,item in enumerate(item_uniq)}
    iter_mapping = lambda x: item_dict[x] if x in item_dict else 0
    
    items = pd.read_csv('items.csv')
    stores = pd.read_csv('stores.csv')
    items2 = reMapDF(items,['family','class'])
    items2[['family','class','perishable']] = \
            items2[['family','class','perishable']].astype('int16')
    stores2 = reMapDF(stores,['city', 'state', 'type'])
    stores2 = stores2.astype('int8')        

    def CreateData(data):
        SI_train_sales = data.groupby(['store_nbr','item_nbr'])[['date','unit_sales','onpromotion']].\
                        agg(lambda x: tuple(x)).reset_index()
        storeTime = data.groupby(['store_nbr'])['date'].agg([np.min,np.max]).reset_index()
        dfs = [items2,stores2,storeTime]
        keys = ['item_nbr','store_nbr','store_nbr']
        SI_train = mergeFillCastsss(SI_train_sales,dfs,keys)
        SI_train['item_nbr'] = SI_train.item_nbr.map(iter_mapping)
        SI_train['amin'] = pd.to_datetime(SI_train['amin'])
        SI_train['amax'] = pd.to_datetime(SI_train['amax'])
        if MLP:
            SI_train['countDays'] = (SI_train['amax'] - SI_train['amin']).dt.days + 1
            
        return SI_train[['store_nbr',
                         'item_nbr',
                         'family',
                         'class',
                         'perishable',
                         'city',
                         'state',
                         'type',
                         'cluster',
                         'date',
                         'unit_sales', 
                         'onpromotion',
                         'countDays',
                         'amin',
                         'amax']]

    trainRNN = CreateData(train)
    valRNN = CreateData(val)
    return trainRNN,valRNN

def createTestDataMain(isTrain=True):
    types = {'id': 'int32',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': bool}
    train = pd.read_csv('train.csv',usecols=['date','item_nbr','store_nbr','unit_sales','onpromotion'],\
                    parse_dates=['date'],dtype=types, infer_datetime_format=True)
    train = train.fillna(2,axis=1)
    train.onpromotion = train.onpromotion.astype(np.int8)
    test = pd.read_csv('test.csv',parse_dates=['date'],dtype=types, infer_datetime_format=True)
    test = test.fillna(2,axis=1)
    test.onpromotion = test.onpromotion.astype(np.int8)
    train.loc[train.unit_sales<0,'unit_sales'] = .0 # clip negative sales to zero
    if isTrain:
        train = train[train.date < '2017-07-31']
    item_uniq = pd.unique(train.item_nbr)
    item_dict = {item:i+1 for i,item in enumerate(item_uniq)}
    iter_mapping = lambda x: item_dict[x] if x in item_dict else 0
    
    items = pd.read_csv('items.csv')
    stores = pd.read_csv('stores.csv')
    items2 = reMapDF(items,['family','class'])
    items2[['family','class','perishable']] = \
            items2[['family','class','perishable']].astype('int16')
    stores2 = reMapDF(stores,['city', 'state', 'type'])
    stores2 = stores2.astype('int8')        

    def CreateData(data):
        SI_train_sales = data.groupby(['store_nbr','item_nbr'])[['id','onpromotion']].\
                        agg(lambda x: tuple(x)).reset_index()
        storeTime = data.groupby(['store_nbr'])['date'].agg([np.min,np.max]).reset_index()
        dfs = [items2,stores2,storeTime]
        keys = ['item_nbr','store_nbr','store_nbr']
        SI_train = mergeFillCastsss(SI_train_sales,dfs,keys)
        SI_train['amin'] = pd.to_datetime(SI_train['amin'])
        SI_train['amax'] = pd.to_datetime(SI_train['amax'])
        return SI_train[['store_nbr',
                         'item_nbr',
                         'family',
                         'class',
                         'perishable',
                         'city',
                         'state',
                         'type',
                         'cluster',
                         'id',                         
                         'onpromotion',
                         'amin',
                         'amax']]
    
    def splitTest(test,train):
        train['S_I'] = train.item_nbr + train.store_nbr/100.0
        test['S_I'] = test.item_nbr + test.store_nbr/100.0
        train_SI = set(train.S_I)
        test_SI = set(test.S_I)
        train_I = set(train.item_nbr)
        test_I = set(test.item_nbr)
        SI_intersection = train_SI & test_SI
        I_newItem = test_I - train_I
        index_SI = test.S_I.isin(SI_intersection)
        index_I_newItem = test.item_nbr.isin(I_newItem)
        #index_other = ~(index_SI | index_I_newItem)
        return test[index_SI].drop('S_I',1),test[index_I_newItem].drop('S_I',1)
    
    test_SI,test_newItem = splitTest(CreateData(test),train)
    test_SI['item_nbr'] = test_SI.item_nbr.map(iter_mapping)
    test_newItem['item_nbr'] = test_newItem.item_nbr.map(iter_mapping)
    
    return test_SI.reset_index(drop=True),test_newItem.reset_index(drop=True)


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
    weight[y==0] = downSample
    yield y, weight, dense_discrete, dense_continue