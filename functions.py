#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:12:15 2017
@author: will
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import cPickle
import os
import re


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

def createDataMain(isTrain):
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
    with open('item_dict_train.pickle' if isTrain else 'item_dict_final.pickle', "rb") as input_file:
        item_dict = cPickle.load(input_file)
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
    if isTrain:
        val = train[train.date >= '2017-07-31']
        train = train[train.date < '2017-07-31']
        trainRNN = CreateData(train)
        val_SI,val_newItem = splitTest(CreateData(val),trainRNN)
        return trainRNN.reset_index(drop=True),val_SI.reset_index(drop=True),val_newItem.reset_index(drop=True)
    else:
        trainRNN = CreateData(train)
        return trainRNN.reset_index(drop=True)


def createTestDataMain(isTrain):
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
    with open('item_dict_train.pickle' if isTrain else 'item_dict_final.pickle', "rb") as input_file:
        item_dict = cPickle.load(input_file)
    iter_mapping = lambda x: item_dict[x] if x in item_dict else 0
    
    items = pd.read_csv('items.csv')
    stores = pd.read_csv('stores.csv')
    items2 = reMapDF(items,['family','class'])
    items2[['family','class','perishable']] = \
            items2[['family','class','perishable']].astype('int16')
    stores2 = reMapDF(stores,['city', 'state', 'type'])
    stores2 = stores2.astype('int8')        

    def CreateData(data):
        SI_train_sales = data.groupby(['store_nbr','item_nbr'])[['date','id','onpromotion']].\
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
                         'date',
                         'id',                         
                         'onpromotion',
                         'amin',
                         'amax']]
    
    
    test_SI,test_newItem = splitTest(CreateData(test),train)
    test_SI['item_nbr'] = test_SI.item_nbr.map(iter_mapping)
    test_newItem['item_nbr'] = test_newItem.item_nbr.map(iter_mapping)
    
    return test_SI.reset_index(drop=True),test_newItem.reset_index(drop=True)





def pd2np(data,batchSize,seqSize,df,key,discreteList):
    
    def RNN_generator2(data,batchSize,seqSize,df,key,discreteList):
        n = data.shape[0]
        for from_ in range(0,n,batchSize):
            for y,Xdiscrete,Xcontinue in timeGenerator2(data.loc[from_:from_+batchSize-1,'date':'amax'],\
                                                          seqSize,df,key,discreteList):
                yield y,Xcontinue, Xdiscrete
                
    def timeGenerator2(data,seqSize,df,key,discreteList):
        # df is time related variables like holiday and seasonality
        # returns y, X_discrete, X_continue
        n = data.shape[0]
        sparse = pd.concat([data.apply(lambda x: pd.Series(x.date,name='date'),axis=1)\
                                        .stack().reset_index().drop(['level_1'],1),\
                            data.apply(lambda x: pd.Series(x.unit_sales),axis=1)\
                                        .stack().reset_index().drop(['level_0','level_1'],1),\
                            data.apply(lambda x: pd.Series(x.onpromotion),axis=1)\
                                        .stack().reset_index().drop(['level_0','level_1'],1)],1)
        sparse.columns = ['level','date','sales','onpromotion']
    
        dense = data.apply(lambda x:pd.Series(pd.date_range(x.amin,periods=seqSize)),axis=1)\
                            .stack().reset_index().drop(['level_1'],1)
        dense.columns = ['level','date']
        dense = pd.merge(pd.merge(dense, df[np.random.randint(10)], how='left', on=key),
                        sparse,how='left',on=['level','date']).fillna(0)
        dense_continue = dense['dcoilwtico'].values.astype(np.float32).reshape((n,seqSize))
        dense_discrete = dense[discreteList].values.astype(np.int32)\
                            .reshape((n,seqSize,len(discreteList)))
        y = dense['sales'].values.astype(np.float32).reshape((n,seqSize))
    
        yield y, dense_discrete, dense_continue
    
    n = data.shape[0]
    Y,Con,Dis = [],[],[]
    for y,con,dis in RNN_generator2(data,batchSize,seqSize,df,key,discreteList):
        Y.append(y)
        Con.append(con)
        Dis.append(dis)       
    weight = np.ones((n,1),dtype=np.float32)
    weight[data['perishable']==1] = 1.25
    return np.concatenate(Y), weight,np.concatenate(Con),\
            np.concatenate(Dis),data.loc[:,'store_nbr':'cluster'].values.astype(np.int32),\
            data['countDays'].values.astype(np.int32)

def pd2np_test(data,batchSize,seqSize,df,key,discreteList):
    
    def RNN_generator2(data,batchSize,seqSize,df,key,discreteList):
        n = data.shape[0]
        for from_ in range(0,n,batchSize):
            for y,Xdiscrete,Xcontinue in timeGenerator2(data.loc[from_:from_+batchSize-1,'date':'amax'],\
                                                          seqSize,df,key,discreteList):
                yield y,Xcontinue, Xdiscrete
                
    def timeGenerator2(data,seqSize,df,key,discreteList):
        # df is time related variables like holiday and seasonality
        # returns y, X_discrete, X_continue
        n = data.shape[0]
        sparse = pd.concat([data.apply(lambda x: pd.Series(x.date,name='date'),axis=1)\
                                        .stack().reset_index().drop(['level_1'],1),\
                            data.apply(lambda x: pd.Series(x.id),axis=1)\
                                        .stack().reset_index().drop(['level_0','level_1'],1),\
                            data.apply(lambda x: pd.Series(x.onpromotion),axis=1)\
                                        .stack().reset_index().drop(['level_0','level_1'],1)],1)
        sparse.columns = ['level','date','id','onpromotion']
    
        dense = data.apply(lambda x:pd.Series(pd.date_range(x.amin,periods=seqSize)),axis=1)\
                            .stack().reset_index().drop(['level_1'],1)
        dense.columns = ['level','date']
        dense = pd.merge(pd.merge(dense, df[np.random.randint(10)], how='left', on=key),
                        sparse,how='left',on=['level','date']).fillna(0)
        dense_continue = dense['dcoilwtico'].values.astype(np.float32).reshape((n,seqSize))
        dense_discrete = dense[discreteList].values.astype(np.int32)\
                            .reshape((n,seqSize,len(discreteList)))
        y = dense['id'].values.astype(np.int32).reshape((n,seqSize))
    
        yield y, dense_discrete, dense_continue
    
    Y,Con,Dis = [],[],[]
    for y,con,dis in RNN_generator2(data,batchSize,seqSize,df,key,discreteList):
        Y.append(y)
        Con.append(con)
        Dis.append(dis)       

    return np.concatenate(Y),np.concatenate(Con),\
            np.concatenate(Dis),data.loc[:,'store_nbr':'cluster'].values.astype(np.int32)

def MLP_generator(y_np, weight_np,Con_np,Dis_list,X_np,Count_np,\
                  batchSize,seqSize,shuffle=True,downSample=1):
    # yield a list [y,Weight,X_continuous] + Xt + X
    n = y_np.shape[0]
    index_ = np.random.permutation(n) if shuffle else np.arange(n)
    for from_ in range(0,n-batchSize,batchSize):
        Index_X = np.reshape(index_[from_:from_+batchSize],(-1,1))
        minT = np.min(Count_np[Index_X])
        Index_Y = np.random.randint(0,minT,(batchSize,seqSize))
        y_ = y_np[Index_X,Index_Y]
        weight = np.ones_like(y_,dtype=np.float32)
        weight[y_==0] = downSample
        yield [y_,weight_np[Index_X]*weight,np.reshape(Con_np[Index_X,Index_Y],(batchSize,seqSize,1))]\
                + [dis[Index_X,Index_Y] for dis in Dis_list] \
                + list(X_np[Index_X.squeeze()].T)

def RNN_generator(y_np, weight_np,Con_np,Dis_list,X_np,Count_np,\
                  batchSize,seqSize,bucketSize,downSample=1):
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)
    # data needs to by sorted by count_np !!
    n = y_np.shape[0]
    bucketNum = n//bucketSize
    weight_adj = n%bucketSize*1.0/bucketSize
    for bucket in np.random.permutation(bucketNum+1):
        Index_X = np.random.randint(bucket*bucketSize,min(n,(bucket+1)*bucketSize),(batchSize,1))
        X_list_ = list(X_np[Index_X.squeeze()].T)
        adj_ = weight_adj if bucket==bucketNum else 1
        for t_ in range(0,np.min(Count_np[Index_X])-seqSize,seqSize):
            Index_Y = np.arange(t_,t_+seqSize)
            Index_Y_1 = np.arange(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield [y_,weight_np[Index_X]*weight*adj_,\
                     np.stack([Con_np[Index_X,Index_Y_1],y_np[Index_X,Index_Y]],-1)]\
                     + [dis[Index_X,Index_Y_1] for dis in Dis_list]\
                     + X_list_ + [t_==0]    
    
def test_MLP_generator(y_np, Con_np,Dis_list,X_np,batchSize):
    # yield a list [y,Weight,X_continuous] + Xt + X
    n = y_np.shape[0]
    for from_ in range(0,n,batchSize):
        yield y_np[from_:from_+batchSize], [Con_np[from_:from_+batchSize,:,np.newaxis]]\
                + [dis[from_:from_+batchSize] for dis in Dis_list] \
                + list(X_np[from_:from_+batchSize].T)    
    
                     
                     
def createGraphRNN(batch_size,seq_len,cardinalitys_X,cardinalitys_T,dimentions_X,dimentions_T,\
                dX,d,keep_prob,n_layers,grad_clip):
    tf.reset_default_graph()
    embedding_X = [tf.get_variable("embedding_X"+str(i), [car, dim],\
                                   initializer=tf.truncated_normal_initializer()) \
                for i,(car,dim) in enumerate(zip(cardinalitys_X,dimentions_X))]
    embedding_Xt = [tf.get_variable("embedding_Xt"+str(i), [car, dim],\
                                    initializer=tf.truncated_normal_initializer()) \
                    for i,(car,dim) in enumerate(zip(cardinalitys_T,dimentions_T))]

    learning_rate = tf.placeholder(tf.float32,shape=[])
    init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
    X = [tf.placeholder(tf.int32, [batch_size,], name='X_'+str(i)) for i,_ in enumerate(dimentions_X)]
    Xt = [tf.placeholder(tf.int32, [batch_size,seq_len], name='Xt_'+str(i)) for i,_ in enumerate(dimentions_T)]
    X_continuous = tf.placeholder(tf.float32, [batch_size,seq_len,2], name='X_continuous')
    Weight = tf.placeholder(tf.float32, [batch_size,seq_len], name='Weight')
    y = tf.placeholder(tf.float32, [batch_size,seq_len], name='y')
    IsStart = tf.placeholder(tf.bool, [], name='IsStart')
    inputs = [y,Weight,X_continuous] + Xt + X + [IsStart,learning_rate,init_state]
    Xt1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_Xt,Xt)] + [X_continuous],2)
    X1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_X,X)],1)    
    Xall = [tf.concat([xt,X1],1) for xt in tf.unstack(Xt1,axis=1)]
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(d),output_keep_prob=keep_prob) \
                                        for _ in range(n_layers)])
    weights_init = tf.Variable(tf.truncated_normal([dX,d*n_layers],
                        stddev=1.0 / np.sqrt(dX)),name='weights_init')
    biases_init = tf.Variable(tf.zeros([d*n_layers]),
                         name='biases_init')
    init_state2 = tf.cond(IsStart,\
                        lambda:tuple(tf.split(tf.matmul(X1,weights_init)+biases_init,n_layers,1)),\
                        lambda:init_state)
    outputs, state = tf.contrib.rnn.static_rnn(cell,Xall,init_state2)
    outputs_flat = tf.stack(outputs,1)
    weights_out = tf.Variable(tf.truncated_normal([d],
                        stddev=1.0 / np.sqrt(d)),name='weights_out')
    biases_out = tf.Variable(tf.zeros([1]),
                         name='biases_out')
    yhat = tf.nn.relu(tf.einsum('btp,p->bt', outputs_flat, weights_out) + biases_out)    
        
    cost = tf.reduce_mean(Weight*(tf.log((yhat+1)/(y+1)))**2)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),grad_clip)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    saver = tf.train.Saver()     
    
    return inputs,train_op,cost,saver,yhat,state
    
def createGraphANN(batch_size,seq_len,cardinalitys_X,cardinalitys_T,\
                dimentions_X,dimentions_T,dX,dT,d,lambdas,l1,l2,l3):
    tf.reset_default_graph()
    embedding_X = [tf.get_variable("embedding_X"+str(i), [car, dim],\
                                   initializer=tf.truncated_normal_initializer()) \
                for i,(car,dim) in enumerate(zip(cardinalitys_X,dimentions_X))]
    embedding_Xt = [tf.get_variable("embedding_Xt"+str(i), [car, dim],\
                                    initializer=tf.truncated_normal_initializer()) \
                    for i,(car,dim) in enumerate(zip(cardinalitys_T,dimentions_T))]
    embeddings = embedding_X + embedding_Xt
    learning_rate = tf.placeholder(tf.float32,shape=[])
    X = [tf.placeholder(tf.int32, [batch_size,], name='X_'+str(i)) for i,_ in enumerate(dimentions_X)]
    Xt = [tf.placeholder(tf.int32, [batch_size,seq_len], name='Xt_'+str(i)) for i,_ in enumerate(dimentions_T)]
    X_continuous = tf.placeholder(tf.float32, [batch_size,seq_len,1], name='X_continuous')
    Weight = tf.placeholder(tf.float32, [batch_size,seq_len], name='Weight')
    y = tf.placeholder(tf.float32, [batch_size,seq_len], name='y')
    inputs = [y,Weight,X_continuous] + Xt + X + [learning_rate]
    Xt1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_Xt,Xt)] + [X_continuous],2)
    X1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_X,X)],1)
    weights1_X = tf.Variable(tf.truncated_normal([dX,d],
                        stddev=1.0 / np.sqrt(dX)),name='weights1_X')
    weights1_Xt = tf.Variable(tf.truncated_normal([dT+1,d],
                            stddev=1.0 / np.sqrt(dT)),name='weights1_Xt')
    biases1 = tf.Variable(tf.zeros([d]),
                         name='biases1')
    X2 = tf.nn.relu(biases1 + \
                   tf.reshape(tf.matmul(X1,weights1_X),(-1 if batch_size is None else batch_size,1,d))+\
                   tf.einsum('btp,pq->btq', Xt1, weights1_Xt))
    weights2 = tf.Variable(tf.truncated_normal([d,d],
                        stddev=1.0 / np.sqrt(d)),name='weights2')
    biases2 = tf.Variable(tf.zeros([d]),
                         name='biases2')
    X3 = tf.nn.relu(tf.einsum('btp,pq->btq', X2, weights2) + biases2)
    weights3 = tf.Variable(tf.truncated_normal([d],
                        stddev=1.0 / np.sqrt(d)),name='weights3')
    biases3 = tf.Variable(tf.zeros([1]),
                         name='biases3')
    yhat = tf.nn.relu(tf.einsum('btp,p->bt', X3, weights3) + biases3) # as target is always positive
    regularizer = sum([l*tf.reduce_sum(v**2) for l,v in zip(lambdas,embeddings)])\
                + l1 * tf.reduce_sum(weights1_X**2) + l1 * tf.reduce_sum(weights1_Xt**2)\
                + l2 * tf.reduce_sum(weights2**2)\
                + l3 * tf.reduce_sum(weights3**2)
    cost = tf.reduce_mean(Weight*(tf.log((yhat+1)/(y+1)))**2)
    augment_cost = cost + regularizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(augment_cost)
    saver = tf.train.Saver()  
    return inputs,train_op,cost,saver,yhat        
    

def init_state_update(sess,inputs,state,batch_size,d,n_layers,y_np,Con_np,X_np,Count_np,Dis_np):
    # init_tot_list (B,d,n_layers)
    init_state = tuple([np.zeros((batch_size,d),dtype=np.float32) for i in range(n_layers)])
    n = Con_np.shape[0]
    init_tot_list = []
    for b in range(0,n,batch_size):
        X_list_ = list(X_np[b:b+batch_size].T)
        count_ = Count_np[b:b+batch_size]
        max_ = np.max(count_) - 1
        init_list = []
        for t_ in range(0,max_):
            init_state = sess.run(state,dict(zip(inputs[2:-2] + [inputs[-1]],\
                        [np.stack([Con_np[b:b+batch_size,t_+1:t_+2],y_np[b:b+batch_size,t_:t_+1]],-1)]\
                         + [dis[b:b+batch_size,t_+1:t_+2] for dis in Dis_np]\
                         + X_list_ + [t_==0,init_state])))
            init_list.append(np.stack(init_state,2))
        init_tot_list.append(np.stack(init_list,3)[np.arange(count_.shape[0]),:,:,count_-2]) 
    init_tot_list = np.concatenate(init_tot_list,0)
    return init_tot_list

LSTM2list = lambda x: [x.c,x.h]

def init_state_update_LSTM(sess,inputs,state,batch_size,d,n_layers,y_np,Con_np,X_np,Count_np,Dis_np):
    # return shape (B,d,2,layers)
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((batch_size,d),dtype=np.float32),\
                                                      np.zeros((batch_size,d),dtype=np.float32))\
                                                        for i in range(n_layers)]) 
    n = Con_np.shape[0]
    init_tot_list = []
    for b in range(0,n,batch_size):
        X_list_ = list(X_np[b:b+batch_size].T)
        count_ = Count_np[b:b+batch_size]
        max_ = np.max(count_) - 1
        init_list = []
        for t_ in range(0,max_):
            init_state = sess.run(state,dict(zip(inputs[2:-2] + [inputs[-1]],\
                        [np.stack([Con_np[b:b+batch_size,t_+1:t_+2],y_np[b:b+batch_size,t_:t_+1]],-1)]\
                         + [dis[b:b+batch_size,t_+1:t_+2] for dis in Dis_np]\
                         + X_list_ + [t_==0,init_state])))
            init_list.append(np.stack([np.stack(LSTM2list(element),2) for element in init_state],3))
        init_tot_list.append(np.stack(init_list,4)[np.arange(count_.shape[0]),:,:,:,count_-2]) 
    init_tot_list = np.concatenate(init_tot_list,0)
    return init_tot_list

def RNN_forecast_LSTM(sess,inputs,state,yhat,batch_size,n_layers,\
                 y_np,Con_np,X_np,Dis_np,init_tot_list):
    # y_np is of shape (B,) for the last point in time sales
    n,T_ = Con_np.shape
    y_tot_list = []
    for b in range(0,n,batch_size):
        X_list_ = list(X_np[b:b+batch_size].T)
        y_list = []
        for t_ in range(0,T_):
            if t_ == 0:
                init_state = tuple([tf.contrib.rnn.LSTMStateTuple(temp_.squeeze()[:,:,0],temp_.squeeze()[:,:,1]) \
                                    for temp_ in np.split(init_tot_list[b:b+batch_size],n_layers,3)])
            init_state,y_np[b:b+batch_size] = sess.run([state,yhat],dict(zip(inputs[2:-2] + [inputs[-1]],\
                        [np.stack([Con_np[b:b+batch_size,t_:t_+1],y_np[b:b+batch_size]],-1)]\
                         + [dis[b:b+batch_size,t_:t_+1] for dis in Dis_np]\
                         + X_list_ + [False,init_state])))
            y_list.append(np.copy(y_np[b:b+batch_size].squeeze()))
        y_tot_list.append(np.stack(y_list,1)) 
    y_tot_list = np.concatenate(y_tot_list,0)
    return y_tot_list


def RNN_forecast(sess,inputs,state,yhat,batch_size,n_layers,\
                 y_np,Con_np,X_np,Dis_np,init_tot_list):
    # y_np is of shape (B,) for the last point in time sales
    n,T_ = Con_np.shape
    y_tot_list = []
    for b in range(0,n,batch_size):
        X_list_ = list(X_np[b:b+batch_size].T)
        y_list = []
        for t_ in range(0,T_):
            if t_ == 0:
                init_state = tuple([temp_.squeeze() for temp_ in np.split(init_tot_list[b:b+batch_size],n_layers,2)])
            init_state,y_np[b:b+batch_size] = sess.run([state,yhat],dict(zip(inputs[2:-2] + [inputs[-1]],\
                        [np.stack([Con_np[b:b+batch_size,t_:t_+1],y_np[b:b+batch_size]],-1)]\
                         + [dis[b:b+batch_size,t_:t_+1] for dis in Dis_np]\
                         + X_list_ + [False,init_state])))
            y_list.append(np.copy(y_np[b:b+batch_size].squeeze()))
        y_tot_list.append(np.stack(y_list,1)) 
    y_tot_list = np.concatenate(y_tot_list,0)
    return y_tot_list

def RNN_forecast_Repeat(repeat,sess,inputs,state,yhat,batch_size,n_layers,\
                 y_np,Con_np,X_np,Dis_np,init_tot_list):
    y_SI = np.zeros_like(Con_np)
    for i in range(repeat):
        y_SI = y_SI + RNN_forecast(sess,inputs,state,yhat,batch_size,n_layers,\
                                 y_np,Con_np,X_np,Dis_np,init_tot_list)
    return y_SI/repeat

def RNN_forecast_Repeat_LSTM(repeat,sess,inputs,state,yhat,batch_size,n_layers,\
                 y_np,Con_np,X_np,Dis_np,init_tot_list):
    y_SI = np.zeros_like(Con_np)
    for i in range(repeat):
        y_SI = y_SI + RNN_forecast_LSTM(sess,inputs,state,yhat,batch_size,n_layers,\
                                 y_np,Con_np,X_np,Dis_np,init_tot_list)
    return y_SI/repeat
       
def loss_func(Weight,yhat,y):
    return np.sqrt(np.sum(Weight*(np.log((yhat+1)/(y+1)))**2)/np.sum(Weight)/16)
    
    

#''' setup '''
#discreteList = ['dayOfWeek','payDay','month','earthquake','type','locale','locale_name','transferred','onpromotion']
#cardinalitys_X = [55, 4001, 34, 337, 2, 23, 17, 6, 18]
#cardinalitys_T = [7, 2, 13, 2, 7, 4, 25, 2, 3]
#dimentions_X = [2, 20, 1, 2, 1, 1, 1, 1, 1]
#dimentions_T = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#dX = sum(dimentions_X)
#dT = sum(dimentions_T)
#d = dX + dT + 2 # 2 for two cont variables
#learningRate = 1e-4
#epoch = 30
#bucketSize = 10000
#prefix = 'train'
#y_np = np.loadtxt(prefix+'_Y',dtype=np.float32, delimiter=",") 
#weight_np = np.loadtxt(prefix+'_Weight',dtype=np.float32, delimiter=",") 
#Con_np = np.loadtxt(prefix+'_Con',dtype=np.float32, delimiter=",") 
#X_np = np.loadtxt(prefix+'_X',dtype=np.int32,delimiter=",") 
#Count_np = np.loadtxt(prefix+'_Count',dtype=np.int32,delimiter=",") 
#Dis_np = [np.loadtxt(prefix+'_Dis'+str(j),dtype=np.int32,delimiter=",")  for j in range(len(discreteList))]
#prefix = 'val_SI'
#y_np_val = np.loadtxt(prefix+'_Y',dtype=np.float32, delimiter=",") 
#weight_np_val = np.loadtxt(prefix+'_Weight',dtype=np.float32, delimiter=",") 
#Con_np_val = np.loadtxt(prefix+'_Con',dtype=np.float32, delimiter=",") 
#X_np_val = np.loadtxt(prefix+'_X',dtype=np.int32,delimiter=",") 
#Count_np_val = np.loadtxt(prefix+'_Count',dtype=np.int32,delimiter=",") 
#Dis_np_val = [np.loadtxt(prefix+'_Dis'+str(j),dtype=np.int32,delimiter=",")  for j in range(len(discreteList))]
#index = np.loadtxt('Index_val',dtype=np.int32,delimiter=",") 
#''' setup '''

def createGraphRNN2(batch_size,seq_len,cardinalitys_X,cardinalitys_T,dimentions_X,dimentions_T,\
                dX,d,keep_prob,n_layers,grad_clip,cell_type,optimizer,actFun):

    tf.reset_default_graph()
    embedding_X = [tf.get_variable("embedding_X"+str(i), [car, dim],\
                                   initializer=tf.truncated_normal_initializer()) \
                for i,(car,dim) in enumerate(zip(cardinalitys_X,dimentions_X))]
    embedding_Xt = [tf.get_variable("embedding_Xt"+str(i), [car, dim],\
                                    initializer=tf.truncated_normal_initializer()) \
                    for i,(car,dim) in enumerate(zip(cardinalitys_T,dimentions_T))]

    learning_rate = tf.placeholder(tf.float32,shape=[])
    X = [tf.placeholder(tf.int32, [batch_size,], name='X_'+str(i)) for i,_ in enumerate(dimentions_X)]
    Xt = [tf.placeholder(tf.int32, [batch_size,seq_len], name='Xt_'+str(i)) for i,_ in enumerate(dimentions_T)]
    X_continuous = tf.placeholder(tf.float32, [batch_size,seq_len,2], name='X_continuous')
    Weight = tf.placeholder(tf.float32, [batch_size,seq_len], name='Weight')
    y = tf.placeholder(tf.float32, [batch_size,seq_len], name='y')
    IsStart = tf.placeholder(tf.bool, [], name='IsStart')
    Xt1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_Xt,Xt)] + [X_continuous],2)
    X1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_X,X)],1)    
    Xall = [tf.concat([xt,X1],1) for xt in tf.unstack(Xt1,axis=1)]
    if actFun == 'tanh':
        actFun = tf.tanh
    else:
        actFun = tf.nn.relu
    # [lstm_cell() for _ in range(number_of_layers)])
    if cell_type == 'residual':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.ResidualWrapper(\
                                            tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'highway':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.HighwayWrapper(\
                                            tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])        
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'NormLSTM':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(d,activation=actFun,dropout_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])
        init_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, [batch_size,d], name='initC_'+str(i)),\
                                                          tf.placeholder(tf.float32, [batch_size,d], name='initH_'+str(i))) \
                            for i in range(n_layers)])
        factor = 2
        
    inputs = [y,Weight,X_continuous] + Xt + X + [IsStart,learning_rate,init_state]
    weights_init = tf.Variable(tf.truncated_normal([dX,d*n_layers*factor],
                        stddev=1.0 / np.sqrt(dX)),name='weights_init')
    biases_init = tf.Variable(tf.zeros([d*n_layers*factor]),
                         name='biases_init')
    if cell_type == 'NormLSTM':
        init_state2 = tf.cond(IsStart,\
                            lambda:tuple([tf.contrib.rnn.LSTMStateTuple(*tf.split(tensor_,2,1)) \
                                          for tensor_ in tf.split(tf.matmul(X1,weights_init)+biases_init,n_layers,1)]),\
                            lambda:init_state)
    else:    
        init_state2 = tf.cond(IsStart,\
                            lambda:tuple(tf.split(tf.matmul(X1,weights_init)+biases_init,n_layers,1)),\
                            lambda:init_state)
    outputs, state = tf.contrib.rnn.static_rnn(cell,Xall,init_state2)
    outputs_flat = tf.stack(outputs,1)
    weights_out = tf.Variable(tf.truncated_normal([d],
                        stddev=1.0 / np.sqrt(d)),name='weights_out')
    biases_out = tf.Variable(tf.zeros([1]),
                         name='biases_out')
    yhat = tf.nn.relu(tf.einsum('btp,p->bt', outputs_flat, weights_out) + biases_out)    
        
    cost = tf.reduce_mean(Weight*(tf.log((yhat+1)/(y+1)))**2)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),grad_clip)
    if optimizer =='SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    saver=tf.train.Saver({v.op.name:v for v in tf.trainable_variables()})        
    
    return inputs,train_op,cost,saver,yhat,state


def createGraphRNN_dynamic(batch_size,seq_len,cardinalitys_X,cardinalitys_T,dimentions_X,dimentions_T,\
                           dX,d,keep_prob,n_layers,grad_clip,cell_type,optimizer,actFun,StopGrad):
    
    tf.reset_default_graph()
    embedding_X = [tf.get_variable("embedding_X"+str(i), [car, dim],\
                                   initializer=tf.truncated_normal_initializer()) \
                for i,(car,dim) in enumerate(zip(cardinalitys_X,dimentions_X))]
    embedding_Xt = [tf.get_variable("embedding_Xt"+str(i), [car, dim],\
                                    initializer=tf.truncated_normal_initializer()) \
                    for i,(car,dim) in enumerate(zip(cardinalitys_T,dimentions_T))]

    learning_rate = tf.placeholder(tf.float32,shape=[])
    X = [tf.placeholder(tf.int32, [batch_size,], name='X_'+str(i)) for i,_ in enumerate(dimentions_X)]
    Xt = [tf.placeholder(tf.int32, [batch_size,seq_len], name='Xt_'+str(i)) for i,_ in enumerate(dimentions_T)]
    y0 = tf.placeholder(tf.float32,[batch_size,1])
    X_continuous = tf.placeholder(tf.float32, [batch_size,seq_len,1], name='X_continuous')
    Weight = tf.placeholder(tf.float32, [batch_size,seq_len], name='Weight')
    y = tf.placeholder(tf.float32, [batch_size,seq_len], name='y')
    IsStart = tf.placeholder(tf.bool, [], name='IsStart')
    Xt1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_Xt,Xt)] + [X_continuous],2)
    X1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embedding_X,X)],1)    
    Xall = [[xt,X1] for xt in tf.unstack(Xt1,axis=1)]
    if actFun == 'tanh':
        actFun = tf.tanh
    else:
        actFun = tf.nn.relu
        
    if cell_type == 'residual':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.ResidualWrapper(\
                                            tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'highway':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.HighwayWrapper(\
                                            tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])        
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'NormLSTM':
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(d,activation=actFun,dropout_keep_prob=keep_prob)\
                                            for _ in range(n_layers)])
        init_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, [batch_size,d], name='initC_'+str(i)),\
                                                          tf.placeholder(tf.float32, [batch_size,d], name='initH_'+str(i))) \
                            for i in range(n_layers)])
        factor = 2
        
    inputs = [y,Weight,X_continuous] + Xt + X + [IsStart,y0,learning_rate,init_state]
    weights_init = tf.Variable(tf.truncated_normal([dX,d*n_layers*factor],
                        stddev=1.0 / np.sqrt(dX)),name='weights_init')
    biases_init = tf.Variable(tf.zeros([d*n_layers*factor]),
                         name='biases_init')
    if cell_type == 'NormLSTM':
        init_state2 = tf.cond(IsStart,\
                            lambda:tuple([tf.contrib.rnn.LSTMStateTuple(*tf.split(tensor_,2,1)) \
                                          for tensor_ in tf.split(tf.matmul(X1,weights_init)+biases_init,n_layers,1)]),\
                            lambda:init_state)
    else:    
        init_state2 = tf.cond(IsStart,\
                            lambda:tuple(tf.split(tf.matmul(X1,weights_init)+biases_init,n_layers,1)),\
                            lambda:init_state)
        
        
    weights_out = tf.Variable(tf.truncated_normal([d],
                        stddev=1.0 / np.sqrt(d)),name='weights_out')
    biases_out = tf.Variable(tf.zeros([1]),
                         name='biases_out')    
    # static_rnn    
    output,state = cell(tf.concat([Xall[0][0],y0,Xall[0][1]],1), init_state2)
    yt_out = tf.nn.relu(tf.einsum('bp,p->b',output,weights_out) + biases_out)
    outputs = [yt_out]
    for rnn_input in Xall[1:]:
        if StopGrad:
            output,state = cell(tf.concat([rnn_input[0],tf.stop_gradient(tf.expand_dims(yt_out,1)),rnn_input[1]],1), state)
        else:
            output,state = cell(tf.concat([rnn_input[0],tf.expand_dims(yt_out,1),rnn_input[1]],1), state)
        yt_out = tf.nn.relu(tf.einsum('bp,p->b',output,weights_out) + biases_out)
        outputs.append(yt_out)
        
    yhat = tf.stack(outputs,1)       
    cost = tf.reduce_mean(Weight*(tf.log((yhat+1)/(y+1)))**2)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),grad_clip)
    if optimizer =='SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    saver = tf.train.Saver({'rnn/'+v.op.name if 'multi_rnn_cell' in v.op.name else v.op.name:v for v in tf.trainable_variables()})    
    
    return inputs,train_op,cost,saver,yhat,state


def hyperSearch(paras):   
    batch_size,seq_len,keep_prob,n_layers,grad_clip,cell_type,downsample,optimizer,actFun = \
	paras[0],paras[1],paras[2],int(paras[3]),paras[4],paras[5],paras[6],paras[7],paras[8]
    # training 
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(batch_size,seq_len,cardinalitys_X,cardinalitys_T,\
                                                    dimentions_X,dimentions_T,dX,d,keep_prob,n_layers,\
                                                    grad_clip,cell_type,optimizer,actFun)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((batch_size,d),dtype=np.float32),\
                                                      np.zeros((batch_size,d),dtype=np.float32))\
                                                        for i in range(n_layers)]) \
                 if cell_type== 'NormLSTM' else \
                 tuple([np.zeros((batch_size,d),dtype=np.float32) for i in range(n_layers)]) 
    for i in range(epoch*100/batch_size):
        for j,X_nps in enumerate(RNN_generator(y_np, weight_np,Con_np,Dis_np,X_np,Count_np,\
                                  batch_size,seq_len,bucketSize,downSample=downsample)):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate,init_state])))
    saver.save(sess,'RNN_fillin_01')
        
    # testing        
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(None,1,cardinalitys_X,cardinalitys_T,\
                                            dimentions_X,dimentions_T,dX,d,keep_prob,n_layers,\
                                            grad_clip,cell_type,optimizer,actFun)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'RNN_fillin_01')
    if cell_type == 'NormLSTM':
        init_tot_list = init_state_update_LSTM(sess,inputs,state,batch_size*10,d,n_layers,\
                      y_np[index],Con_np[index],X_np[index],Count_np[index],\
                      [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat_LSTM(10,sess,inputs,state,yhat,batch_size*10,n_layers,\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
    else:    
        init_tot_list = init_state_update(sess,inputs,state,batch_size*10,d,n_layers,\
                              y_np[index],Con_np[index],X_np[index],Count_np[index],\
                              [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat(10,sess,inputs,state,yhat,batch_size*10,n_layers,\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
    loss = loss_func(weight_np_val[:,np.newaxis],y_val_hat,y_np_val)    
    print "loss:{} ,batch_size:{} ,seq_len:{} ,keep_prob:{} ,n_layers:{} ,grad_clip:{} ,cell_type:{} ,downsample:{} ,optimizer:{} ,actFun:{} \n"\
          .format(loss,batch_size,seq_len,keep_prob,n_layers,grad_clip,cell_type,downsample,optimizer,actFun)
    return 100 if (np.isnan(loss) or np.isinf(loss)) else loss

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

def hyperSearch_epoch(paras,fixedPara,learningRate,index,SavePath,check_points=[15,20,25,30]):
    downsample = paras['downsample']
    RNN_paras = paras['model_para'].copy()
    check_points = [i*100/RNN_paras['batch_size'] for i in check_points]
    print RNN_paras
    print '\n'
    
    RNN_paras.update(fixedPara)
    RNN_paras_oneStep = RNN_paras.copy()
    RNN_paras_oneStep['batch_size'] = None
    RNN_paras_oneStep['seq_len'] = 1
    
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    model_name = '-'
    model_name = model_name.join([name+':'+str(value) for name,value in [i for i in paras['model_para'].iteritems()]+[('downsample',downsample)]])
    best_name = ''
    best_loss = 100
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((RNN_paras['batch_size'],RNN_paras['d']),\
                                                                   dtype=np.float32),\
                                                      np.zeros((RNN_paras['batch_size'],RNN_paras['d']),\
                                                                   dtype=np.float32))\
                                                    for i in range(RNN_paras['n_layers'])]) \
                 if RNN_paras['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((RNN_paras['batch_size'],RNN_paras['d']),dtype=np.float32) \
                        for i in range(RNN_paras['n_layers'])]) 
    for i in range(1,max(check_points)+1):
        for j,X_nps in enumerate(RNN_generator(y_np, weight_np,Con_np,Dis_np,X_np,Count_np,\
                                  RNN_paras['batch_size'],RNN_paras['seq_len'],10000,downSample=downsample)):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate,init_state])))
        
        if i in check_points:
            saver.save(sess,'RNN_temp_model')
            inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras_oneStep)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,'RNN_temp_model')

            if RNN_paras['cell_type'] == 'NormLSTM':
                init_tot_list = init_state_update_LSTM(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                              y_np[index],Con_np[index],X_np[index],Count_np[index],\
                              [dis[index] for dis in Dis_np])
                y_val_hat = RNN_forecast_Repeat_LSTM(10,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                                np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                                Con_np_val,X_np_val,Dis_np_val,init_tot_list)
            else:    
                init_tot_list = init_state_update(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                                      y_np[index],Con_np[index],X_np[index],Count_np[index],\
                                      [dis[index] for dis in Dis_np])
                y_val_hat = RNN_forecast_Repeat(10,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                                np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                                Con_np_val,X_np_val,Dis_np_val,init_tot_list)
            loss = loss_func(weight_np_val[:,np.newaxis],y_val_hat,y_np_val)                
            if loss < best_loss:
                if best_name != '':
                    purge(SavePath, best_name+'*')
                fullPath = SavePath+'/'+model_name+'-epoch:'+str(i)
                saver.save(sess,fullPath)
                best_name = model_name+'-epoch:'+str(i)
                best_loss = loss

            inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,'RNN_temp_model')

            print "loss:{} ,epoch:{} \n".format(loss,i)

    return best_name


def RNN_Train_Forecast(paras,fixedPara,learningRate,epoch,SavePath,repeat,\
                       y_np, weight_np,Con_np,Dis_np,X_np,Count_np,index,\
                       Con_np_val,X_np_val,Dis_np_val,d):
    downsample = paras['downsample']
    RNN_paras = paras['model_para'].copy()    
    RNN_paras.update(fixedPara)
   
    # Training
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    model_name = '-'
    model_name = model_name.join([name+':'+str(value) for name,value in [i for i in paras['model_para'].iteritems()]+[('downsample',downsample)]])
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((RNN_paras['batch_size'],RNN_paras['d']),\
                                                                   dtype=np.float32),\
                                                      np.zeros((RNN_paras['batch_size'],RNN_paras['d']),\
                                                                   dtype=np.float32))\
                                                    for i in range(RNN_paras['n_layers'])]) \
                 if RNN_paras['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((RNN_paras['batch_size'],RNN_paras['d']),dtype=np.float32) \
                        for i in range(RNN_paras['n_layers'])]) 
    for i in range(epoch*100/RNN_paras['batch_size']):
        for j,X_nps in enumerate(RNN_generator(y_np, weight_np,Con_np,Dis_np,X_np,Count_np,\
                                  RNN_paras['batch_size'],RNN_paras['seq_len'],10000,downSample=downsample)):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate,init_state])))
        
    saver.save(sess,SavePath+'/'+model_name)
    
    # Forecast
    RNN_paras['batch_size'] = None
    RNN_paras['seq_len'] = 1        
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,SavePath+'/'+model_name)
    
    if RNN_paras['cell_type'] == 'NormLSTM':
        init_tot_list = init_state_update_LSTM(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                      y_np[index],Con_np[index],X_np[index],Count_np[index],\
                      [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat_LSTM(repeat,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
    else:    
        init_tot_list = init_state_update(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                              y_np[index],Con_np[index],X_np[index],Count_np[index],\
                              [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat(repeat,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
        
    return y_val_hat    

def RNN_Forecast(paras,fixedPara,learningRate,epoch,SavePath,repeat):
    downsample = paras['downsample']
    RNN_paras = paras['model_para'].copy()    
    RNN_paras.update(fixedPara)
      
    model_name = '-'
    model_name = model_name.join([name+':'+str(value) for name,value in [i for i in paras['model_para'].iteritems()]+[('downsample',downsample)]])
          
    # Forecast
    RNN_paras['batch_size'] = None
    RNN_paras['seq_len'] = 1        
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,SavePath+'/'+model_name)
    
    if RNN_paras['cell_type'] == 'NormLSTM':
        init_tot_list = init_state_update_LSTM(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                      y_np[index],Con_np[index],X_np[index],Count_np[index],\
                      [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat_LSTM(repeat,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
    else:    
        init_tot_list = init_state_update(sess,inputs,state,1000,d,RNN_paras['n_layers'],\
                              y_np[index],Con_np[index],X_np[index],Count_np[index],\
                              [dis[index] for dis in Dis_np])
        y_val_hat = RNN_forecast_Repeat(repeat,sess,inputs,state,yhat,1000,RNN_paras['n_layers'],\
                                        np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                        Con_np_val,X_np_val,Dis_np_val,init_tot_list)
        
    return y_val_hat    


def subset(test,train):
    train['S_I'] = train.item_nbr + train.store_nbr/100.0
    test['S_I'] = test.item_nbr + test.store_nbr/100.0
    train_SI = set(train.S_I)
    test_SI = set(test.S_I)
    SI_intersection = train_SI & test_SI
    index_train_SI = train.S_I.isin(SI_intersection)
    index_test_SI = test.S_I.isin(SI_intersection)
    return test[index_test_SI].drop('S_I',1).reset_index(drop=True),train[index_train_SI].drop('S_I',1).reset_index(drop=True)

def createDataMainSecondStage(isTrain):
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
    if not isTrain:
        test = pd.read_csv('test.csv',parse_dates=['date'],dtype=types, infer_datetime_format=True)
        test = test.fillna(2,axis=1)
        test.onpromotion = test.onpromotion.astype(np.int8)
    with open('item_dict_train.pickle' if isTrain else 'item_dict_final.pickle', "rb") as input_file:
        item_dict = cPickle.load(input_file)
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
    
    def CreateTestData(data):
        SI_train_sales = data.groupby(['store_nbr','item_nbr'])[['date','id','onpromotion']].\
                        agg(lambda x: tuple(x)).reset_index()
        storeTime = data.groupby(['store_nbr'])['date'].agg([np.min,np.max]).reset_index()
        dfs = [items2,stores2,storeTime]
        keys = ['item_nbr','store_nbr','store_nbr']
        SI_train = mergeFillCastsss(SI_train_sales,dfs,keys)
        SI_train['item_nbr'] = SI_train.item_nbr.map(iter_mapping)
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
                         'date',
                         'id',                         
                         'onpromotion',
                         'amin',
                         'amax']]
    
    if isTrain:
        val = train[train.date >= '2017-07-30']
        train = train[(train.date <= '2017-07-30') & (train.date >= '2017-05-27')]
        valRNN,trainRNN = subset(val,train)
        return CreateData(trainRNN).reset_index(drop=True),CreateData(valRNN).reset_index(drop=True)
    else:
        train = train[train.date >= '2017-06-12']
        testRNN,trainRNN = subset(test,train)
        return CreateData(trainRNN).reset_index(drop=True),CreateTestData(testRNN).reset_index(drop=True)
    
    
def RNN_generator_static(y_np, weight_np,Con_np,Dis_list,X_np,\
                  batchSize,seqSize,startDate,downSample=1,iterAll=False,permutate=True):
    # time dimention needs to have T+1 as y needs a lag!!
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)

    n,d = y_np.shape
    Index_perm = np.random.permutation(n) if permutate else np.arange(n)
    for i in range(0,n if iterAll else n-batchSize,batchSize):
        Index_X = Index_perm[i:i+batchSize]
        X_list_ = list(X_np[Index_X].T)
        for t_ in range(startDate,d-1,seqSize):
            Index_Y = slice(t_,t_+seqSize)
            Index_Y_1 = slice(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield [y_,weight_np[Index_X,np.newaxis]*weight,\
                     np.stack([Con_np[Index_X,Index_Y_1],y_np[Index_X,Index_Y]],-1)]\
                     + [dis[Index_X,Index_Y_1] for dis in Dis_list]\
                     + X_list_ + [t_==startDate] 
                    
                    
def RNN_generator_dynamic(y_np, weight_np,Con_np,Dis_list,X_np,\
                  batchSize,seqSize,startDate,downSample=1,iterAll=False,permutate=True):
    # time dimention needs to have T+1 as y needs a lag!!
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)

    n,d = y_np.shape
    Index_perm = np.random.permutation(n) if permutate else np.arange(n)
    for i in range(0,n if iterAll else n-batchSize,batchSize):
        Index_X = Index_perm[i:i+batchSize]
        X_list_ = list(X_np[Index_X].T)
        for t_ in range(startDate,d-1,seqSize):
            Index_Y_1 = slice(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield [y_,weight_np[Index_X,np.newaxis]*weight,\
                     Con_np[Index_X,Index_Y_1,np.newaxis]]\
                     + [dis[Index_X,Index_Y_1] for dis in Dis_list]\
                     + X_list_ + [t_==startDate,y_np[Index_X,t_:t_+1]]                     
              

def hyperSearch2(paras):   
    # paras[0] is one of the modelName
    model_para = model_para_list[paras[0]]['model_para'].copy()
    model_name = 'grad_clip:{}-cell_type:{}-optimizer:{}-actFun:{}-seq_len:{}-n_layers:{}-keep_prob:{}-batch_size:{}-downsample:{}'\
                 .format(model_para['grad_clip'],model_para['cell_type'],model_para['optimizer'],model_para['actFun'],model_para['seq_len'],\
                         model_para['n_layers'],model_para['keep_prob'],model_para['batch_size'],model_para_list[paras[0]]['downsample']) 
    model_para['batch_size'] = paras[1]
    model_para['seq_len'] = paras[2]
    model_para['grad_clip'] = paras[3]
    model_para['optimizer'] = paras[4] 
    model_para.update(fixedPara)
    trainModeParas = paras[5]
    downsample = paras[6]
    startDate = 0 # set to zero as there is only 64 days
    if trainModeParas['trainMode'] == 'dynamic':
        model_para['StopGrad'] = trainModeParas['StopGrad']
        inputs,train_op,cost,saver,yhat,state = createGraphRNN_dynamic(**model_para)
    else:
        inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para)
        
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,path+model_name)
    
    # Training
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32),\
                                                      np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32))\
                                                    for i in range(model_para['n_layers'])]) \
                 if model_para['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32) \
                        for i in range(model_para['n_layers'])]) 
        
    generator_ = RNN_generator_dynamic if trainModeParas['trainMode'] == 'dynamic' else RNN_generator_static
    for i in range(epoch):
        for X_nps in generator_(y_np, weight_np,Con_np,Dis_np,X_np,\
                                paras[1],paras[2],startDate=startDate,downSample=downsample):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate2,init_state])))
    
    saver.save(sess,'RNN_SS_temp')
        
    # Testing        
    model_para2 = model_para.copy()
    model_para2['batch_size'] = None
    model_para2['seq_len'] = 16
    if trainModeParas['trainMode'] == 'dynamic':
        del model_para2['StopGrad']
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para2)   
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'RNN_SS_temp')
    
    ## get init_state
    init_tot_list = []
    for X_nps in RNN_generator_static(y_np, weight_np,Con_np,Dis_np,X_np,\
                                      100,16,startDate=startDate,downSample=1,iterAll=True,permutate=False):
        if X_nps[-1]:
            init_tot_list.append(init_state)
        init_state = sess.run(state,dict(zip(inputs,X_nps+[learningRate2,init_state])))
    init_tot_list.append(init_state)
    init_tot_list = init_tot_list[1:]
    
    # prediction    
    model_para2['StopGrad'] = False # does not matter for prediction
    inputs,train_op,cost,saver,yhat,state = createGraphRNN_dynamic(**model_para2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'RNN_SS_temp')
    
    loss = 0
    w_ = 0         
    for i,X_nps in enumerate(RNN_generator_dynamic(y_np_val, weight_np_val,Con_np_val,Dis_np_val,X_np_val,\
                                                 100,16,startDate=0,downSample=1,iterAll=True,permutate=False)): 
        X_nps[-2] = False
        loss = loss + sess.run(cost,dict(zip(inputs,X_nps+[learningRate2,init_tot_list[i]])))*X_nps[0].shape[0]*16
        w_ = w_ + np.sum(X_nps[1])
    loss = np.sqrt(loss/w_)
    if trainModeParas['trainMode'] == 'dynamic':
        print "loss:{} , model:{}, trainMode:{}, batch_size:{} ,seq_len:{} ,grad_clip:{} ,downsample:{} ,optimizer:{}  \n"\
              .format(loss,paras[0],model_para['StopGrad'],model_para['batch_size'],\
                      model_para['seq_len'],model_para['grad_clip'],downsample,model_para['optimizer'])
    else:
        print "loss:{} , model:{}, trainMode:{}, batch_size:{} ,seq_len:{} ,grad_clip:{} ,downsample:{} ,optimizer:{}  \n"\
              .format(loss,paras[0],trainModeParas['trainMode'],model_para['batch_size'],\
                      model_para['seq_len'],model_para['grad_clip'],downsample,model_para['optimizer'])
    return 100 if (np.isnan(loss) or np.isinf(loss)) else loss        



def RNN_Train_Forecast_SS(paras):   
    # paras[0] is one of the modelName
    model_para = model_para_list[paras[0]]['model_para'].copy()
    model_name = 'grad_clip:{}-cell_type:{}-optimizer:{}-actFun:{}-seq_len:{}-n_layers:{}-keep_prob:{}-batch_size:{}-downsample:{}'\
                 .format(model_para['grad_clip'],model_para['cell_type'],model_para['optimizer'],model_para['actFun'],model_para['seq_len'],\
                         model_para['n_layers'],model_para['keep_prob'],model_para['batch_size'],model_para_list[paras[0]]['downsample']) 
    model_para['batch_size'] = paras[1]
    model_para['seq_len'] = paras[2]
    model_para['grad_clip'] = paras[3]
    model_para['optimizer'] = paras[4] 
    model_para.update(fixedPara)
    downsample = paras[5]
    startDate = 0 # set to zero as there is only 64 days
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para)
        
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,SavePath+model_name)
    
    # Training
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32),\
                                                      np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32))\
                                                    for i in range(model_para['n_layers'])]) \
                 if model_para['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32) \
                        for i in range(model_para['n_layers'])]) 

    for i in range(epoch):
        for X_nps in RNN_generator_static(y_np, weight_np,Con_np,Dis_np,X_np,\
                                paras[1],paras[2],startDate=startDate,downSample=downsample):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate2,init_state])))
            
    model_name_new = "Model:{}-batch_size:{}-seq_len:{}-grad_clip:{}-optimizer:{}-downsample:{}"\
                      .format(paras[0],model_para['batch_size'],\
                      model_para['seq_len'],model_para['grad_clip'],model_para['optimizer'],downsample)
    saver.save(sess,SavePath_SS+model_name_new)
    
    # Testing        
    model_para2 = model_para.copy()
    model_para2['batch_size'] = None
    model_para2['seq_len'] = 16

    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para2)   
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,SavePath_SS+model_name_new)
    
    ## get init_state
    init_tot_list = []
    for X_nps in RNN_generator_static(y_np, weight_np,Con_np,Dis_np,X_np,\
                                      100,16,startDate=startDate,downSample=1,iterAll=True,permutate=False):
        if X_nps[-1]:
            init_tot_list.append(init_state)
        init_state = sess.run(state,dict(zip(inputs,X_nps+[learningRate2,init_state])))
    init_tot_list.append(init_state)
    init_tot_list = init_tot_list[1:]
      
    # prediction    
    model_para2['StopGrad'] = False # does not matter for prediction
    inputs,train_op,cost,saver,yhat,state = createGraphRNN_dynamic(**model_para2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,SavePath_SS+model_name_new)

    Yhat = []      
    for i,X_nps in enumerate(RNN_generator_dynamic(y_np_val, np.ones(y_np_val.shape[0]),Con_np_val,Dis_np_val,X_np_val,\
                                                 100,16,startDate=0,downSample=1,iterAll=True,permutate=False)): 
        X_nps[-2] = False
        Yhat.append(np.mean(np.stack([sess.run(yhat,dict(zip(inputs,X_nps+[learningRate2,init_tot_list[i]])))\
                                       for _ in range(repeat)],2),2))
    return np.concatenate(Yhat)        
   