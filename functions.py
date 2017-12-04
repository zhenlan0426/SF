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
    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(d),keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
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
    
       
def loss_func(Weight,yhat,y):
    return np.sqrt(np.sum(Weight*(np.log((yhat+1)/(y+1)))**2)/np.sum(Weight)/16)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














