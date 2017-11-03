def RNN_generator(data,batchSize,seqSize,df,key,discreteList,shuffle=True,downSample=1):
    # return bool (if init should be reset) and a list [y,weight,Xcontinue] + [Xdiscrete] + [X]
    # finetune by use store-item pair in testset only
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    n = data.shape[0]

    for from_ in range(0,n,batchSize):
        X = list(data.loc[from_:from_+batchSize,'store_nbr':'cluster'].values.astype(np.int32).T)
        weight = np.ones((batch_size,1),dtype=np.float32)
        weight[data.loc[from_:from_+batchSize,'perishable']==1] = 1.25
        for j,(y,w,Xdiscrete,Xcontinue) in enumerate(timeGenerator(\
                           data.loc[from_:from_+batchSize,'date':'amax'],seqSize,downSample,df,key,discreteList)):
            yield j==0, [y,weight*w,Xcontinue] + Xdiscrete + X
            
def timeGenerator(data,seqSize,downSample,df,key,discreteList):
    # df is time related variables like holiday and seasonality
    # returns y, weight, X_discrete, X_continue
    n = data.shape[0]
    data['curr'] = data.amin 
    sparse = pd.concat([data.apply(lambda x: pd.Series(x.date,name='date'),axis=1)\
                                    .stack().reset_index().drop(['level_1'],1),\
                        data.apply(lambda x: pd.Series(x.unit_sales),axis=1)\
                                    .stack().reset_index().drop(['level_0','level_1'],1)],1)
    sparse.columns = ['level','date','sales']
    while np.all(data.curr + pd.DateOffset(seqSize) <= data['amax']):
        dense = data.apply(lambda x:pd.Series(pd.date_range(x.curr,periods=seqSize+1)),axis=1)\
                            .stack().reset_index().drop(['level_1'],1)
        dense.columns = ['level','date']
        dense = pd.merge(pd.merge(dense, df, how='left', on=key),
                        sparse,how='left',on=['level','date']).fillna(0)
        dense_continue = dense[['dcoilwtico','sales']].values.astype(np.float32)\
                                  .reshape((n,seqSize+1,2))[:,:seqSize,:]
        dense_discrete = list(np.moveaxis(dense[discreteList].values.astype(np.int32)\
                                                .reshape((n,seqSize+1,len(discreteList)))[:,:seqSize,:]\
                                          ,2,0))
        y = dense['sales'].values.astype(np.float32).reshape((n,seqSize+1))[:,1:]
        weight = np.ones_like(y,dtype=np.float32)
        weight[y==0] = downSample
        yield y, weight, dense_discrete, dense_continue
        data.curr = data.curr + pd.DateOffset(seqSize)            
