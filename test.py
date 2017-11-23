# bucket for different length
# time series vs non series model for different item-store pair
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
        weight[y_==0] = downSamplea
        yield [y_,weight_np[Index_X]*weight,np.reshape(Con_np[Index_X,Index_Y],(batchSize,seqSize,1))]\
                + [dis[Index_X,Index_Y] for dis in Dis_list] \
                + list(X[Index_X].T)

def RNN_generator(y_np, weight_np,Con_np,Dis_list,X_np,Count_np,\
                  batchSize,seqSize,bucketSize,shuffle=True,downSample=1):
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)
    # data needs to by sorted by count_np !!
    n = y_np.shape[0]
    bucketNum = n//bucketSize
    weight_adj = n%bucketSize*1.0/bucketSize
    for bucket in np.random.permutation(bucketNum+1):
        Index_X = np.random.randint(bucket*bucketSize,min(n,(bucket+1)*bucketSize),(batchSize,1))
        X_list_ = list(X[Index_X].T)
        adj_ = weight_adj if bucket==bucketNum else 1
        for t_ in range(0,np.min(Count_np[Index_X])-seqSize,seqSize):
            Index_Y = np.arange(t_,t_+seqSize)
            Index_Y_1 = np.arange(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield t_==0, [y_,weight_np[Index_X]*weight*adj_,\
                         np.stack([Con_np[Index_X,Index_Y_1],y_np[Index_X,Index_Y]],-1)]\
                         + [dis[Index_X,Index_Y_1] for dis in Dis_list]
                         + X_list_            
