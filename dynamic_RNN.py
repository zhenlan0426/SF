# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf


'''For data subsetting, time dimention needs to have T+1 as y needs a lag!!'''

def RNN_generator_static(y_np, weight_np,Con_np,Dis_list,X_np,\
                  batchSize,seqSize,startDate,downSample=1,iterAll=False):
    # time dimention needs to have T+1 as y needs a lag!!
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)

    n,d = y_np.shape
    Index_perm = np.random.permutation(n)
    for i in range(0,n if iterAll else n-batchSize,batchSize):
        Index_X = Index_perm[i:i+batchSize]
        X_list_ = list(X_np[Index_X].T)
        for t_ in range(startDate,d,seqSize):
            Index_Y = np.arange(t_,t_+seqSize)
            Index_Y_1 = np.arange(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield [y_,weight_np[Index_X]*weight,\
                     np.stack([Con_np[Index_X,Index_Y_1],y_np[Index_X,Index_Y]],-1)]\
                     + [dis[Index_X,Index_Y_1] for dis in Dis_list]\
                     + X_list_ + [t_==startDate] 
                     
              
def RNN_generator_dynamic(y_np, weight_np,Con_np,Dis_list,X_np,\
                  batchSize,seqSize,startDate,downSample=1,iterAll=False):
    # time dimention needs to have T+1 as y needs a lag!!
    # return [y (B,T),weight (B,T),Xcontinue of shape (B,T,2)] + [Xdiscrete] of shape (B,T) + [X] of shape (B,)

    n,d = y_np.shape
    Index_perm = np.random.permutation(n)
    for i in range(0,n if iterAll else n-batchSize,batchSize):
        Index_X = Index_perm[i:i+batchSize]
        X_list_ = list(X_np[Index_X].T)
        for t_ in range(startDate,d,seqSize):
            Index_Y = np.arange(t_,t_+seqSize)
            Index_Y_1 = np.arange(t_+1,t_+seqSize+1)
            y_ = y_np[Index_X,Index_Y_1]
            weight = np.ones_like(y_,dtype=np.float32)
            weight[y_==0] = downSample            
            yield [y_,weight_np[Index_X]*weight,\
                     Con_np[Index_X,Index_Y_1,np.newaxis]\
                     + [dis[Index_X,Index_Y_1] for dis in Dis_list]\
                     + X_list_ + [t_==startDate,y_np[Index_X,t:t+1]] 

                     
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
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'highway':
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.HighwayWrapper(tf.contrib.rnn.GRUCell(d,actFun)),output_keep_prob=keep_prob)
        init_state = tuple([tf.placeholder(tf.float32, [batch_size,d], name='initState_'+str(i)) for i in range(n_layers)])
        factor = 1
    elif cell_type == 'NormLSTM':
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(d,activation=actFun,dropout_keep_prob=keep_prob)
        init_state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, [batch_size,d], name='initC_'+str(i)),\
                                                          tf.placeholder(tf.float32, [batch_size,d], name='initH_'+str(i))) \
                            for i in range(n_layers)])
        factor = 2
        
    inputs = [y,Weight,X_continuous] + Xt + X + [IsStart,y0,learning_rate,init_state]
    cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
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
    saver = tf.train.Saver()     
    
    return inputs,train_op,cost,saver,yhat,state


'''setup '''
Fixed = 'keep_prob,n_layers,cell_type,actFun,cardinalitys_X,cardinalitys_T,dimentions_X,dimentions_T,dX,d'
Tuning = 'batch_size,seq_len,grad_clip,downsample,optimizer'
model_paras = {'nameSavedOnDisk':{'keep_prob':0.8,'n_layers':3,'cell_type':'NormLSTM','actFun':'tanh',...}}
input_ = [y_np_FT, weight_np_FT,Con_np_FT,Dis_list_FT,X_np_FT] # fine-tuning dataset
input2_ = [y_np_val, weight_np_val,Con_np_val,Dis_list_val,X_np_val] # test dataset
learningRate2 = learningRate * 0.5
'No longer needs index, as input and input2 have the same n. JUST MAKE SURE the first dimention match!!'
'''setup '''


def hyperSearch2(paras):   
    # paras[0] is one of the modelName
    model_para = model_paras[paras[0]]
    model_para['batch_size'] = paras[1]
    model_para['seq_len'] = paras[2]
    model_para['grad_clip'] = paras[3]
    model_para['optimizer'] = paras[4]
    trainMode = paras[5]
    downsample = paras[6]
    startDate = paras[7]
    if trainMode == 'dynamic':
        model_para['StopGrad'] = paras[8]
        inputs,train_op,cost,saver,yhat,state = createGraphRNN_dynamic(**model_para)
    else:
        inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para)
        
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,paras[0])
    
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32),\
                                                      np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32))\
                                                    for i in range(model_para['n_layers'])]) \
                 if model_para['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32) \
                        for i in range(model_para['n_layers'])]) 
        
    generator_ = RNN_generator_dynamic if trainMode == 'dynamic' else RNN_generator_static
    for i in range(epoch):
        for X_nps in generator_(y_np_FT, weight_np_FT,Con_np_FT,Dis_list_FT,X_np_FT,\
                                paras[1],paras[2],startDate=startDate,downSample=downsample):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate2,init_state])))
           
    saver.save(sess,paras[0]+'+FT')
        
    # testing        
    model_para2 = model_para.copy()
    model_para2['batch_size'] = None
    model_para['seq_len'] = 16
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**model_para2)   
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,paras[0]+'+FT')
    
    # get init_state
    init_tot_list = []
    for X_nps in RNN_generator_static(y_np_FT, weight_np_FT,Con_np_FT,Dis_list_FT,X_np_FT,\
                                      100,16,startDate=startDate,downSample=downsample,iterAll=True):
        if X_nps[-1]:
            init_tot_list.append(init_state)
        init_state = sess.run(state,dict(zip(inputs,X_nps+[learningRate2,init_state])))
    init_tot_list.append(init_state)
    init_tot_list = init_tot_list[1:]
    
    # prediction    
    inputs,train_op,cost,saver,yhat,state = createGraphRNN_dynamic(**model_para2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,paras[0]+'+FT')
    
    loss = 0
    w_ = 0
    for i,X_nps in enumerate(RNN_generator_dynamic(y_np_val, weight_np_val,Con_np_val,Dis_list_val,X_np_val,\
                                                 100,16,startDate=0,downSample=1,iterAll=True)): 
        loss = loss + sess.run(cost,dict(zip(inputs,X_nps+[learningRate2,init_tot_list[i]])))*X_nps[0].shape[0]
        w_ = w_ + np.sum(X_nps[2])
    loss = np.sqrt(loss/w_)
    print "loss:{} ,batch_size:{} ,seq_len:{} ,keep_prob:{} ,n_layers:{} ,grad_clip:{} ,cell_type:{} ,downsample:{} ,optimizer:{} ,actFun:{} \n"\
          .format(loss,batch_size,seq_len,keep_prob,n_layers,grad_clip,cell_type,downsample,optimizer,actFun)
    return 100 if (np.isnan(loss) or np.isinf(loss)) else loss                     

