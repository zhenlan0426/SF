# TODO: SI start at Store start date vs SI start date
# fixedPara = [(cardinalitys_X,v1),cardinalitys_T,dimentions_X,dimentions_T,dX,d] a list of tuple
# fixedPara = [(cardinalitys_X,v1),cardinalitys_T,dimentions_X,dimentions_T,dX,d] a list of tuple
import os
def hyperSearch_epoch(y_np, weight_np,Con_np,Dis_np,X_np,Count_np,\
                      y_np_val, weight_np_val,Con_np_val,Dis_np_val,X_np_val,Count_np_val,\
                      paras,fixedPara,learningRate,downsample,index,SavePath,check_points=[15,20,25,30]):
    # paras should be a list of tuple [(paraName,paraValue)...]
    RNN_paras = dict(paras+fixedPara)
    RNN_paras_oneStep = RNN_paras.copy()
    RNN_paras_oneStep['batch_size'] = None
    RNN_paras_oneStep['seq_len'] = 1
    
    inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    model_name = '-'
    model_name.join([name+':'+str(value) for name,value in paras+[('downsample',downsample)]])
    best_name = ''
    best_loss = 100
    init_state = tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros((model_para['batch_size'],model_para['d']),\
                                                                   dtype=np.float32),\
                                                      np.zeros((model_para['batch_size'],model_para['d']),\
                                                                   dtype=np.float32))\
                                                    for i in range(model_para['n_layers'])]) \
                 if model_para['cell_type'] == 'NormLSTM' else \
                 tuple([np.zeros((model_para['batch_size'],model_para['d']),dtype=np.float32) \
                        for i in range(model_para['n_layers'])]) 
    for i in range(1,max(check_points)+1):
        for j,X_nps in enumerate(RNN_generator(y_np, weight_np,Con_np,Dis_np,X_np,Count_np,\
                                  model_para['batch_size'],model_para['seq_len'],10000,downSample=downsample)):
            _,init_state = sess.run([train_op,state],\
                                 dict(zip(inputs,X_nps+[learningRate,init_state])))
        
        if i in check_points:
            saver.save(sess,'RNN_temp_model')
            inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras_oneStep)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,'RNN_temp_model')

            if model_para['cell_type'] == 'NormLSTM':
                init_tot_list = init_state_update_LSTM(sess,inputs,state,1000,d,model_para['n_layers'],\
                              y_np[index],Con_np[index],X_np[index],Count_np[index],\
                              [dis[index] for dis in Dis_np])
                y_val_hat = RNN_forecast_Repeat_LSTM(10,sess,inputs,state,yhat,1000,model_para['n_layers'],\
                                                np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                                Con_np_val,X_np_val,Dis_np_val,init_tot_list)
            else:    
                init_tot_list = init_state_update(sess,inputs,state,1000,d,model_para['n_layers'],\
                                      y_np[index],Con_np[index],X_np[index],Count_np[index],\
                                      [dis[index] for dis in Dis_np])
                y_val_hat = RNN_forecast_Repeat(10,sess,inputs,state,yhat,1000,model_para['n_layers'],\
                                                np.expand_dims(y_np[index,Count_np[index]-1],-1),\
                                                Con_np_val,X_np_val,Dis_np_val,init_tot_list)
            loss = loss_func(weight_np_val[:,np.newaxis],y_val_hat,y_np_val)                
            if loss < best_loss:
                if best_name != '':
                    os.remove(best_name)
                fullPath = SavePath+'/'+model_name+'-epoch:'+str(i)
                saver.save(sess,fullPath)
                best_name = fullPath
                best_loss = loss

            inputs,train_op,cost,saver,yhat,state = createGraphRNN2(**RNN_paras)
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,'RNN_temp_model')

            print "loss:{} ,epoch:{} ,batch_size:{} ,seq_len:{} ,keep_prob:{} ,n_layers:{} ,grad_clip:{} ,cell_type:{} ,downsample:{} ,optimizer:{} ,actFun:{} \n"\
              .format(loss,i,batch_size,seq_len,keep_prob,n_layers,grad_clip,cell_type,downsample,optimizer,actFun)

    return best_name
