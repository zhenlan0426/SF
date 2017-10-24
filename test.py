## TODO: clip y to zero in training data
cardinalitys = []
dimentions = [] # of resulting tensor
d = sum(dimentions)

lambda1,lambda2,lambda3 = 1,1,1

tf.reset_default_graph()

embeddings = [tf.get_variable("embedding_"+str(i), [car, dim],initializer=tf.truncated_normal_initializer()) \
                for i,(car,dim) in enumerate(zip(cardinalitys,dimentions))]

X_discretes = [tf.placeholder(tf.int32, [batch_size,], name='X_discrete_'+str(i)) for i,_ in enumerate(dimentions)]
X_continuous = tf.placeholder(tf.float32, [batch_size,], name='X_continuous')
Weight = tf.placeholder(tf.float32, [batch_size,], name='Weight')
y = tf.placeholder(tf.float32, [batch_size,], name='y')
X1 = tf.concat([tf.nn.embedding_lookup(emb,x) for emb,x in zip(embeddings,X_discretes)] + [X_continuous],1)

weights1 = tf.Variable(tf.truncated_normal([d,d],
                        stddev=1.0 / np.sqrt(d)),name='weights1')
biases1 = tf.Variable(tf.zeros([d]),
                     name='biases1')
X2 = tf.nn.relu(tf.matmul(X1, weights1) + biases1)

weights2 = tf.Variable(tf.truncated_normal([d,1],
                        stddev=1.0 / np.sqrt(d)),name='weights2')
b0 = np.mean(***,dtype=np.float32)
biases2 = tf.get_variable("biases2",
    initializer=tf.constant(b0))
yhat = tf.squeeze(tf.nn.relu(tf.matmul(X2, weights2) + biases2)) # as target is always positive

cost = tf.reduce_mean(Weight*(tf.log((yhat+1)/(y+1)))**2)
regularizer = 
augment_cost = cost + 
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
