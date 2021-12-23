import numpy as np
import tensorflow as tf

def Unit(x):
    return (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))

def RMSE(x,y):
    return np.sqrt( ( ( x.flatten() - y.flatten() )**2 ).mean() )

def rmse_error(y, y_hat):
    return tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)))

def CC(x,y):
    return np.corrcoef(x.flatten(), y.flatten())[0,1]

def extreme(Prediction, Label, ratio):
    mat = np.concatenate([Prediction.reshape(-1,1), Label.reshape(-1,1)], axis=1)
    mat1 = mat[mat[:,1].argsort()][::-1]
    position = int(ratio * len(mat))
    threshold_value = mat1[position][1]
    Mat = []
    for i in range(len(mat)):
        if mat[i,1] >= threshold_value:
            Mat.append(mat[i])
    consider = np.array(Mat)        
    new_pred, new_label = consider[:,0], consider[:,1]        
    return RMSE(new_pred, new_label)

def distribution(vector, label, bins):
    Max = np.max(label)
    Min = np.min(label)
    interval = np.linspace(Min, Max, bins + 1) 
    revised_interval = np.array([  (interval[i]+interval[i+1])/2 for i in range(len(interval)-1) ])
    
    dist_vector = np.array([ (vector < interval[i+1]).sum() - (vector < interval[i]).sum() for i in range(bins) ])
    dist_label  = np.array([ (label  < interval[i+1]).sum() - (label  < interval[i]).sum() for i in range(bins) ])
    
    # 가로 X 세로 = (Max - Min)/bins * dist*bins/((Max - Min)*len(dist))
    pdf_vector  = dist_vector*bins/((Max - Min)*len(vector))
    pdf_label   = dist_label *bins/((Max - Min)*len(label))
       
    common_area = np.sum(np.minimum((Max - Min)/bins *pdf_vector, (Max - Min)/bins *pdf_label))
    return revised_interval, pdf_vector, pdf_label, common_area

def split_MLP(data, ratio):
    x_data     = Unit(data[:-1])
    y_data     = Unit(data[1:, [-1]])
    ntrain = int(ratio*len(data))
    trainX, trainY, testX, testY = x_data[:ntrain], y_data[:ntrain], x_data[ntrain:], y_data[ntrain:]
    return trainX, trainY, testX, testY

def split_MLP2(data, ratio, factor):
    def operation(x,factor): # From ROR (rate of return) to widening ; y = x**(factor)
        return x**factor
    operated = np.array([ operation(data[:,i],factor) for i in range(data.shape[1]) ]).T
    x_data     = Unit(operated[:-1])
    y_data     = Unit(operated[1:, [-1]])
    ntrain = int(ratio*len(data))
    trainX, trainY, testX, testY = x_data[:ntrain], y_data[:ntrain], x_data[ntrain:], y_data[ntrain:]
    return trainX, trainY, testX, testY

def split_LSTM(data, depth, ratio):
    x_data     = Unit(data[:-1])
    y_data     = Unit(data[depth:, -1])

    x_setting = []
    for i in range(depth-1):
        x_setting.append(x_data[i:-(depth-i-1)])
    x_setting.append(x_data[depth-1:])
    y_setting = y_data

    ntrain = int(ratio*len(x_setting[0]))
    trainX = [ x_setting[i][:ntrain] for i in range(depth) ]
    trainY = y_setting[:ntrain].reshape(-1,1)
    testX  = [ x_setting[i][ntrain:] for i in range(depth) ]
    testY  = y_setting[ntrain:].reshape(-1,1)
    return trainX, trainY, testX, testY

def split_LSTM2(data, depth, ratio, factor):
    def operation(x,factor): # From ROR (rate of return) to widening ; y = x**(factor)
        return x**factor
    operated = np.array([ operation(data[:,i],factor) for i in range(data.shape[1]) ]).T
    x_data     = Unit(operated[:-1])
    y_data     = Unit(operated[depth:, -1])

    x_setting = []
    for i in range(depth-1):
        x_setting.append(x_data[i:-(depth-i-1)])
    x_setting.append(x_data[depth-1:])
    y_setting = y_data

    ntrain = int(ratio*len(x_setting[0]))
    trainX = [ x_setting[i][:ntrain] for i in range(depth) ]
    trainY = y_setting[:ntrain].reshape(-1,1)
    testX  = [ x_setting[i][ntrain:] for i in range(depth) ]
    testY  = y_setting[ntrain:].reshape(-1,1)
    return trainX, trainY, testX, testY

def train_batch_MLP(trainX, trainY, batch_size):
    if len(trainX) % batch_size == 0:
        L = int(len(trainX[0]) / batch_size)
    else:
        L = int(len(trainX[0]) / batch_size) + 1
    TrainX = [ trainX[i*batch_size : (i+1)*batch_size] for i in range(L) ]
    TrainY = [ trainY[i*batch_size : (i+1)*batch_size] for i in range(L) ]
    return TrainX, TrainY

def train_batch_LSTM(trainX, trainY, depth, batch_size):
    if len(trainX[0]) % batch_size == 0:
        L = int(len(trainX[0]) / batch_size)
    else:
        L = int(len(trainX[0]) / batch_size) + 1
        
    TrainX, TrainY = [ [] for i in range(L) ], []
    for i in range(L):
        TrainY.append(trainY[i*batch_size : (i+1)*batch_size])
    for dep in range(depth):
        for i in range(L):
            TrainX[i].append(trainX[dep][i*batch_size : (i+1)*batch_size])  
    return TrainX, TrainY 

def LSTM(trainX, trainY, testX, testY, depth, hidden_dim, lr1, lr2, 
         batch_size, epochs, us):
    
    tf.reset_default_graph()
    input_dim  = trainX[0].shape[1]
    output_dim = trainY.shape[1]

    activation1 = tf.nn.sigmoid
    activation2 = tf.math.tanh
    activation3 = tf.nn.relu
    
    TrainX, TrainY = train_batch_LSTM(trainX, trainY, depth, batch_size)
    
    X = [tf.placeholder(tf.float32, [None,  input_dim]) for i in range(depth)]
    Y =  tf.placeholder(tf.float32, [None, output_dim]) 

    Weight1 = tf.Variable(tf.random_uniform([input_dim , 4*hidden_dim], -us, us))
    Weight2 = tf.Variable(tf.random_uniform([hidden_dim, 4*hidden_dim], -us, us))
    bias    = tf.Variable(tf.random_uniform([            4*hidden_dim], -us, us))

    W = tf.Variable(tf.random_uniform([hidden_dim, 1], -us, us))
    b = tf.Variable(tf.random_uniform([            1], -us, us))

    A = tf.matmul(X[0], Weight1) + bias
    F, I ,O, G, C, H = [],[],[],[],[],[]
    F.append(activation1(A[:,              : 1*hidden_dim]))
    I.append(activation1(A[:, 1*hidden_dim : 2*hidden_dim]))
    O.append(activation1(A[:, 2*hidden_dim : 3*hidden_dim]))
    G.append(activation2(A[:, 3*hidden_dim :             ]))
    C.append( I[-1] * G[-1] )
    H.append( O[-1] * activation2( C[-1] ))
    for i in range(1, depth):
        B = tf.matmul(X[i], Weight1) + tf.matmul(H[-1], Weight2) + bias    
        F.append(activation1(B[:,              : 1*hidden_dim]))
        I.append(activation1(B[:, 1*hidden_dim : 2*hidden_dim]))
        O.append(activation1(B[:, 2*hidden_dim : 3*hidden_dim]))
        G.append(activation2(B[:, 3*hidden_dim :             ]))
        C.append( F[-1] * C[-1] + I[-1] * G[-1] )
        H.append( O[-1] * activation2( C[-1] )) 
      
    #output = activation3(tf.matmul(H[-1], W) + b)
    #output = activation1(tf.matmul(H[-1], W) + b    )
    output = tf.matmul(H[-1], W) + b
     
    cost = tf.reduce_mean(tf.square(Y - output)) 
    lr   = tf.placeholder(tf.float32, [])
    gogo = tf.train.AdamOptimizer(lr).minimize(cost)

    real = tf.placeholder(tf.float32, [None, output_dim]) 
    pred = tf.placeholder(tf.float32, [None, output_dim]) 
    rmse = tf.reduce_mean( tf.square( real - pred ) ) 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    LR = np.linspace(lr1, lr2, epochs)
    for epoch in range(epochs):
        for i in range(len(TrainX)):
            feed1 = {Y:TrainY[i], lr:LR[epoch]}
            for dep in range(depth):
                feed1[X[dep]] = TrainX[i][dep]                  
            sess.run(gogo, feed_dict = feed1)
            
        if epoch % int(epochs/5) == 0:  
            feed2 = {}
            for dep in range(depth):
                feed2[X[dep]] = testX[dep]
            training_error = sess.run(tf.sqrt(cost),  feed_dict = feed1)
            prediction     = sess.run(output       ,  feed_dict = feed2) 
            testing_error  = sess.run(tf.sqrt(rmse),  feed_dict = {real:testY, pred:prediction})    
            #print('Epoch:', epoch, 'Training Error:',training_error,'and','Testing Error:', testing_error)
          
    return prediction.flatten()

def MLP(trainX, trainY, testX, testY, dim, lr1, lr2, batch_size, epochs, us):
    tf.reset_default_graph()
    activation1 = tf.nn.sigmoid
    activation2 = tf.math.tanh
    activation3 = tf.nn.relu
    
    TrainX, TrainY = train_batch_MLP(trainX, trainY, batch_size)
    
    X = tf.placeholder(tf.float32, [None, trainX.shape[1]])
    Y = tf.placeholder(tf.float32, [None, trainY.shape[1]])
    
    W = [ tf.Variable(tf.random_normal([dim[i], dim[i+1]])) for i in range(len(dim) - 1) ]
    b = [ tf.Variable(tf.random_normal([dim[i+1]]))         for i in range(len(dim) - 1) ]
    A = [X]
    for i in range(len(dim) - 2):
        A.append(activation1(tf.matmul(A[-1],W[i]) + b[i]))
    
    #output = activation3(tf.matmul(A[-1], W[-1]) + b[-1])
    #output = tf.tanh(tf.matmul(A[-1], W[-1]) + b[-1])
    output = tf.matmul(A[-1], W[-1]) + b[-1]
    cost = tf.reduce_mean(tf.square(Y - output)) 
    lr   = tf.placeholder(tf.float32, [])
    gogo = tf.train.AdamOptimizer(lr).minimize(cost)
    real = tf.placeholder(tf.float32, [None, trainY.shape[1]])
    pred = tf.placeholder(tf.float32, [None, trainY.shape[1]])
    rmse = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(real - pred))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    LR = np.linspace(lr1, lr2, epochs)
    for epoch in range(epochs):
        for i in range(len(TrainX)):
            feed1 = {X:TrainX[i], Y:TrainY[i], lr:LR[epoch]}
            sess.run(gogo, feed_dict=feed1)
        if epoch % int(epochs/5) == 0:
            feed2 = {X:testX}
            training_error = sess.run(tf.sqrt(cost),  feed_dict = feed1)
            prediction     = sess.run(output       ,  feed_dict = feed2) 
            testing_error  = sess.run(tf.sqrt(rmse),  feed_dict = {real:testY, pred:prediction})    
            #print('Epoch:', epoch, 'Training Error:',training_error,'and','Testing Error:', testing_error)
    return prediction.flatten()

def RNN(trainX, trainY, testX, testY, depth, hidden_dim, lr1, lr2, 
         batch_size, epochs, us):
    
    tf.reset_default_graph()
    input_dim  = trainX[0].shape[1]
    output_dim = trainY.shape[1]

    activation1 = tf.nn.sigmoid
    activation2 = tf.math.tanh
    activation3 = tf.nn.relu
    
    TrainX, TrainY = train_batch_LSTM(trainX, trainY, depth, batch_size)
    
    X = [tf.placeholder(tf.float32, [None,  input_dim]) for i in range(depth)]
    Y =  tf.placeholder(tf.float32, [None, output_dim]) 

    Weight1 = tf.Variable(tf.random_uniform([input_dim , hidden_dim], -us, us))
    Weight2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -us, us))
    bias    = tf.Variable(tf.random_uniform([            hidden_dim], -us, us))
    
    W = tf.Variable(tf.random_uniform([hidden_dim, 1], -us, us))
    b = tf.Variable(tf.random_uniform([            1], -us, us))

    A = tf.matmul(X[0], Weight1) + bias
    H = [ activation1(A) ]
    for i in range(1, depth):
        H.append( activation1(tf.matmul(X[i], Weight1) + tf.matmul(H[-1], Weight2) + bias) )
      
    #output = activation3(tf.matmul(H[-1], W) + b)
    #output = tf.tanh(tf.matmul(H[-1], W) + b)
    output = tf.matmul(H[-1], W) + b
    cost = tf.reduce_mean(tf.square(Y - output)) 
    lr   = tf.placeholder(tf.float32, [])
    gogo = tf.train.AdamOptimizer(lr).minimize(cost)

    real = tf.placeholder(tf.float32, [None, output_dim]) 
    pred = tf.placeholder(tf.float32, [None, output_dim]) 
    rmse = tf.reduce_mean( tf.square( real - pred ) ) 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    LR = np.linspace(lr1, lr2, epochs)
    for epoch in range(epochs):
        for i in range(len(TrainX)):
            feed1 = {Y:TrainY[i], lr:LR[epoch]}
            for dep in range(depth):
                feed1[X[dep]] = TrainX[i][dep]                  
            sess.run(gogo, feed_dict = feed1)
            
        if epoch % int(epochs/5) == 0:  
            feed2 = {}
            for dep in range(depth):
                feed2[X[dep]] = testX[dep]
            training_error = sess.run(tf.sqrt(cost),  feed_dict = feed1)
            prediction     = sess.run(output       ,  feed_dict = feed2) 
            testing_error  = sess.run(tf.sqrt(rmse),  feed_dict = {real:testY, pred:prediction})    
            #print('Epoch:', epoch, 'Training Error:',training_error,'and','Testing Error:', testing_error)
          
    return prediction.flatten()