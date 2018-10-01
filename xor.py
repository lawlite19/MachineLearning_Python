## References 
# https://datascience.stackexchange.com/questions/9850/neural-networks-which-cost-function-to-use

# xor for 2 inputs

import tensorflow as tf
import numpy as np

# fixing randome number seed
#SEED = 42
#tf.set_random_seed(SEED)

# parameters
epochs = 100000
num_of_hidden_layers = 1
learning_rate = 0.01
num_of_neurons = [2,3,1]
size_of_data = 4
features = 2
labels = 1


def initialisation(shape):
	return tf.Variable(tf.random_normal(shape=shape,mean=0,stddev=1.0/float(features)))

X = tf.placeholder(tf.float32,shape=(None,features))
Y = tf.placeholder(tf.float32,shape=(None,labels))

W0 = initialisation([2,3])
W1 = initialisation([3,1])
B0 = initialisation([3])
B1 = initialisation([1])


# declaring activation layers
Y1 = tf.nn.sigmoid((tf.matmul(X,W0) + B0),name='activationLayer1')
Y2 = tf.nn.sigmoid((tf.matmul(Y1,W1) + B1),name='activationLayer2')
#predict = tf.argmax(Y2, axis=1)

# dataset
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]


#cost function
cost_function = tf.reduce_mean(( (Y * tf.log(Y2)) + 
        ((1 - Y) * tf.log(1.0 - Y2)) ) * -1)
    
#optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

model = tf.global_variables_initializer()

#Create a saver object which will save all the variables
saver = tf.train.Saver()

'''with tf.Session() as session:
    session.run(model)
    #print(session.run(W))   # print the value
    for i in range(0,epochs):
	    session.run(train_step, feed_dict={X: XOR_X, Y: XOR_Y})
	    if(i%1000==0):
	    	print ("Iteration ",i," ==>",'cost ', session.run(cost_function, feed_dict={X: XOR_X, Y: XOR_Y}))

    saver.save(session,'./xor_for_2_inputs.ckpt')
    session.close()

'''

# Prediction by loading the trained model

with tf.Session() as session:
	saver.restore(session, './xor_for_2_inputs.ckpt')
	print("Give number of input data :")
	n = input()
	for i in range(int(n)):
		a = input()
		b = input()
		x_input = np.array([[float(a),float(b)]])
		print (session.run(Y2, feed_dict={X: x_input}))
	session.close()