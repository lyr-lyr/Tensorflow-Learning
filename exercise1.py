"""
exercise 1
Use the GradientDescentOptimizer in TensorFlow 
to give an example of linear learning model 
"""

import tensorflow as tf
import numpy as np

#Generate 100 random numbers
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2

#Constructe a linear model
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#Minimize two cost function
loss = tf.reduce_mean(tf.square(y_data-y))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

#Training the linear model,output parameters every 20 inerations
with tf.Session() as sess:
	sess.run(init)
	for step in range (201):
		sess.run(train)
		if step%20 ==0:
			print(step,sess.run([k,b]))

