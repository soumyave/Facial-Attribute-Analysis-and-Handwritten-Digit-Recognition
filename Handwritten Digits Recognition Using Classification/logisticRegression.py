import tensorflow as tf
import numpy as np
import math

def logisticRegression(): 
    n_input = 784  
    n_classes = 10
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_classes]))
    }
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    out_layer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    return out_layer,x,y

def train_and_test_LR(mnist,uspsImages,uspsLabels):
    learning_rate = 1.0
    training_epochs = 3500
    batch_size = 950
    pred,x,y = logisticRegression()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init) 
        for epoch in range(training_epochs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy for MNIST test data for logistic regression trained data:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})*100)
        print("Accuracy for USPS test data for logistic regression trained data:", accuracy.eval({x: uspsImages, y: uspsLabels})*100)