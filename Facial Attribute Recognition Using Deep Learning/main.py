from PIL import Image
import numpy as np
import tensorflow as tf
import math

def getFileName(num):
    s='000000'
    length=len(s)-1
    i=0
    while num!=0:
        rem=num%10
        temp=list(s)
        temp[length-i]=str(rem)
        s="".join(temp)
        #s[length-i]=re
        num=int(num/10)
        i+=1
    return s

def getImageData(start,size):
    Image_list=[]
    if start==0:
        start+=1
    for i in range(start,(start+size)):
        filename=getFileName(i)
        im=Image.open('img_align_celeba/img_align_celeba/'+filename+'.jpg')
        im=im.resize((28,28))
        temp1=list(im.getdata())
        Image_list.append(temp1)
        im.close() 
    return Image_list

def getLabels():
    filepath = 'list_attr_celeba.txt' 
    labels_list=[]
    i=0
    with open(filepath) as fp:  
        line = fp.readline()
        while line:
            line = fp.readline()
            if "jpg" in line:
                temp=[0,0]
                arr=line.split()
                x=int(arr[16])
                if x==-1:
                    temp[0]=1
                else:
                    temp[1]=1
                labels_list.append(temp)
    return labels_list

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(xl, W):
      return tf.nn.conv2d(xl, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(xl):
      return tf.nn.max_pool(xl, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def create_convoluted_multilayer_perceptron1():
    n_hidden_1 = 256  
    n_input = 784  
    n_classes = 2
    x = tf.placeholder("float", [None, n_input,3])
    y = tf.placeholder("float", [None, n_classes])
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    out_layer = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return out_layer,x,y,keep_prob

def train_and_test_convNN(labels):
    training_epochs = 1
    batch_size = 50
    pred,x,y,keep_prob = create_convoluted_multilayer_perceptron1()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    init = tf.global_variables_initializer()
    testing_labels=labels[40000:45000]
    testing_data=getImageData(40000,5000)
    batch_start=0
    with tf.Session() as sess:
        sess.run(init) 
        for epoch in range(training_epochs):
            batch_x=getImageData(batch_start,batch_size)
            batch_y=labels[batch_start:batch_start+batch_size]
            #print(len(batch_x),len(batch_y),batch_start,labels[batch_start])
            c = optimizer.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            batch_start+=batch_size
        #print('done')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy for Celebrity test data for convoluted Neural network trained data for reduced resolution data:", accuracy.eval({x: testing_data, y: testing_labels, keep_prob: 1.0})*100)
        
labels=getLabels()
train_and_test_convNN(labels)