#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cifar import Cifar
from tqdm import tqdm
import pretrained
import numpy as np
import tensorflow as tf
import helper
import pickle

import os
import sys
import time
from tf_cnnvis import *


n_classes = 10
learning_rate = 0.00001
batch_size = 16
no_of_epochs = 1
no_of_test_splits = 20
image_size = 224

conv5 = tf.layers.flatten(pretrained.maxpool5)
hidden1=pretrained.maxpool1
hidden5=pretrained.maxpool5

weights = tf.Variable(tf.zeros([9216, n_classes]), name="output_weight")
bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
model = tf.matmul(conv5, weights) + bias

outputs = tf.placeholder(tf.float32, [None, n_classes])

cost = tf.losses.softmax_cross_entropy(outputs, model)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cifar = Cifar(batch_size=batch_size)
cifar.create_test_set(dim=n_classes)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(no_of_epochs):
        h1, h5, y_tst =[], [], []
        for i in tqdm(range(len(cifar.batches)),
                desc="Epoch {}".format(epoch),
                unit=" batch "):
        #    continue
        #    import pdb; pdb.set_trace()
            this_batch = cifar.batch(i)
            input_batch, out = helper.reshape_batch(this_batch, (image_size, image_size), n_classes)

            
            sess.run([optimizer],
                        feed_dict={
                            pretrained.x: input_batch,
                            outputs: out })
        #    break
            


        total_acc = 0
        test_batch_counter = 0
        
        if epoch==no_of_epochs-1:
           inp_test, out_test = cifar.test_set
           inp_test = np.split(inp_test, no_of_test_splits)
           out_test = np.split(out_test, no_of_test_splits)
           for each_inp_test, each_out_test in tqdm(zip(inp_test, out_test),
                desc="Test".format(epoch),
                unit=" batch ",
                total=no_of_test_splits):
                test_batch_counter += 1
                if test_batch_counter < 20:
                    each_inp_test = cifar.resize_batch_input_test(each_inp_test)
      #      each_test_acc = sess.run(accuracy,
      #              feed_dict={
      #                  pretrained.x: each_inp_test,
      #                  outputs: each_out_test })                                
      #      total_acc = total_acc + each_test_acc
     
            #the activation values of h1 of one batch of the test
            
      #              each_h1 = hidden1.eval(feed_dict={pretrained.x:each_inp_test})
                    each_h5 = hidden5.eval(feed_dict={pretrained.x:each_inp_test})
      #              h1.append(each_h1)
                    h5.append(each_h5)
                    y_tst.append(each_out_test)
          
                 
      #  test_acc = total_acc / no_of_test_splits
      #  print("Test Acc: {}".format(test_acc))
      
    input_batch, out = helper.reshape_batch(cifar.batch(0), (image_size, image_size), n_classes)  
    input_batch_1, out_1 = input_batch[1:2], out[1:2]
    feed_dict = {pretrained.x:input_batch_1, outputs: out_1} 
    # deconv visualization
    layers = ["r", "p", "c"]
    total_time = 0
    start = time.time()
    # api call
    is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
                                  input_tensor= pretrained.x, layers=layers, 
                                  path_logdir=os.path.join("Log","Cifar10Example"), 
                                  path_outdir=os.path.join("Output","Cifar10Example"))
    start = time.time() - start
    print("Total Time = %f" % (start))
    
    #calculate the values for the hidden activiations of the first hidden layer
    #h1=np.concatenate( h1, axis=0 )
    h5=np.concatenate( h5, axis=0 )
    y_tst=np.concatenate( y_tst, axis=0 )
    y_tst = np.argmax(y_tst, axis=1)
    
#    f = open('h1.pckl', 'wb')
#    pickle.dump(h1, f)
#    f.close()
    
    f = open('h5.pckl', 'wb')
    pickle.dump(h5, f)
    f.close()
    
    f = open('testY.pckl', 'wb')
    pickle.dump(y_tst, f)
    f.close()