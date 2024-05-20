from cifar import Cifar
from tqdm import tqdm
import tensornets as nets
import tensorflow as tf
import numpy as np
import helper
import pickle
import sys
#from sklearn.externals import joblib

import os
import time
#from tf_cnnvis import *

learning_rate = 0.00001
batch_size = 16
no_of_epochs = 10
print(no_of_epochs)
n_classes = 10
no_of_test_splits = 100
image_size = 224

inputs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
outputs = tf.placeholder(tf.float32, [None, n_classes])

vgg = nets.VGG19(inputs, is_training=True, classes=n_classes)

hidden1 = tf.get_default_graph().get_tensor_by_name("vgg19/conv1/pool/MaxPool:0")
hidden2 = tf.get_default_graph().get_tensor_by_name("vgg19/conv2/pool/MaxPool:0")
hidden3 = tf.get_default_graph().get_tensor_by_name("vgg19/conv3/pool/MaxPool:0")
hidden4 = tf.get_default_graph().get_tensor_by_name("vgg19/conv4/pool/MaxPool:0")
hidden5 = tf.get_default_graph().get_tensor_by_name("vgg19/conv5/pool/MaxPool:0")

model = tf.identity(vgg, name='logits')
cost = tf.losses.softmax_cross_entropy(outputs, vgg)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
vgg.print_outputs()
vgg.print_summary()

cifar = Cifar(batch_size=batch_size)
cifar.create_test_set(dim=n_classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(vgg.pretrained())
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    
    for epoch in range(no_of_epochs):
        h1, h2, h3, h4, h5, y_tst =[], [], [], [], [], []
        for i in tqdm(range(len(cifar.batches)),
                desc="Epoch {}".format(epoch),
                unit=" batch "):
            #continue
            this_batch = cifar.batch(i)
            input_batch, out = helper.reshape_batch(this_batch, (image_size, image_size), n_classes)
            sess.run([optimizer],
                        feed_dict={
                            inputs: input_batch,
                            outputs: out },
                        options=run_options)
        acc, loss = sess.run([accuracy, cost],
                       feed_dict={
                            inputs: input_batch,
                            outputs: out },
                       options=run_options)

        print("Last Batch Acc: {} Loss: {}".format(acc, loss))

        #Only test when all the epoches are completed
        if epoch==no_of_epochs-1:
            #inp_test, out_test = cifar.test_set
            #out_test = np.argmax(out_test, axis=1)
            
            #select 1000 test instances with 100 instances in each class
            #np.random.seed(seed=222)
            #idx=np.empty(0,dtype="int8")
            #for i in range(0,len(np.unique(out_test))):
            #    idx=np.append(idx,np.random.choice(np.where((out_test[0:len(out_test)])==i)[0],100,replace=False))
            #inp_test = inp_test[idx]
            #out_test = out_test[idx]
            
            f=open('inp_test.pckl','rb')
            inp_test = pickle.load(f)
            f.close()
            
            f=open('out_test.pckl', 'rb')
            out_test = pickle.load(f)
            f.close()
            
            print(inp_test.shape)
            print(out_test.shape)
            print(np.unique(out_test,return_counts=True))
            
            #reconvert testY into onehot
            b = np.zeros((1000, 10))
            b[np.arange(1000), out_test] = 1
            out_test = b
            
            inp_test = np.split(inp_test, no_of_test_splits)
            out_test = np.split(out_test, no_of_test_splits)
            
            #test starts here
            total_acc = 0
            for each_inp_test, each_out_test in tqdm(zip(inp_test, out_test),
                    desc="Test".format(epoch),
                    unit=" batch ",
                    total=no_of_test_splits):
                each_inp_test = cifar.resize_batch_input_test(each_inp_test, new_size=(image_size, image_size))
                each_test_acc = sess.run(accuracy,
                        feed_dict={
                            inputs: each_inp_test,
                            outputs: each_out_test},
                        options=run_options)
                total_acc = total_acc + each_test_acc
                
                each_h1 = hidden1.eval(feed_dict={inputs:each_inp_test})
                each_h2 = hidden2.eval(feed_dict={inputs:each_inp_test})
                each_h3 = hidden3.eval(feed_dict={inputs:each_inp_test})
                each_h4 = hidden4.eval(feed_dict={inputs:each_inp_test})
                each_h5 = hidden5.eval(feed_dict={inputs:each_inp_test})
                #print(sys.getsizeof(each_h1))
                #f = open('h1_%s.pckl'%i, 'wb')

                h1.append(each_h1)
                h2.append(each_h2)
                h3.append(each_h3)
                h4.append(each_h4)
                h5.append(each_h5)
                y_tst.append(each_out_test) 
               
            print('*'*20)   
            test_acc = total_acc / no_of_test_splits
            print("Test Acc: {}".format(test_acc))
            
      
    #calculate the values for the hidden activiations of the first hidden layer
    h1= np.concatenate( h1, axis=0 )
    h2= np.concatenate( h2, axis=0 )
    h3= np.concatenate( h3, axis=0 )
    h4= np.concatenate( h4, axis=0 )
    h5 = np.concatenate( h5, axis=0 )
    y_tst = np.concatenate( y_tst, axis=0 )
    y_tst = np.argmax(y_tst, axis=1)
    
    #joblib.dump(h1, 'h1.pkl', compress=1)
    #del h1
    
    #joblib.dump(h5, 'h5.pkl', compress=1)
    #del h5
    
    f = open('h1.pckl', 'wb')
    pickle.dump(h1, f)
    f.close()
    del h1
    
    f = open('h2.pckl', 'wb')
    pickle.dump(h2, f)
    f.close()
    del h2
    
    f = open('h3.pckl', 'wb')
    pickle.dump(h3, f)
    f.close()
    del h3
    
    f = open('h4.pckl', 'wb')
    pickle.dump(h4, f)
    f.close()
    del h4
    
    f = open('h5.pckl', 'wb')
    pickle.dump(h5, f)
    f.close()
    del h5
    
    f = open('testY.pckl', 'wb')
    pickle.dump(y_tst, f)
    f.close()
    #import pdb;pdb.set_trace()
    #######################################################################
    # feature visualization starts here, visualize only one instance
    #input_batch, out = helper.reshape_batch(cifar.batch(0), (image_size, image_size), n_classes)  

    #input_batch is a list with 16 elements, input_batch[1:2]select the second element with size (1, 224, 224, 3)
    #input_batch_1, out_1 = input_batch[1:2], out[1:2]
    #feed_dict = {inputs:input_batch_1, outputs: out_1} 
    # print sample image
    #import scipy.misc
    #import pdb; pdb.set_trace()
    #print(input_batch_1[0].shape)
    #scipy.misc.imsave('sample_test_img.jpg', input_batch_1[0])
    
    # deconv visualization
    #layers = ["p"]
    #total_time = 0
    #start = time.time()
    # api call
    #is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
    #                              input_tensor= inputs, layers=layers, 
    #                              path_logdir=os.path.join("Log","Cifar10Example"), 
    #                              path_outdir=os.path.join("Output","Cifar10Example"))
    #start = time.time() - start
    #print("Total Time = %f" % (start)) 
    #######################################################################
