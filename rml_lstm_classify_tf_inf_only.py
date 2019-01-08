import numpy as np
import tensorflow as tf
import sys
import operator
from numpy import linalg as la 
from collections import Counter
import os
import time
import pickle

maxlen = 128 
snrs=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
mods=['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
test_idx = ""
lbl = ""

with open("./testdata/test_idx.pkl", "rb") as fp:
  test_idx = pickle.load(fp)

with open("./testdata/lbl.pkl", "rb") as fp:
  lbl =pickle.load(fp)

X_test = np.load("./testdata/xtest.npy")
Y_test = np.load("./testdata/ytest.npy")


print("--"*50)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

# model setting

lstmInputs = tf.placeholder(tf.float32, [None, maxlen, 2])
target = tf.placeholder(tf.float32, [None, len(mods)])

lstmCell1 = tf.contrib.rnn.LSTMBlockCell(128)
lstmCell2 = tf.contrib.rnn.LSTMBlockCell(128)
stackedLstm = tf.contrib.rnn.MultiRNNCell([lstmCell1, lstmCell2])
lstmOutputs, state = tf.nn.dynamic_rnn(cell=stackedLstm, inputs=lstmInputs, dtype=tf.float32)
fcOut = tf.contrib.layers.fully_connected(lstmOutputs[:, -1], 10, activation_fn=None)
predictions = tf.contrib.layers.softmax(fcOut)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    accList=[]
    saver.restore(sess, "./tfmodel/rml_model.ckpt")

    classes = mods
    acc={}
    
    for snr in snrs:
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        # estimate classes
        start_time = time.time()
        test_Y_i_hat = sess.run(predictions, {lstmInputs: test_X_i})
        duration = time.time() - start_time

        conf = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print(test_X_i.shape)
        print("SNR:{0} Overall Accuracy: {1:3.2f}, Time:{2:5.4f}s, #records:{3}".format(snr, cor / (cor+ncor), duration, test_X_i.shape[0]))
        acc[snr] = 1.0*cor/(cor+ncor)

    print(acc)
    sess.close()
