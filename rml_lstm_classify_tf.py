import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import operator
import random
import _pickle as cPickle
from numpy import linalg as la 
from collections import Counter
import os
import pickle

maxlen = 128 
snrs=""
mods=""
test_idx=""
lbl =""
def gendata(fp, nsamples):
    global snrs, mods, test_idx, lbl
    Xd = cPickle.load(open(fp,'rb'), encoding="latin")
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    print(mods, snrs)
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.9)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    X_test =  X[test_idx]
    def to_onehot(yy):
        yylist=list(yy)
        yy1 = np.zeros([len(yylist), max(yylist)+1])
        yy1[np.arange(len(yylist)),yylist] = 1
        return yy1
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
   
    return (X_train,X_test,Y_train,Y_test)


def norm_pad_zeros(X_train,nsamples):
    print("Pad: ",X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train


def to_amp_phase(X_train,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
    
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train, X_test)


# xtrain1,xtest1,ytrain1,ytest1 = gendata("RML2016.10a_dict.dat",128)
xtrain1,xtest1,ytrain1,ytest1 = gendata("RML2016.10b.dat",128)


xtrain1,xtest1 = to_amp_phase(xtrain1,xtest1,128)

xtrain1 = xtrain1[:,:maxlen,:]
xtest1 = xtest1[:,:maxlen,:]

xtrain1 = norm_pad_zeros(xtrain1,maxlen)
xtest1 = norm_pad_zeros(xtest1,maxlen)

X_train = xtrain1
X_test = xtest1

Y_train = ytrain1
Y_test = ytest1

np.save("./testdata/xtest", X_test)
np.save("./testdata/ytest", Y_test)
with open("./testdata/test_idx.pkl", "wb") as fp:
  pickle.dump(test_idx, fp)

with open("./testdata/lbl.pkl", "wb") as fp:
  pickle.dump(lbl, fp)

ans=input("continue to run lstm? (Y/N):")
if(ans == 'N'):
  exit()


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt

print("--"*50)
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

# model setting

batchSize = 200

lstmInputs = tf.placeholder(tf.float32, [None, maxlen, 2])
target = tf.placeholder(tf.float32, [None, len(mods)])

lstmCell1 = tf.contrib.rnn.LSTMBlockCell(128)
lstmDropout = tf.contrib.rnn.DropoutWrapper(lstmCell1,input_keep_prob=1, output_keep_prob=0.8)
lstmCell2 = tf.contrib.rnn.LSTMBlockCell(128)
stackedLstm = tf.contrib.rnn.MultiRNNCell([lstmDropout, lstmCell2])
lstmOutputs, state = tf.nn.dynamic_rnn(cell=stackedLstm, inputs=lstmInputs, dtype=tf.float32)
fcOut = tf.contrib.layers.fully_connected(lstmOutputs[:, -1], 10, activation_fn=None)
predictions = tf.contrib.layers.softmax(fcOut)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=predictions)
optimizer = tf.train.AdamOptimizer(0.001)
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(predictions, 1))
error = tf.count_nonzero(mistakes)/batchSize

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    noOfBatches = int(X_train.shape[0]/batchSize)
    isTrain = False

    accList=[]
    if isTrain:
        epoch = 70
        for i in range(epoch):
            randTrainIdx = np.random.choice(range(0,X_train.shape[0]), size=batchSize, replace=False)
            ptr = 0
            for j in range(noOfBatches):
                inp, out = X_train[ptr:ptr+batchSize], Y_train[ptr:ptr+batchSize]
                ptr += batchSize
                sess.run(minimize,{lstmInputs: inp, target: out})
            randTestIdx = np.random.choice(range(0,X_test.shape[0]), size=batchSize, replace=False)
            randTestX = X_test[randTestIdx]
            randTestY = Y_test[randTestIdx]
            correct = sess.run(error,{lstmInputs: randTestX, target: randTestY})
            print('Epoch {0:2d} acc {1:3.1f}%'.format(i + 1, 100 * correct))
            accList.append(correct)
        savePath = saver.save(sess, "./tfmodel/rml_model.ckpt")
        print(accList)
        os.system("pause")
    else:
        saver.restore(sess, "./tfmodel/rml_model.ckpt")

    classes = mods
    acc={}
    for snr in snrs:
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        # estimate classes
        test_Y_i_hat = sess.run(predictions, {lstmInputs: test_X_i})
        print(test_Y_i_hat)
        width = 4.1 
        height = width / 1.618
        plt.figure(figsize=(width, height))
        plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1))
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("images\\confmat_"+str(snr)+".pdf")
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1 
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        plt.figure()
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor 
        print("Overall Accuracy: {0:3.2f}".format(cor / (cor+ncor)))
        acc[snr] = 1.0*cor/(cor+ncor)
    print(acc)

    sess.close()
