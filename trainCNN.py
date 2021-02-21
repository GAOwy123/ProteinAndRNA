import os
os.environ['KERAS_BACKEND']='tensorflow'
import time
import classify_DataProcessing as cdp
import re
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import classify_CreateModel as ccm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import calculate_performace as cp
import getopt,sys


path = ''
RBP = ''
file01 = ''
file02 = ''
file = ''

try:
    opts, args = getopt.getopt(sys.argv[1:],"h:p:r:1:2:f:",["path=","RBP=","file01=","file02=","file="])
except getopt.GetoptError:
    print("python trainCNN.py -p <path> -r <RBP name> -1 <file01> -2 <file02> -f <file>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
         print('python trainCNN.py -p <path> -r <RBP name> -1 <file01> -2 <file02> -f <file>')
         sys.exit()
    elif opt in ("-p", "--path"):
         path = arg
    elif opt in ("-r", "--RBP"):
         RBP = arg
    elif opt in ("-1", "--file01"):
         file01 = arg
    elif opt in ("-2", "--file02"):
         file02 = arg
    elif opt in ("-f", "--file"):
         file = arg

print('The path is: ', path)
print('The training RBP is: ', RBP)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
learning_rate = [0.001, 0.01, 0.1]
batch_size = 512
momentum = [0.1, 0.5, 0.9]
weight_decay = [0.0001, 0.001, 0.01]
dropout = [0.2, 0.5, 0.8]
min_delta = 0.01
decay = 0.0  # decay = [0.0001, 0.001, 0.01]
patience = 10
epochs = 1000
shuffle = True
verbose = 1

os.chdir(path)
if not os.path.exists(RBP):
    os.makedirs(RBP)

os.chdir(path+'/'+RBP)


if not os.path.exists("models"):
    os.makedirs("models")
# data processing
seq_len, train_validation_x, train_validation_y, train_validation_y_encoded, test_x, test_y, test_y_encoded = \
    cdp.data_processing(file=file,
                        file01=file01,
                        file02=file02,
                        seed=0,
                        test_frac=0.1)


np.save(file="train_validation_x.npy", arr=train_validation_x)
np.save(file="train_validation_y.npy", arr=train_validation_y)
np.save(file="train_validation_y_encoded.npy", arr=train_validation_y_encoded)
np.save(file="test_x.npy", arr=test_x)
np.save(file="test_y.npy", arr=test_y)
np.save(file="test_y_encoded.npy", arr=test_y_encoded)

print('The train, validation and test data had been saved.')
print('Training CNN model begins!')
train_validation_x = np.load(file="train_validation_x.npy")
train_validation_y = np.load(file="train_validation_y.npy")
train_validation_y_encoded = np.load(file="train_validation_y_encoded.npy")

seq_len = train_validation_x.shape[1]
#w = open("cvperformance.txt","a")
#w.write('learning_rate'+"\t"+'weight_decay'+"\t"+'dropout'+"\t"+'batch_size'+"\t"+'accuracy'+"\t"+'precision'+"\t"+'sensitivity'+"\t"+'specificity'+"\t"+'MCC'+"\t"+'AUC'+'\n')

#result = pd.DataFrame(columns=('learning_rate', 'weight_decay', 'dropout', 'momentum', 
#                      'accuracy', 'precision', 'sensitivity', 'specificity', 'MCC', 'AUC','Auc'))
result = pd.DataFrame(columns=('learning_rate', 'weight_decay', 'dropout', 'momentum', 
                      'accuracy', 'precision', 'sensitivity', 'specificity', 'MCC', 'AUC'))
result.to_csv("cvperformance.csv", sep=",", header=True, index=False, mode='a')

for a in learning_rate:
    for d in weight_decay:
        for e in dropout:
            for f in momentum:
                modelcheckpoint = ModelCheckpoint(filepath="models/length" +
                                                  str(seq_len) + "-learning_rate-" + str(a) + "-weight_decay-" +
                                                  str(d) + "-dropout-" + str(e) + "-momentum-" + str(f) + ".hdf5",
                                                  monitor="val_auc", save_best_only=True, mode="max")
                earlystopping = EarlyStopping(monitor='val_auc', min_delta=min_delta, patience=patience,
                                              mode='max', verbose=2)  # val_loss
                callbacks = [earlystopping, modelcheckpoint]
                flag = 0
                kf = KFold(n_splits=5, shuffle=True, random_state=1)
                for train, val in kf.split(train_validation_x, train_validation_y_encoded):
                    flag = flag + 1
                    np.save(file="trainset_x"+str(flag)+".npy", arr = train_validation_x[train])
                    np.save(file="trainset_y"+str(flag)+".npy", arr=train_validation_y_encoded[train])
                    np.save(file="validationset_x"+str(flag)+".npy", arr=train_validation_x[val])
                    np.save(file="validationset_y"+str(flag)+".npy", arr=train_validation_y_encoded[val])
                    model = ccm.create_classify_model(input_shape=(seq_len, 4, 1), padding='same', 
                                                      pool_size=(1, 4), strides=1, learning_rate=a, 
                                                      weight_decay=d, dropout=e,momentum=f)
                    model.fit(train_validation_x[train], train_validation_y_encoded[train], batch_size=batch_size,
                              epochs=epochs, shuffle=shuffle, verbose=verbose,
                              validation_data=(train_validation_x[val], train_validation_y_encoded[val]),
                              callbacks=callbacks)
                    model.save("models/length" + str(seq_len) + "-learning_rate-" + str(a) + "-weight_decay-" +
                        str(d) + "-dropout-" + str(e) + "-momentum-" + str(f) + "-cvmodel-"+str(flag)+".hdf5", include_optimizer=True)
                    #calculate the six performance
                    train_pred_y = model.predict(train_validation_x[val])
                    #fpr, tpr, threshold = roc_curve(train_validation_y_encoded[val][:, 1], train_pred_y[:, 1])
                    #Auc = auc(fpr, tpr)
                    pred_y = []
                    for y in train_pred_y:
                        pred_y.append(np.argmax(y))
                        
                    label_y = []
                    for y in train_validation_y_encoded[val]:
                        label_y.append(np.argmax(y))
                        
                    accuracy, precision, sensitivity, specificity, MCC, AUC = cp.calculate_performace(test_num=len(pred_y),
                                                                                                      pred_y=pred_y,
                                                                                                      labels=label_y)
                    #w.write(str(a)+"\t"+str(d)+"\t"+str(e)+"\t"+str(f)+"\t"+str(accuracy)+"\t"+str(precision)+"\t"+str(sensitivity)+"\t"+str(specificity)+"\t"+str(MCC)+"\t"+str(AUC)+"\t"+str(Auc)+"\n")
                    #result.loc[0] = [a, d, e, f, accuracy, precision, sensitivity, specificity, MCC, AUC, Auc]
                    result.loc[0] = [a, d, e, f, accuracy, precision, sensitivity, specificity, MCC, AUC]
                    result.to_csv("cvperformance.csv", sep=",", header=False, index=False, mode='a')

#w.close()

print('Training CNN model has been finished!')
