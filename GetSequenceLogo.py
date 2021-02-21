import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import load_model
import numpy as np
import sys, getopt
import re
from keras.models import Model
import AUC

modelFile = ''
outputfile = ''
test_x_file = ''
test_y_file = ''
test_y_encoded_file = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:o:x:y:e:", ["modelfile=", "outputfile=", "test_x=","test_y=","test_y_encoded="])
except getopt.GetoptError:
    print("python GetSequenceLogo.py -m <modelFile> -o <outputfile> -x <test x file> -y <test y file> -e <test y encoded file>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('python GetSequenceLogo.py -m <modelFile> -o <outputfile> -x <test x file> -y <test y file> -e <test y encoded file>')
        sys.exit()
    elif opt in ("-m", "--modelfile"):
        modelFile = arg
    elif opt in ("-o", "--outputfile"):
        outputfile = arg
    elif opt in ("-x", "--test_x"):
        test_x_file = arg
    elif opt in ("-y", "--test_y"):
        test_y_file = arg
    elif opt in ("-e", "--test_y_encoded"):
        test_y_encoded_file = arg
    
print('The model file is: ', modelFile)
print('The output file is: ', outputfile)



test_x = np.load(file=test_x_file)  # (291, 222, 4, 1)
test_y = np.load(file=test_y_file)  # (291,)
test_y_encoded = np.load(file=test_y_encoded_file)  # (291, 2)

test_x_true = test_x[test_y == 1]
test_y_encoded_true = test_y_encoded[test_y == 1]
# train_validation_x_true = train_validation_x[train_validation_y == 1]
# train_validation_y_encoded_true = train_validation_y_encoded[train_validation_y == 1]
# test_x_true = np.concatenate((test_x_true,train_validation_x_true), axis=0)
# test_y_encoded_true = np.concatenate((test_y_encoded_true,train_validation_y_encoded_true), axis=0)
seq_len = test_x.shape[1]

model = load_model(modelFile, custom_objects={'auc': AUC.auc})

if not os.path.exists("motif"):
    os.makedirs("motif")
os.chdir("motif")
def OneHot2Sequence(matrix):
    seq_len = matrix.shape[0]
    sequence = ['N'] * seq_len
    for i in range(seq_len):
        if sum(matrix[i]) == 1:
            if np.argmax(matrix[i]) == 0:
                sequence[i] = 'A'
            if np.argmax(matrix[i]) == 1:
                sequence[i] = 'U'
            if np.argmax(matrix[i]) == 2:
                sequence[i] = 'C'
            if np.argmax(matrix[i]) == 3:
                sequence[i] = 'G'
    return ''.join(sequence)


ave_layer_model = Model(inputs=model.inputs, outputs=model.layers[4].output)
ave_result =ave_layer_model.predict(test_x_true)

seq_list = []
for i in range(ave_result.shape[0]):
    position = np.where(ave_result[i, :, :, :] == np.max(ave_result[i, :, :, :]))[0][0]
    test_x_true_sequence = OneHot2Sequence(test_x_true[i, :, :, 0])
    if position < 12:
        seq_list.append(test_x_true_sequence[0:position+12])
    elif position > ave_result.shape[1]-12:
        seq_list.append(test_x_true_sequence[position-12:ave_result.shape[1]])
    else:
        seq_list.append(test_x_true_sequence[(position-12):(position + 12)])


with open(outputfile, 'w+') as w:
     for i in seq_list:
         i = i.strip('N')
         i = i.center(24, "N").upper()
         w.write(i+"\n")

# with open(RBP+'-sequencelogo.txt', 'w+') as w:
    # for i in seq_list:
        # i = i.strip('N')
        # if len(i) == 24:
            # w.write(i+"\n")



