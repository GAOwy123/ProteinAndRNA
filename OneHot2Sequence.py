import numpy as np
import os


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
    return ''.join(sequence).strip('N')

# os.chdir('D:/nn/HepG2_BUD13/')
test_x = np.load(file="test_x.npy")  # (291, 222, 4, 1)
test_y = np.load(file="test_y.npy")  # (291,)
test_y_encoded = np.load(file="test_y_encoded.npy")  # (291, 2)
train_validation_x = np.load(file="train_validation_x.npy")
train_validation_y = np.load(file="train_validation_y.npy")
train_validation_y_encoded = np.load(file="train_validation_y_encoded.npy")
# model = load_model('length223-learning_rate-0.1-momentum-0.9-weight_decay-0.01-dropout-0.2-batch_size-128.hdf5', custom_objects={'auc': auc})
#
test_x_pos = test_x[test_y == 1]
test_x_neg = test_x[test_y == 0]
train_validation_x_pos = train_validation_x[train_validation_y == 1]
train_validation_x_neg = train_validation_x[train_validation_y == 0]

hash = dict()
def WriteFastaFile(inputdata, outFile, label):
    # hash = dict()
    with open(outFile, 'a') as w:
        for i in range(inputdata.shape[0]):
            if OneHot2Sequence(inputdata[i, :, :, 0]) not in hash.keys():
                hash[OneHot2Sequence(inputdata[i, :, :, 0])] = OneHot2Sequence(inputdata[i, :, :, 0])
                if len(OneHot2Sequence(inputdata[i, :, :, 0])) > 9:
                    w.write(">"+label+str(i)+"\n")
                    w.write(OneHot2Sequence(inputdata[i, :, :, 0])+"\n")




if not os.path.exists("gkmsvm"):
    os.makedirs("gkmsvm")
os.chdir("gkmsvm")


WriteFastaFile(inputdata=test_x_pos, outFile='test_pos.fa',label='test-pos')
WriteFastaFile(inputdata=test_x_neg, outFile='test_neg.fa',label='test-neg')
WriteFastaFile(inputdata=train_validation_x_pos, outFile='train_validation_pos.fa',label='train-validation-pos')
WriteFastaFile(inputdata=train_validation_x_neg, outFile='train_validation_neg.fa',label='train-validation-neg')

#combine the test-pos.fa file and the test-neg.fa file
with open('test.fa', 'a') as w:
    with open('test_pos.fa', 'r') as r:
        for line in r.readlines():
            w.write(line)
    with open('test_neg.fa', 'r') as r:
        for line in r.readlines():
            w.write(line)

