# predict the SNP
import os
os.environ['KERAS_BACKEND']='tensorflow'
import re
import sys, getopt
from keras.models import load_model
from keras.models import Model
import numpy as np
from AUC import auc
from OneHot import one_hot


modelFile = ''
outputfile = ''
path = ''
RBP = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], "hr:m:o:d:", ["RBP=","modelfile=", "outputfile=", "directory="])
except getopt.GetoptError:
    print("python RBPAndLncrnaPredict.py -r <RBP name> -m <modelFile> -o <outputfile> -d <lncRNA SNP directory>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('python RBPAndLncrnaPredict.py -r <RBP name> -m <modelFile> -o <outputfile> -d <lncRNA SNP directory>')
        sys.exit()
    elif opt in ("-r", "--RBP"):
        RBP = arg
    elif opt in ("-m", "--modelfile"):
        modelFile = arg
    elif opt in ("-o", "--outputfile"):
        outputfile = arg
    elif opt in ("-d", "--directory"):
        path = arg

print('The RBP is: ', RBP)
print('The model file is: ', modelFile)
print('The predict result file is: ', outputfile)
print('The lncRNAs with SNP directory is: ', path)


w = open(outputfile,mode='w+')

motif_len = 24

import AUC
model = load_model(modelFile, custom_objects={'auc': auc})
model_input_len = model.layers[0].input_shape[1]

#path="/share/pub/zhanggs/gaowenyan/DeFine/DeFine_data/RNA-protein/lncRNA-SNP/"
files = os.listdir(path)
for file in files :
    r = open(path+file,'r')
    print(file + " is running!")
    lines = r.readlines()
    for line in lines:
        data = line.strip().split("\t")# one snp
        snpID = data[0]
        w.write(RBP+"\t"+file+"\t"+snpID+"\t")
        for seq in data[1:len(data)]:
            seq2 = seq
            if len(seq2) <= model_input_len:
                seq2 = seq2.center(model_input_len, "N").upper()
            else:
                strip_len = (len(seq2)-model_input_len) / 2
                if int(strip_len) == strip_len:
                    seq2 = seq2.strip()[int(strip_len):-int(strip_len)]
                else:
                    strip_len = int(strip_len)
                    if strip_len == 0:
                        seq2 = seq2[1::]
                    else:
                        seq2 = seq2.strip()[(strip_len+1):-strip_len]
            mat = [one_hot(seq2)]
            mat = np.array(mat)
            mat = [np.transpose(m) for m in mat]
            mat = list(map(lambda x: x.reshape(mat[0].shape[0], mat[0].shape[1], 1), mat))
            mat2 = np.array(mat)
            # predict the sample
            classprobality = model.predict(mat2)
            Class = np.argmax(classprobality) #classify the sample, if the value is 1, binding; otherwise unbinding
            w.write(str(Class)+"\t"+str(classprobality[0][0])+"\t"+str(classprobality[0][1])+";")
        w.write("\n")
    r.close()
    print(file+" is finished!")

w.close()


# seq = "AAGGAAGAGGAGCTGTTCAACAGGAAGCAGGTAGCCACTTAGGCAACCATGGTAGGTGAGGAGTCTGATGTCTTATTGCGCTGTTCAACAGGAAGCAGGTAGCCACTTAGGCAACCATGGTAGGTGAGGAGTCTGATGTCTTATTGCCTAGATGCAGTGATCTGGTAAGTTTCAGTTTATTGACACTATCTAGGAAGCCTGATGGTTGGATGCCTGAGAAAGAAACTGAGCAAAGACAAATGTAACTTCCTCAAGTTTCAAGACTGGGA"

# model =199
# seq = 250
# 250-199=51
# 51/2=26

# def window(sequence, windowSize, step): 
	# arraySize = len(sequence);
	# if windowSize > arraySize:
		# window = [sequence];
	# else:
		# window = [];
		# leftStart = 0;
		# rightEnd = leftStart + windowSize;
		# while rightEnd <= arraySize-1:
			# subsequence = sequence[leftStart:rightEnd];
			# window.append(subsequence);
			# leftStart = leftStart + step;
			# rightEnd = rightEnd+step;
		# if (rightEnd > arraySize-1) & (leftStart < arraySize-1):
			# window.append(sequence[leftStart:arraySize])
	# return window;
