# ProteinAndRNA #

**get_sequence.py**

**This script could extract peak sequences from the bed file and reference genome fasta file (hg19.fa).**

Note: the necessary modules: pyfasta

	pip install pyfasta

Usage: 

    python get_sequence.py -i TBRG4_IDR -o TBRG4_IDR-sequence -f hg19.fa

For example:
	
	python get_sequence.py -i /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A_HepG2_IDR -o /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A_HepG2_IDR-sequence -f /home/gao/neuralnetwork/coding/2.CNN/example.data/hg19.fa

Note: the hg19.fa file is downloaded from UCSC(https://www.genome.ucsc.edu/). You can also download from  link https://pan.baidu.com/s/1UK3pv_qbvgCrAKT8ywiWcQ and its
extraction code is j41c.

    
**trainCNN.py**

**This script could split raw dataset into train, validation and test datasets. After that, it will train CNN model and return the performances(accuracy, precision, sensitivity, specificity, MCC, AUC ) of models. The performance file is saved in cvperformance.csv. The models with different parameters are saved in models path.**

 
Note: the necessary modules: tensorflow, sklearn, keras, numpy, pandas

Recommandation install:

    conda create -n tensorflow2 python=3.6.5 -y
    conda activate tensorflow2
    conda install scikit-learn=0.19.0 -y
    conda install tensorflow=1.3.0 -y 
	pip install keras==2.2.4
	conda install numpy
	conda install pandas

Usage: 

    python trainCNN.py -p <path> -r <RBP name> -1 <file01> -2 <file02> -f <file> 

For example:
 
	python trainCNN.py -p /home/gao/neuralnetwork/coding/2.CNN/example.data -r ZC3H11A -1 /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A_HepG2_rep01-sequence -2 /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A_HepG2_rep02-sequence -f /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A_HepG2_IDR-sequence


**GetSequenceLogo.py**

**This script could get the motif sequences.**

Usage: 

    python GetSequenceLogo.py -m <modelFile> -o <outputfile> -x <test x file> -y <test y file> -e <test y encoded file>

parameters:

    -m : the model file
    -o : the output motif sequences file.
    -x : the test x file(.npy file), it could get from trainCNN.py results.
    -y : the test y file(.npy file), it could get from trainCNN.py results.
    -e : the test y encoded file(.npy file), it could get from trainCNN.py results.

For example:

    python GetSequenceLogo.py \
       -m /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/models/length227-learning_rate-0.001-weight_decay-0.0001-dropout-0.2-momentum-0.1.hdf5 \
       -o /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/ZC3H11A-sequenceLogo.txt \
	   -x /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/test_x.npy \
	   -y /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/test_y.npy \
	   -e /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/test_y_encoded.npy

**DrawLogoHeatmap.R**

**This script could draw motif heatmap.**


Note: the necessary R packages: ggseqlogo, ggplot2, reshape2, cowplot

Usage: 

    Rscript DrawLogoHeatmap.R <motif sequences file> <figure name>

For example: 

    Rscript DrawLogoHeatmap.R /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/ZC3H11A-sequenceLogo.txt /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/ZC3H11A.motif.png




**RBPAndLncrnaPredict.py**

**This script could predict the protein and lncRNA binding situation.**

Usage: 

	python RBPAndLncrnaPredict.py -r <RBP name> -m <modelFile> -o <outputfile> -d <lncRNA SNP directory>
 
For example:

	python RBPAndLncrnaPredict.py -r ZC3H11A -m /home/gao/neuralnetwork/coding/2.CNN/example.data/ZC3H11A/ZC3H11A-finalmodel.hdf5 -o /home/gao/neuralnetwork/coding/2.CNN/example.data/lncRNA-SNP-predict/classify-result.txt -d /home/gao/neuralnetwork/coding/2.CNN/example.data/lncRNA-SNP/

parameters:
	
	-r : the TF/RBP name.
    -m : the TF/RBP model file has been trained.
    -o : the prediction result file.
    -d : the directory of lncRNAs sequences with SNP.
    


