from pyfasta import Fasta
import sys,getopt
 
inputfile = ''
outputfile = ''
fastafile = ''
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:f:",["ifile=","ofile=","fastafile="])
except getopt.GetoptError:
    print("python get_sequence.py -i <inputfile> -o <outputfile> -f <fastafile>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
         print('python get_sequence.py -i <inputfile> -o <outputfile> -f <fastafile>')
         sys.exit()
    elif opt in ("-i", "--ifile"):
         inputfile = arg
    elif opt in ("-o", "--ofile"):
         outputfile = arg
    elif opt in ("-f", "--fastafile"):
         fastafile = arg

print('the input file is: ', inputfile)
print('the output file is: ', outputfile)
print('the fasta file is: ', fastafile)

#f = Fasta('/share/pub/zhanggs/gaowenyan/DeFine/DeFine_data/RNA-protein/hg19.fa')
f = Fasta(fastafile)
#flag = 0
with open(inputfile,"r") as r:
        while True:
                line = r.readline()
                if line!="":
                        list = line.strip().split("\t")
                        sequence = f.sequence({'chr': list[0], 'start': int(list[1]), 'stop': int(list[2])-1, 'strand': list[5]})
                        #flag = flag + 1
                        with open(outputfile,"a") as w:
                                w.write(list[0]+'\t'+list[1]+'\t'+list[2]+'\t'+sequence+'\t'+list[3]+'\t'+list[4]+'\t'+list[5]+'\t'+list[6]+'\t'+list[7]+'\t'+list[8]+'\t'+list[9]+'\n')
                        #print(str(flag)+"\n")
                if not line:
                        break


