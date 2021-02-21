#Draw Logo Heatmap.

#https://www.plob.org/article/12630.html
#https://www.jianshu.com/p/00aaf3ab5926
#conda activate R3.6.0-env
library(ggseqlogo)
library(ggplot2)
library(reshape2)
library(cowplot)

args <-  commandArgs(trailingOnly = TRUE)
#RBPfile<-read.table("D:/2019/找motif/HepG2-TF-9.txt")
#RBPfile<-as.vector((as.matrix(RBPfile)))
#for(RBP in RBPfile){
con <- file(args[1], "r")
seq<-c()
line<-readLines(con,n=1)
seq<-c(seq, line)
while( length(line) != 0 ) {
line<-readLines(con,n=1)
seq<-c(seq, line)
}
close(con)
#ggseqlogo(seq,method="bits")
###########################################
mat<-matrix(0,ncol=24,nrow=4)
rownames(mat)<-c("A","U","C","G")
for(s in seq){
a<-unlist(strsplit(s,''))
mat[1,grep("A", a, ignore.case = T)]<-mat[1,grep("A", a, ignore.case = T)]+1
mat[2,grep("U", a, ignore.case = T)]<-mat[2,grep("U", a, ignore.case = T)]+1
mat[3,grep("C", a, ignore.case = T)]<-mat[3,grep("C", a, ignore.case = T)]+1
mat[4,grep("G", a, ignore.case = T)]<-mat[4,grep("G", a, ignore.case = T)]+1
}

# library(reshape2)
mat.m <- melt(mat) 
p1<-ggplot()+geom_logo(mat)+theme_logo()+#,method="probability",rev_stack_order = T
theme(plot.margin=unit(c(0,0,0,0),"mm"),
	  axis.title = element_blank(),
	  axis.text.x= element_blank(),
	  axis.text.y = element_blank())+
scale_y_continuous(expand = c(0,0))+
scale_x_discrete(expand = c(0, 0))

p2 <- ggplot(mat.m, aes(x=Var2, y=Var1)) +
geom_tile(aes(fill = value),colour = "white") + 
scale_fill_gradient(low = "#FAF0E6",high = "red")+
theme(plot.margin=unit(c(0,0,0,0),"mm"),
	  plot.background = element_blank(),
	  axis.title = element_blank(),
	  axis.text.x= element_blank(),
	  axis.ticks.x = element_blank(),
	  axis.ticks.y = element_blank(),
	  axis.text.y = element_text(face="bold", color=c("#0A9344","#D62839","#255C99","#F7B229"), size=16),
	  legend.position = "none",
	  panel.background = element_blank())+
scale_y_discrete(expand = c(0,0),labels=c("A","U","C","G"))+
scale_x_discrete(expand = c(0, 0)) 

#library(cowplot)
png(args[2],height = 200,width=700)
print(plot_grid(p1,p2,ncol = 1, align = "v",axis="none",
		hjust = 0, vjust = 0,rel_heights = c(2,1)))
#在循环中，一定要有print
dev.off()

# maxValue<-apply(mat,2,max)
# getName<-function(x){
# maxValue<-max(x)
# maxPosition<-which(x %in% maxValue)
# return(maxPosition[1])
# }

# maxPosition<-unlist(apply(mat,2,getName))
# label<-rownames(mat)[maxPosition]
# sequence<-paste(label,collapse="")

# write.table(paste(RBP,sequence,sep="\t"),"D:/2019/找motif/motif-sequence.txt",
		  # append = TRUE,row.names = F,col.names = F, quote=F)

#}

# library(ggseqlogo)
# library(ggplot2)

#RBP<-"EIF3D"
# data<-data.frame(position=1:24,label=label,value=maxValue)
# ggplot(data,aes(x=position,y=value,label=label))+
#    #geom_label(aes(fill = factor(label)),colour = "white", fontface = "bold")+
#    #geom_text(aes(size = value)) + scale_radius(range = c(3,6))
#    geom_logo(seq,method="probability",rev_stack_order = TRUE)
# 
# 

