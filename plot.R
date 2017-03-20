tmp = seq(11,1,by=-1)
library(ggplot2)
library(RColorBrewer)
library(reshape2)
library(gridExtra)
library(dplyr)

# heatmap
gap_safe = read.delim("gap_safe_screen.txt",sep=" ",header=F)
dynamic_safe = read.delim("dynamic_safe.txt",sep=" ",header=F)
dst3 = read.delim("dst3.txt",sep=" ",header=F)
static_safe = read.delim("static_safe.txt",sep=" ",header=F)

plot_heatmap<-function(s_data,caption){
  s_data = s_data[,1:10]
  s_data = s_data/10000
  tmp = seq(10,1,by=-1)
  tmp_data = as.matrix(s_data[,tmp])
  colnames(tmp_data) = 1:10
  tmp_data_melted = melt(tmp_data)
  hm.palette <- colorRampPalette(rev(brewer.pal(11, 'Spectral')), space='Lab') 
  p1<-ggplot(tmp_data_melted,aes(x=Var1,y=Var2,fill=value))+
    geom_tile()+
    coord_fixed(0.5)+
    scale_fill_gradientn(colors = hm.palette(100)) +
    scale_x_continuous(labels=c("0"="0","5"="1","10"="2","13"="2.5")) + 
    scale_y_continuous(labels=c("0"="10","3"="8","6"="5","9"="3","12"="0")) + 
    labs(x=expression(paste(-log[10],"(",lambda/lambda[max],")")),y=expression(log[2](K)))+
    theme(legend.position = "none")+
    geom_text(x=10,y=9,label=caption,color="white")+
    theme(legend.title = element_blank())
  return(p1)
}

p1<-list(plot_heatmap(static_safe,"STATIC SAFE"),
         plot_heatmap(dynamic_safe,"DYNAMIC SAFE"),
         plot_heatmap(dst3,"DST3"),
         plot_heatmap(gap_safe,"GAP SAFE"))
tmp <- ggplot_gtable(ggplot_build(p1[[1]] + theme(legend.position = "bottom")))
leg <- which(sapply(tmp$grobs, function(x) x$name) ==  "guide-box")
p1[[length(p1)+1]] <- tmp$grobs[[leg]]
lay<-matrix(1:(length(p1)-1),2,byrow=T) %>% rbind(rep(length(p1),2))
pdf("screening.pdf",width=6,height=6) #use jpeg instead
grid.arrange(grobs=p1,ncol=2,layout_matrix=lay,heights=unit(c(4,4,1),"cm"))
dev.off()

# time 
time_passed = read.delim("time_passed.txt",sep=" ",header=F)
tmp = c(1,6,3,4,2)

time_passed = time_passed[,tmp]
rownames(time_passed)<-c("4","6","8")
colnames(time_passed)<-c("NO SCREENING","STATIC SAFE","DYNAMIC SAFE","DST3","GAP SAFE")
time_passed_melt = melt(as.matrix(time_passed))
p1<-ggplot(time_passed_melt) + 
      geom_bar(aes(x=factor(Var1),y=value,fill=Var2),stat="identity",position="dodge") + 
      guides(fill = guide_legend(title="Screening Rule")) + 
      labs(x=expression(paste(-log[10],"(","duality gap",")")),y="Time (s)")

pdf("time.pdf",width=6,height=4)
p1
dev.off()



