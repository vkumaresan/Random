# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(plyr)
library(reshape2)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.
debate <- read_csv("../input/debate.csv")
presDebate <- subset(debate, Date == "2016-10-09" & (Speaker == "Clinton" | Speaker == "Trump" | Speaker == "Cooper" | Speaker == "Raddatz"))

table(presDebate$Speaker)

str(presDebate)

presDebate$words <- sapply(presDebate$Text, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
totals <- with( presDebate, aggregate(words ~ Speaker, FUN = sum))
totals <- totals[order(totals$Speaker),]
totals
ggplot(data = totals, aes(x=Speaker, y=words, fill = Speaker))+geom_bar(stat="identity")+ scale_fill_manual(values = c("#4169E1", "#00ff00", "#E91D0E", "#E9E90D"))

totals$wpercent <- round(totals$words/sum(totals$words)*100, digits=1)
totals2 <- count(presDebate$Speaker)
colnames(totals2) <- c("Speaker", "lines")
Total <- join(totals, totals2)
Total$lpercent <- round(Total$lines/sum(Total$lines)*100, digits=1)
Total

percentTotal <- Total[,c(1,3,5)]
melted <- melt(percentTotal, id.var=c("Speaker"))

ggplot(data=melted)+geom_bar(aes(x=variable, y=value, fill = Speaker), stat="identity")+ scale_fill_manual(values = c("#4169E1", "#00ff00", "#E91D0E", "#E9E90D"))


