#================================= Create DataSet =================================
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)


p <- 60 #number of nodes
probability <- 0.2
n <- 1000 #number of sample
vars <- c(paste0(1:p))
set.seed(43)

gGtrue <- randomDAG(p, prob = probability, lB = 0.1, uB = 1, V = vars)
N1 <- runif(p, 0.5, 1.0)
Sigma1 <- matrix(0, p, p)
diag(Sigma1) <- N1
eMat <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma1)
gmG <- list(x = rmvDAG(n, gGtrue, errMat = eMat), g = gGtrue)
suffStat <- list(C = cor(gmG$x), n = nrow(gmG$x))

#write dataset
write.table(gmG$x,file="data/dataset.csv", row.names=FALSE, na= "",col.names= FALSE, sep=",")
