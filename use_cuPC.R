#================================= A test case =================================
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)

source("cuPC.R")

# read data
dataset_path <- file.path("data/dataset.csv", fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",")

# Prepare data
corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))

tic()
stable_fast_fit <- pc(suffStat, indepTest=gaussCItest, p=p, skel.method="stable.fast", alpha=0.05, NAdelete=TRUE)
print("the total time consumed by stable.fast is:")
toc()
print(stable_fast_fit)


tic()
cuPC_fit <- cu_pc(suffStat, p=p, alpha=0.05)
print("The total time consumed by cuPC is:")
toc()


if (require(Rgraphviz)) {
  ## show estimated CPDAG
  par(mfrow=c(1,2))
  plot(stable_fast_fit, main = "Estimated CPDAG (stable.fast)")
  plot(cuPC_fit, main = "Estimated CPDAG (cuPC)")
}