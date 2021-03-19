# cuPC
cuPC is a CUDA-based parallel implementation of PC-stable algorithm for causal structure learning on GPU. The main highlights of cuPC are as follows:
* Easy usage.
* Compatible with [pcalg](https://cran.r-project.org/web/packages/pcalg/index.html) software.
* 100X to 10,000X speedup compared to serial implementation on CPU.


# Installation
#### CUDA toolkit
To install CUDA toolkit please use [this link](https://developer.nvidia.com/cuda-downloads).

#### R
```
sudo echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -

sudo apt update
sudo apt install r-base r-base-dev
```

#### Linux dependencies
```
sudo apt install libv8-3.14-dev
sudo apt install libcurl4-openssl-dev
sudo apt install libgmp3-dev
```

#### R dependencies
First, enter R by executing the following command:
```
sudo -i R
```

Now inside the R environment, run the following commands:

```
install.packages("tictoc")
source("http://bioconductor.org/biocLite.R")
biocLite(c("graph", "RBGL", "Rgraphviz"))
install.packages("pcalg")
```

#### Compile and execute

* Execute "nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu" to compile .cu files
* A test example exists in use_cuPC.R
* Data_generator.R create gaussian-distributed data

# Publication

Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi, Saber Salehkaleybar, [cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU](https://ieeexplore.ieee.org/document/8823064), IEEE Transactions on Parallel and Distributed Systems (TPDS), Vol. 31, No. 3, March 2020.

#### Abstract
The main goal in many fields in the empirical sciences is to discover causal relationships among a set of variables from observational data. PC algorithm is one of the promising solutions to learn underlying causal structure by performing a number of conditional independence tests. In this paper, we propose a novel GPU-based parallel algorithm, called cuPC, to execute an order-independent version of PC. The proposed solution has two variants, cuPC-E and cuPC-S, which parallelize PC in two different ways for multivariate normal distribution. Experimental results show the scalability of the proposed algorithms with respect to the number of variables, the number of samples, and different graph densities. For instance, in one of the most challenging datasets, the runtime is reduced from more than 11 hours to about 4 seconds. On average, cuPC-E and cuPC-S achieve 500 X and 1300 X speedup, respectively, compared to serial implementation on CPU.

#### Original version of cuPC
The original source code which was employed in the above published article is available at our lab webpage [here](http://lis.ee.sharif.edu/pub/cupc/).

#### Citation
Please cite cuPC in your publications if it helps your research:
```
@article{cupc,
author = {Behrooz Zarebavani and Foad Jafarinejad and Matin Hashemi and Saber Salehkaleybar},
title = {{cuPC}: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU},
journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
year = {2020},
volume = {31},
number = {3},
pages = {530 - 542}
} 
```
