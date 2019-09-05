# cuPC
cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU

Please refer to our lab webpage to download the source code:
http://lis.ee.sharif.edu/pub/cupc/

# Publication

Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi, Saber Salehkaleybar, "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU", IEEE Transactions on Parallel and Distributed Systems (TPDS).

# Abstract

The main goal in many fields in the empirical sciences is to discover causal relationships among a set of variables from observational data. PC algorithm is one of the promising solutions to learn underlying causal structure by performing a number of conditional independence tests. In this paper, we propose a novel GPU-based parallel algorithm, called cuPC, to execute an order-independent version of PC. The proposed solution has two variants, cuPC-E and cuPC-S, which parallelize PC in two different ways for multivariate normal distribution. Experimental results show the scalability of the proposed algorithms with respect to the number of variables, the number of samples, and different graph densities. For instance, in one of the most challenging datasets, the runtime is reduced from more than 11 hours to about 4 seconds. On average, cuPC-E and cuPC-S achieve 500 X and 1300 X speedup, respectively, compared to serial implementation on CPU.
Source Code

# Citation

Please cite cuPC in your publications if it helps your research:
```
@article{cupc,
author = {Behrooz Zarebavani and Foad Jafarinejad and Matin Hashemi and Saber Salehkaleybar},
title = {{cuPC}: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU},
journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
year = {2019},
volume = {},
number = {},
pages = {}
doi = {10.1109/TPDS.2019.2939126}
} 
```
