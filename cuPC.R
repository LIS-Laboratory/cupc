library(pcalg)


cu_pc <- function(suffStat, indepTest, alpha, labels, p,
                      fixedGaps = NULL, fixedEdges = NULL, NAdelete = TRUE, m.max = Inf,
                      u2pd = c("relaxed", "rand", "retry"),
                      skel.method = c("stable", "original", "stable.fast"),
                      conservative = FALSE, maj.rule = FALSE,
                      solve.confl = FALSE, verbose = FALSE)
{ 
  ## Initial Checks
  cl <- match.call()
  if(!missing(p)) stopifnot(is.numeric(p), length(p <- as.integer(p)) == 1, p >= 2)
  if(missing(labels)) {
    if(missing(p)) stop("need to specify 'labels' or 'p'")
    labels <- as.character(seq_len(p))
  } else { ## use labels ==> p  from it
    stopifnot(is.character(labels))
    if(missing(p)) {
      p <- length(labels)
    } else if(p != length(labels))
      stop("'p' is not needed when 'labels' is specified, and must match length(labels)")
    else
      message("No need to specify 'p', when 'labels' is given")
  }
  seq_p <- seq_len(p)
  
  u2pd <- match.arg(u2pd)
  skel.method <- match.arg(skel.method)
  if(u2pd != "relaxed") {
    if (conservative || maj.rule)
      stop("Conservative PC and majority rule PC can only be run with 'u2pd = relaxed'")
    
    if (solve.confl)
      stop("Versions of PC using lists for the orientation rules (and possibly bi-directed edges)\n can only be run with 'u2pd = relaxed'")
  }
  
  if (conservative && maj.rule) stop("Choose either conservative PC or majority rule PC!")
  
  ## Skeleton
  skel <- cu_skeleton(suffStat, indepTest, alpha, labels=labels, NAdelete=NAdelete, m.max=m.max, verbose=verbose)
  skel@call <- cl # so that makes it into result
  
  ## Orient edges
  if (!conservative && !maj.rule) {
    switch (u2pd,
            "rand" = udag2pdag(skel),
            "retry" = udag2pdagSpecial(skel)$pcObj,
            "relaxed" = udag2pdagRelaxed(skel, verbose=verbose))
    
  }
  else { ## u2pd "relaxed" : conservative _or_ maj.rule
    
    ## version.unf defined per default
    ## Tetrad CPC works with version.unf=c(2,1)
    ## see comment on pc.cons.intern for description of version.unf
    pc. <- pc.cons.intern(skel, suffStat, indepTest, alpha,
                          version.unf=c(2,1), maj.rule=maj.rule, verbose=verbose)
    udag2pdagRelaxed(pc.$sk, verbose=verbose,
                     unfVect=pc.$unfTripl)
  }
}


cu_skeleton <- function(suffStat, indepTest, alpha, labels, p, m.max = Inf, NAdelete = TRUE, verbose = FALSE)
{ 
  cl <- match.call()
  if(!missing(p)) stopifnot(is.numeric(p), length(p <- as.integer(p)) == 1, p >= 2)
  if(missing(labels)) {
    if(missing(p)) stop("need to specify 'labels' or 'p'")
    labels <- as.character(seq_len(p))
  } else { ## use labels ==> p  from it
    stopifnot(is.character(labels))
    if(missing(p)) {
      p <- length(labels)
    } else if(p != length(labels))
      stop("'p' is not needed when 'labels' is specified, and must match length(labels)")
    else
      message("No need to specify 'p', when 'labels' is given")
  }
  
  seq_p <- seq_len(p)
  pval <- NULL
  #Convert SepsetMatrix to sepset
  sepset <- lapply(seq_p, function(.) vector("list",p))# a list of lists [p x p]
  # save maximal p value
  pMax <- matrix(0, nrow = p, ncol = p)
  number_of_levels = 50
  threshold <- matrix(0, nrow = 1, ncol = number_of_levels)
  for (i in 0:(min(number_of_levels, suffStat$n - 3) - 1)){
    threshold[i] <- abs(qnorm((alpha/2), mean = 0, sd = 1)/sqrt(suffStat$n - i - 3))  
  }

  G <- matrix(TRUE, nrow = p, ncol = p)
  diag(G) <- FALSE
  done <- TRUE
  ord <- 0
  G <- G * 1

  if (m.max == Inf){
    max_level = 14
  } else{
    max_level = m.max
  }
  
  sepsetMatrix <- matrix(-1, nrow = p * p, ncol = 14)
  dyn.load("Skeleton.so")

  start_time <- proc.time()
  z <- .C("Skeleton",
        C = as.double(suffStat$C),
        p = as.integer(p),
        G = as.integer(G),
        Th = as.double(threshold),
        l = as.integer(ord),
        max_level = as.integer(max_level),
        pmax = as.double(pMax),
        sepsetmat = as.integer(sepsetMatrix))
  
  ord <- z$l
  G <- (matrix(z$G, nrow = p, ncol = p)) > 0
  
  pMax <- (matrix(z$pmax, nrow = p, ncol = p)) 
  pMax[which(pMax == -100000)] <- -Inf
  if(ord <= 14){
    sepsetMatrix <- t(matrix(z$sepsetmat, nrow = 14, ncol = p ^ 2))	  
    index_of_cuted_edge <- row(sepsetMatrix)[which(sepsetMatrix != -1)]
    for (i in index_of_cuted_edge) {
      y <- floor(i / p) + 1
      x <- i - ((y - 1) * p) + 1
      #find index
      j <- 1
      for (j in 1:14){
        if (sepsetMatrix[i, j] == -1){
          j <- j - 1
          break
        }
      }
      sepset[[x]][[y]] <- sepset[[y]][[x]] <- sepsetMatrix[i, 1:j]
    }
  } else{
    # TODO: Update sepset for more than 14 level
  }
  print(ord)
  ## transform matrix to graph object :
  Gobject <-
    if (sum(G) == 0) {
      new("graphNEL", nodes = labels)
    } else {
      colnames(G) <- rownames(G) <- labels
      as(G,"graphNEL")
    }
  ## final object
  new("pcAlgo", graph = Gobject, call = cl, n = integer(0),
      max.ord = as.integer(ord - 1), n.edgetests = 0,
      sepset = sepset, pMax = pMax, zMin = matrix(NA, 1, 1))

}## end{ skeleton }
