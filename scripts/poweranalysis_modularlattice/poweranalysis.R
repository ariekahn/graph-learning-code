#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library("parallel")
library("tidyverse")
library("magrittr")
library("lme4")
library("lmerTest")

# Load datasets and combine

base.dir <- args[1]
out.path <- args[2]
n.bootstrap.samples <- args[3]
n.threads <- args[4]
print(out.path)
print(n.bootstrap.samples)
out.filename <- paste0(out.path, "/powertest.", n.bootstrap.samples, ".RData")

# Load in our old data as well, to keep processing consistent
load(paste(base.dir, "preprocessed/task.Rdata", sep="/"))
task.orig <- task
rm(task)


# Pretty table confint function

get_confint <- function(model) {
  return(get_confint_summary(summary(model)))
}

get_confint_summary <- function(summary) {
  coefficients <- summary$coefficients
  
  lb <- coefficients[,c("Estimate")] - 1.96*coefficients[,c("Std. Error")]
  ub <- coefficients[,c("Estimate")] + 1.96*coefficients[,c("Std. Error")]
  
  coef.start <- coefficients[,1:2]
  coef.end <- coefficients[,3:dim(coefficients)[[2]]]
  
  coef.start <- cbind(coef.start, lb)
  colnames(coef.start)[[length(colnames(coef.start))]] <- "95% Conf. Int."
  
  coef.start <- cbind(coef.start, ub) 
  colnames(coef.start)[[length(colnames(coef.start))]] <- ""
  
  coef.start <- cbind(coef.start, coef.end)
  
  
  coef.start <- as_tibble(coef.start, rownames="Variable")
  
  coef.start <-
    coef.start %>%
    mutate(Sig. = if_else(`Pr(>|t|)` < 0.1, ".", "")) %>%
    mutate(Sig. = if_else(`Pr(>|t|)` < 0.05, "*", Sig.)) %>%
    mutate(Sig. = if_else(`Pr(>|t|)` < 0.01, "**", Sig.)) %>%
    mutate(Sig. = if_else(`Pr(>|t|)` < 0.001, "***", Sig.))
  
  colnames(coef.start)[[1]] <- " "
  colnames(coef.start)[[5]] <- " "
  
  return(coef.start)
}

# START HERE

# Table: Modular-Lattice Effect Modeling
# Old data, experiment 1

modularlattice.data <- task.orig %>%
  filter(valid == TRUE) %>%
  filter(format == "walktest") %>%
  filter(stage == 1) %>%
  filter(graph != "Random")

head(modularlattice.data)
  
#surprisal.model <- lmer(rt ~ target + z.log.trial*transition_in_1 + (1 + z.log.trial*transition_in_1 | subject), 
#                        modularlattice.data)


# Quick power analysis for N=15

compute.sample <- function(x, n.samples) {
  subjects.modularlattice.data <- unique(modularlattice.data$subject)
  subjects.sample <- sample(subjects.modularlattice.data, n.samples, replace=TRUE)
  sample.data.list <- list()
  for (i in 1:n.samples) {
    sample.data.list[[i]] <- modularlattice.data %>%
      filter(subject == subjects.sample[i]) %>%
      mutate(subject.num = i)
  }
  sample.data <- bind_rows(sample.data.list)
  sample.modularlattice.model <- lmer(rt ~ target + z.log.trial*graph + (1 + z.log.trial*graph | subject.num), 
                                 sample.data)
  return(get_confint(sample.modularlattice.model)[17,1:8])
}

#n.sample.runs <- 10000
#cl <- makeForkCluster(nnodes=n.threads)
#clusterExport(cl, list("get_confint", "get_confint_summary", "compute.sample", "modularlattice.data"))
#clusterEvalQ(cl, library(tidyverse))
#clusterEvalQ(cl, library(lmerTest))
#sample.results <- parSapply(cl, 1:n.sample.runs, compute.sample, n.bootstrap.samples)
#save(sample.results, file=out.filename)
#stopCluster(cl)
