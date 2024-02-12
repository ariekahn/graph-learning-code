#!/usr/bin/env Rscript
if (!require("parallel")) install.packages("parallel", repos='http://cran.us.r-project.org'); library("parallel")
if (!require("tidyverse")) install.packages("tidyverse", repos='http://cran.us.r-project.org'); library("tidyverse")
if (!require("magrittr")) install.packages("magrittr", repos='http://cran.us.r-project.org'); library("magrittr")
if (!require("lme4")) install.packages("lme4", repos='http://cran.us.r-project.org'); library("lme4")
if (!require("lmerTest")) install.packages("lmerTest", repos='http://cran.us.r-project.org'); library("lmerTest")
