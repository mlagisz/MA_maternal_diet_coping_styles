### ML script for MA on the effects of maternal nutrition on offspring's coping styles

### STEP 4. setting parallel runs for MCMCglmm models (will not work on Rstudio)

# IMPORTANT
# Uncomment the desired block of code and then 
# Run this script in the Terminal:
# R
# setwd(" ")
# source("MA_step4b_parallel_runs_lnRR.R")

options(scipen=100)
rm(list=ls())

#getRversion()
#install.packages(c("MCMCglmm", "foreach", "doMC"))
library(Matrix)
library(lme4)
library(MCMCglmm)
library(foreach)
library(matrixcalc)
library(doMC)
registerDoMC()

load("data/Z_data_list_lnRR.Rdata") ####upload data in a list format: for the 9 subsets

Z_data_list[[2]] <- NULL #exclude subset data2 from the data list - carefull the indexes will shuffle by 1!!!
#dim(Z_data_list)
#Zdata <- Z_data_list[[4]] #select one subset, for example

# ###########################################
# ###### START TEST 1 - single subset, single chain, null model
# df <- Z_data_list[[7]]
# names(df)
# head(df)
# 
# ### WITHOUT variance-covariance matrix
# d <- df$lnRR
# vofd <- df$lnRR_Var
# prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
# model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=df,verbose=T,mev=vofd,nitt=40000,thin=4000,burnin=4000,pr=T,prior=prior1)
# summary(model)
# 
# ### WITH variance-covariance matrix
# d <- df$lnRR
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- df[p1,"varianceofcontrols"] / (df[p1,"con_n"] * (df[p1,"con_mean"]^2)) #(CVar / (CN * (CMean^2)))
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$lnRR_Var # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
# model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=Z_data_list[[1]],ginverse=list(comp_ID=AnivG),verbose=T,nitt=40000,thin=4000,burnin=4000,pr=T,prior=prior2)
# summary(model)
###### END TEST 1

# ###########################################
# 
# # ###### START TEST 2 - with external loop for data subset - non-parallel test:
# df <- Z_data_list[[6]]
# names(df)
# 
# ### WITHOUT variance-covariance matrix
# chains <- 1:3 #3 chains for each model
# prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {
#     filename <- paste("runs/M1.",i,"_test_lnRR.Rdata",sep="") # Remember to change the number so that files have different names.
#     df <- Z_data_list[[2]]
#     d <- df$lnRR
#     vofd <- df$lnRR_Var
#     model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=df,verbose=T,mev=vofd,nitt=4000,thin=40,burnin=40,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
### WITH variance-covariance matrix
# chains <- 1:3 #3 chains for each model
# prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
# multi.MCMC <- system.time({
#     models <- foreach(i = chains) %do% {
#       filename <- paste("runs/M1.",i,"_sub_lnRR7.Rdata",sep="") # Remember to change the number so that files have different names.
#       #filename <- paste("runs/M1.",i,"test.Rdata",sep="") # Remember to change the number so that files have different names.
#       df <- Z_data_list[[6]] #runs for 1 to 5 and for 8, does not run for 6 and 7 (data6 and data7)
#       d <- df$lnRR
#       m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#       rownames(m) <- df$comp_ID 
#       colnames(m) <- df$comp_ID 
#       shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#       combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#       # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#       store <- 0
#       for (i in 1:dim(combinations)[1]){
#         p1 <- combinations[i,1]
#         p2 <- combinations[i,2]
#         p1_p2_cov <- df[p1,"varianceofcontrols"] / (df[p1,"con_n"] * (df[p1,"con_mean"]^2)) #(CVar / (CN * (CMean^2)))
#         m[p1,p2] <- p1_p2_cov
#         m[p2,p1] <- p1_p2_cov
#         store[i] <- p1_p2_cov
#       }
#       # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#       diag(m) <- df$lnRR_Var  # m is the variance-covariance matrix to be used in the test run
# 
#       dim(m)
#       is.positive.definite(m)
#       is.symmetric.matrix(m)
#       table(is.na(m)) #if contains NA value
#       #which( is.na(m), arr.ind=T ) #comp613
#       #df[df$comp_ID=="comp_613",]
#       eigen(m)$values 
#       range(eigen(m)$values) #check whether some eigenvalues < 0      
#       
#       range(store)
#       range(df$lnRR_Var)
#       
#       AinvG <- solve(m)
#       AnivG <- as(AinvG,"dgCMatrix")
#       model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=40000,thin=4000,burnin=4000,pr=T,prior=prior2)
#       #model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#       save(model, file = filename)
#     }
#   })
#   multi.MCMC
###### END TEST 2


# ######################################################################################
# ### Null models - study_ID and animal_ID as random effects (Model 1), controlling for shared controls subsets 1-5 and 8 (data1, data3-6, data9)
# ######################################################################################
# 
# #We dont use species or strain as random effect, because there are only 2 species and 4 strains! 
# #We will use strains as fixed effects in the Strain and Full models instead!
# 
# prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
# 
# subsets <- c(1:5,8) #make external loop to go over the 8 subsets in the data list
# chains <- 1:3 #3 chains for each model
# #df <- Z_data_list[[6]]
# 
# multi.MCMC <- system.time({
#   models6 <- foreach(j = subsets) %dopar% {
#     df <- Z_data_list[[j]]
#     d <- df$lnRR
#     m <- matrix(0, nrow = dim(df)[1], ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#     rownames(m) <- df$comp_ID 
#     colnames(m) <- df$comp_ID 
#     shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#     combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#     # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#     for (i in 1:dim(combinations)[1]){
#       p1 <- combinations[i,1]
#       p2 <- combinations[i,2]
#       p1_p2_cov <- df[p1,"varianceofcontrols"] / (df[p1,"con_n"] * (df[p1,"con_mean"]^2)) #(CVar / (CN * (CMean^2)))
#       m[p1,p2] <- p1_p2_cov
#       m[p2,p1] <- p1_p2_cov
#     }
#     # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#     diag(m) <- df$lnRR_Var # m is the variance-covariance matrix to be used in the test run
#     AinvG <- solve(m)
#     AnivG <- as(AinvG,"dgCMatrix")
#     models <- foreach(i = chains) %dopar% {
#       filename <- paste("runs/M1.",i,"_sub_lnRR",j,".Rdata",sep="") # Remember to change the number so that files have different names.
#       model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#       save(model, file = filename)
#     }
#   }
# })
# multi.MCMC

######################################################################################
### Null models - study_ID and animal_ID as random effects (Model 1), WITHOUT controlling for shared controls subsets 6-7 (data7, data8)
######################################################################################

#We dont use species or strain as random effect, because there are only 2 species and 4 strains! 
#We will use strains as fixed effects in the Strain and Full models instead!

prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior

subsets <- c(6,7) #make external loop to go over the 8 subsets in the data list
chains <- 1:3 #3 chains for each model

# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {
#     filename <- paste("runs/M1.",i,"_test_lnRR.Rdata",sep="") # Remember to change the number so that files have different names.
#     df <- Z_data_list[[2]]
#     d <- df$lnRR
#     vofd <- df$lnRR_Var
#     model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=df,verbose=T,mev=vofd,nitt=4000,thin=40,burnin=40,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })

multi.MCMC <- system.time({
  models2 <- foreach(j = subsets) %dopar% {
    df <- Z_data_list[[j]]
    d <- df$lnRR
    vofd <- df$lnRR_Var
      models <- foreach(i = chains) %dopar% {
      filename <- paste("runs/M1.",i,"_sub_lnRR",j,".Rdata",sep="") # Remember to change the number so that files have different names.
      model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=df,verbose=T,mev=vofd,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior1)
      save(model, file = filename)
    }
  }
})
multi.MCMC


# ######################################################################################
# ### Strain models - strain_ID as fixed effect, study_ID and animal_ID as random effects (Model 2), controlling for shared controls
# ######################################################################################
# 
#  prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
#  chains <- 1:3 #3 chains for each model
# 
# multi.MCMC <- system.time({
#     df <- Z_data_list[[6]]
#     d <- df$H.d
#     m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#     rownames(m) <- df$comp_ID 
#     colnames(m) <- df$comp_ID 
#     shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#     combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#     # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#     for (i in 1:dim(combinations)[1]){
#       p1 <- combinations[i,1]
#       p2 <- combinations[i,2]
#       p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#       m[p1,p2] <- p1_p2_cov
#       m[p2,p1] <- p1_p2_cov
#     }
#     # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#     diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
#     AinvG <- solve(m)
#     AnivG <- as(AinvG,"dgCMatrix")
#     
#     models <- foreach(i = chains) %dopar% {
#       filename <- paste("runs/M2.",i,"_sub6.Rdata",sep="") # Remember to change the number so that files have different names
#       model <- MCMCglmm(d~strain-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#       save(model, file = filename)
#     }
# })
# multi.MCMC
# 
# 
# multi.MCMC <- system.time({
#   df <- Z_data_list[[7]]
#   d <- df$H.d
#   m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#   rownames(m) <- df$comp_ID 
#   colnames(m) <- df$comp_ID 
#   shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#   combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#   # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#   for (i in 1:dim(combinations)[1]){
#     p1 <- combinations[i,1]
#     p2 <- combinations[i,2]
#     p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#     m[p1,p2] <- p1_p2_cov
#     m[p2,p1] <- p1_p2_cov
#   }
#   # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#   diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
#   AinvG <- solve(m)
#   AnivG <- as(AinvG,"dgCMatrix")
#   
#   models <- foreach(i = chains) %dopar% {
#     filename <- paste("runs/M2.",i,"_sub7.Rdata",sep="") # Remember to change the number so that files have different names
#     model <- MCMCglmm(d~strain-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# multi.MCMC <- system.time({
#   df <- Z_data_list[[8]]
#   d <- df$H.d
#   m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#   rownames(m) <- df$comp_ID 
#   colnames(m) <- df$comp_ID 
#   shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#   combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#   # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#   for (i in 1:dim(combinations)[1]){
#     p1 <- combinations[i,1]
#     p2 <- combinations[i,2]
#     p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#     m[p1,p2] <- p1_p2_cov
#     m[p2,p1] <- p1_p2_cov
#   }
#   # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#   diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
#   AinvG <- solve(m)
#   AnivG <- as(AinvG,"dgCMatrix")
#   
#   models <- foreach(i = chains) %dopar% {
#     filename <- paste("runs/M2.",i,"_sub8.Rdata",sep="") # Remember to change the number so that files have different names
#     model <- MCMCglmm(d~strain-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 

# #######################################################################################
# ### Full models: sex, choice, age, timing  etc.  (Model 3), controlling for shared controls
# ######################################################################################
# 
#  prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
#  
#  chains <- 1:3 #3 chains for each model
# 
# df <- Z_data_list[[1]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#     models <- foreach(i = chains) %do% {  
#       filename <- paste("runs/M3.",i,"_sub1.Rdata",sep="") # Remember to change the number so that files have different names.
#       #model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#       #removed last fixed effect due to model not converging well, rerun:
#       model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#       save(model, file = filename)
#     }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[2]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub2.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[3]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# #levels(df$sex)
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub3.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[4]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub4.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~Z_nom_manip_val-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[5]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub5.Rdata",sep="") # Remember to change the number so that files have different names.
#     #model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     #removed last fixed effect due to model not converging well, rerun:
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[6]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub6.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP+Z_dam_diet_end_dPC-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[7]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub7.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# df <- Z_data_list[[8]]
# d <- df$H.d
# 
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# df <- within(df, sex <- relevel(sex, ref = which(levels(df$sex)=="m"))) #set the reference level for factor sex at males
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("runs/M3.",i,"_sub8.Rdata",sep="") # Remember to change the number so that files have different names.
#     #model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP+Z_dam_diet_end_dPC-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     #re-run without the last moderator:
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 















###### ALL BELOW ARE WITHOUT CONTROLLING FOR SHARED CONTROL

# ######################################################################################
# ### Null models - study_ID and animal_ID as random effects (model1)
# ######################################################################################
# 
# #We dont use species or strain as random effect, because there are only 2 species and 4 strains! 
# #We will use strains as fixed effects in the Strain and Full models instead!
# 
# prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
# 
# subsets <- 1:8 #make external loop to go over the 8 subsets in the data list
# chains <- 1:3 #3 chains for each model
#
# multi.MCMC <- system.time({
#   models8 <- foreach(j = subsets) %dopar% {
#     models <- foreach(i = chains) %dopar% {
#       filename <- paste("ML_runs/M1.",i,"_sub",j,".Rdata",sep="") # Remember to change the number so that files have different names.
#       d <- Z_data_list[[j]]$H.d
#       vofd <- Z_data_list[[j]]$SE.d
#       model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=Z_data_list[[j]],verbose=T,mev=vofd,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior1)
#       save(model, file = filename)
#     }
#   }
# })
# multi.MCMC


# ######################################################################################
# ### Strain models - strain_ID as fixed effect, study_ID and animal_ID as random effects (model2)
# ######################################################################################
# 
# prior1<-list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
#  
# subsets <- 6:8 #make external loop to go over the 6,7,8 subsets in the data list
# chains <- 1:3 #3 chains for each model
# 
# multi.MCMC <- system.time({
#   models8 <- foreach(j = subsets) %dopar% {
#     models <- foreach(i = chains) %dopar% {
#       filename <- paste("ML_runs/M2.",i,"_sub",j,".Rdata",sep="") # Remember to change the number so that files have different names.
#       d<-Z_data_list[[j]]$H.d
#       vofd<-Z_data_list[[j]]$SE.d
#       model <- MCMCglmm(d~strain-1,random=~study_ID+animal_ID,family="gaussian",data=Z_data_list[[j]],verbose=T,mev=vofd,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior1)
#       save(model, file = filename)
#     }
#   }
# })
# multi.MCMC
# 
# # #single test run
# # d<-Z_data_list[[6]]$H.d
# # vofd<-Z_data_list[[6]]$SE.d
# # model <- MCMCglmm(d~strain-1,random=~study_ID+animal_ID,family="gaussian",data=Z_data_list[[6]],verbose=T,mev=vofd,nitt=40000,thin=400,burnin=400,pr=T,prior=prior1)
# # summary(model)


#  
# # #######################################################################################
# # ### Full models: sex, choice, age, timing  etc.  (model3)
# # ######################################################################################
# 
# prior1<-list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
# 
# chains <- 1:3 #3 chains for each model
# 
# data<-Z_data_list[[1]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#     models <- foreach(i = chains) %do% {  
#       filename <- paste("ML_runs/M3.",i,"_sub1.Rdata",sep="") # Remember to change the number so that files have different names.
#       model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#       save(model, file = filename)
#     }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[2]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub2.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[3]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub3.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[4]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub4.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~Z_nom_manip_val-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[5]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub5.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[6]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub6.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP+Z_dam_diet_end_dPC-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[7]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub7.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 
# 
# data<-Z_data_list[[8]]
# H.d<-data$H.d
# SE.d<-data$SE.d
# #which(contrasts(data$sex)==contrasts(data$sex)["m",])
# contrasts(data$sex)<-contr.treatment(levels(data$sex),base=which(contrasts(data$sex)==contrasts(data$sex)["m",])) #making males the reference sex
# multi.MCMC <- system.time({
#   models <- foreach(i = chains) %do% {  
#     filename <- paste("ML_runs/M3.",i,"_sub8.Rdata",sep="") # Remember to change the number so that files have different names.
#     model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC+Z_response_age_dPP+Z_dam_diet_end_dPC-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=4000000,thin=400,burnin=4000,pr=T,prior=prior1)
#     save(model, file = filename)
#   }
# })
# multi.MCMC
# 

# # #single test run
# H.d<-data$H.d
# SE.d<-data$SE.d
# model <- MCMCglmm(H.d~factor(sex)+Z_nom_manip_val+Z_dam_diet_start_dPC-1,random=~study_ID+animal_ID,family="gaussian",data=data,verbose=T,mev=SE.d,nitt=40000,thin=40,burnin=400,pr=T,prior=prior1)
# # summary(model)
# # 
# # #test run
# # #m2.0<-MCMCglmm(d~ factor(Offspr_sex) + factor(Dam_choice_diet) + factor(Offspr_choice_diet) + Z_TBodyweight_age_dPC + Z_Dam_diet_start + Z_Dam_diet_end,random=~Study_ID+Strain,family="gaussian",data=fa,verbose=T,mev=vofd,nitt=200000,thin=25,burnin=25000,pr=T,prior=prior1)
# # 
# # chains <- 1:3
# # 
# # multi.MCMC <- system.time({
# #   models <- foreach(i = chains) %dopar% {
# #     CSmodel <- MCMCglmm(d~factor(Offspr_sex) + factor(Dam_choice_diet) + factor(Offspr_choice_diet) + Z_TBodyweight_age_dPC + Z_Dam_diet_start + Z_Dam_diet_end,random=~Strain+Study_ID,family="gaussian",data=fa,verbose=T,mev=vofd,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior1)
# #     filename <- paste("OF_model2.",i,"_BW.Rdata",sep="") # Remember to change the number so that files have different names.
# #     save(CSmodel, file = filename)
# #   }
# # })
# # 
# # multi.MCMC



# # ################################################################################################################
# # ######################################## PARAMETER-EXPANDED RUNS - test

# ###### START TEST 1 - single subset, single chain, null model
# df <- Z_data_list[[1]]
# names(df)
# 
# ### WITHOUT variance-covariance matrix
# d <- df$H.d
# vofd <- df$H.d_Var_combined
# prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002))) #inverse-Wishart prior
# prior1 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000),G2=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000))) #parameter-expanded prior
# model <- MCMCglmm(d~1,random=~study_ID+animal_ID,family="gaussian",data=df,verbose=T,mev=vofd,nitt=40000,thin=4000,burnin=4000,pr=T,prior=prior1)
# summary(model)
# 
# ### WITH variance-covariance matrix
# d <- df$H.d
# m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
# rownames(m) <- df$comp_ID 
# colnames(m) <- df$comp_ID 
# shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
# combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
# # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
# for (i in 1:dim(combinations)[1]){
#   p1 <- combinations[i,1]
#   p2 <- combinations[i,2]
#   p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#   m[p1,p2] <- p1_p2_cov
#   m[p2,p1] <- p1_p2_cov
# }
# # add the diagonal - use df$H.d_Var_combined as matrix diagonal
# diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
# AinvG <- solve(m)
# AnivG <- as(AinvG,"dgCMatrix")
# prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior
# prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000),G2=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000),G3=list(V=1,fix=1))) #parameter-expanded prior
# 
# model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=Z_data_list[[1]],ginverse=list(comp_ID=AnivG),verbose=T,nitt=40000,thin=4000,burnin=4000,pr=T,prior=prior2)
# summary(model)
# ###### END TEST 1
# 

######################################################################################
# ### Null models - study_ID and animal_ID as random effects (Model 1), controlling for shared controls - with parameter-expanded prior!
# ######################################################################################
# 
# #We dont use species or strain as random effect, because there are only 2 species and 4 strains! 
# #We will use strains as fixed effects in the Strain and Full models instead!
# 
# prior2e <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000),G2=list(V=1,nu=.002,alpha.mu=0,alpha.V=1000),G3=list(V=1,fix=1))) #inverse-Wishart prior
# #prior2 <- list(R=list(V=1,nu=.002),G=list(G1=list(V=1,nu=.002),G2=list(V=1,nu=.002),G3=list(V=1,fix=1))) #inverse-Wishart prior - original, non-expanded
# 
# subsets <- 1:8 #make external loop to go over the 8 subsets in the data list
# chains <- 1:3 #3 chains for each model
# 
# multi.MCMC <- system.time({
#   models8 <- foreach(j = subsets) %dopar% {
#     df <- Z_data_list[[j]]
#     d <- df$H.d
#     m <- matrix(0,nrow = dim(df)[1],ncol = dim(df)[1])# create square matrix matching N of ES, filled with zeros
#     rownames(m) <- df$comp_ID 
#     colnames(m) <- df$comp_ID 
#     shared_coord <- which(df$con_ID %in% df$con_ID[duplicated(df$con_ID)]==TRUE) # find start and end coordinates for the subsets
#     combinations <- do.call("rbind", tapply(shared_coord, df[shared_coord,"con_ID"], function(x) t(combn(x,2)))) # matrix of combinations of coordinates for each experiment with shared control
#     # calculate covariance values between Hd values at the positions in shared_list and place them on the matrix
#     for (i in 1:dim(combinations)[1]){
#       p1 <- combinations[i,1]
#       p2 <- combinations[i,2]
#       p1_p2_cov <- 1/df[p1,"con_n"] + (df[p1,"H.d"]*df[p2,"H.d"]) / (2*df[p1,"N_total"])
#       m[p1,p2] <- p1_p2_cov
#       m[p2,p1] <- p1_p2_cov
#     }
#     # add the diagonal - use df$H.d_Var_combined as matrix diagonal
#     diag(m) <- df$H.d_Var_combined # m is the variance-covariance matrix to be used in the test run
#     AinvG <- solve(m)
#     AnivG <- as(AinvG,"dgCMatrix")
# 
#     models <- foreach(i = chains) %dopar% {
#       filename <- paste("runs/M1.",i,"_sub_expanded",j,".Rdata",sep="") # Remember to change the number so that files have different names.
# 
#       model <- MCMCglmm(d~1,random=~study_ID+animal_ID+comp_ID,family="gaussian",data=df,ginverse=list(comp_ID=AnivG),verbose=T,nitt=4000000,thin=4000,burnin=40000,pr=T,prior=prior2e)
#       save(model, file = filename)
#     }
#   }
# })
# multi.MCMC
# 
