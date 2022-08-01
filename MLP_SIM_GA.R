# clearning workspace
rm(list = ls())
gc()  #free up memrory and report the memory usage.

# loding the libray --------------------------------
library(keras)
library(ggplot2)
library(R.matlab) # reading/writing .mat files

# loading data fro mat file  --------------------


# inData <- readMat("D:/My works/Current/Deep clustering/ERP_SIM_GA/inDaGA.mat")
# Lab <- readMat("D:/My works/Current/Deep clustering/ERP_SIM_GA/CC_label.mat")
# iData=as.matrix(inData$inDaGA.M1)
# Lb=as.matrix(Lab$CC.label)

# normal shape of data without reforming
# inData <- readMat("D:/My works/Current/S_ConClust/ERP_CC_MS/CC_SIM/inData_GA.mat")
# Lab <- readMat

# inData <- readMat("D:/My works/Current/Deep clustering/DC_Simulated data/inDaGA_M1.mat")
# Lab <- readMat("D:/My works/Current/Deep clustering/DC_Simulated data/CC_M1_idx.mat")
# 
# iData=as.matrix(inData$inDaGA.M1)
# Lb=as.matrix(Lab$Clu.idx2)

inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimData_Gans.mat")
Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_GT.mat")
iData=as.matrix(inData$SimData.Gans[,,6]) # selecting dataset for training
# noise levels : 1=-10dB, 2=-5dB, 3=0dB, 4=5dB, 5=10dB, 6=20dB, 7=50dB
Lb=as.matrix(Lab$CC.GT) # no noise and ground-truth
# Lab <- readMat("C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/CC_label.mat")
# Lb=as.matrix(Lab$CC.label)


# defining training and test sets ------------------
nbcl=6
SPLT=0.8
N_Samp=length(Lb)    #600   # 19500  
# loopback=length(iData[1,])
d_dim=ncol(iData)
b=floor(SPLT*N_Samp)

train.ind = c(1:b)

test.ind=c((b+1):N_Samp)
x_test = as.matrix(iData[test.ind, 1:d_dim])
y_test= as.matrix(Lb[test.ind])


# ----------------------------------------------------
# Initialize model
model <- keras_model_sequential()
model %>% 
  
  layer_dense(units = 64, activation = 'relu', input_shape = c(d_dim)) %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = 512, activation = "relu") %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = 256, activation = "relu") %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = nbcl, activation = 'softmax')

summary(model)

sgd <- optimizer_sgd(lr = 0.001)

# Try using different optimizers and different optimizer configs
model %>% compile(
  optimizer = 'adam',
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)


# Cross-validation ----------------------------------------------------

library(caret)

# Training only on train set
idx=c(1:b)
ep=100
folds <- createFolds(idx, k = 5, list = F)

# to save results of training
info<- matrix(nrow = 4,ncol = ep)
vacc_all<-c()
vloss_all<-c()
score_acc <-c()
score_los <- c()
tr_inf=list()
cnt<-1

for(f in unique(folds)){
  
  cat("\n Fold: ", f)
  ind <- which(folds == f) 
  train_df <- iData[-ind,]
  y_train <- as.matrix(Lb[-ind])
  valid_df <- as.matrix(iData[ind,])
  y_valid <- as.matrix(Lb[ind])
  
  # Fit the model 
  model_1 <-model %>% fit(
    train_df, y_train, 
    epochs = ep, 
    batch_size = 150,
    verbose = 1,
    callbacks = callback_tensorboard("logs/run_a"),
    validation_data = list(valid_df, y_valid)
  )
  
  plot(model_1)
  # validation_split = 0.12
  
  vacc_all[(cnt*ep-ep+1):(cnt*ep)]=c(model_1$metrics$val_accuracy)
  vloss_all[(cnt*ep-ep+1):(cnt*ep)]=c(model_1$metrics$val_loss)
  cnt<-cnt+1
  
  # The training information
  info[1,]=c(model_1$metrics$loss)
  info[2,]=c(model_1$metrics$accuracy)
  info[3,]=c(model_1$metrics$val_loss)
  info[4,]=c(model_1$metrics$val_accuracy)
  
  # saving the information
  tr_inf[f]<-list(matrix(info,nrow = 4,ncol = ep))
  
}

plot(vacc_all)
plot(vloss_all)

score <- model%>% keras::evaluate(x_test, y_test)

# extracting a layer in designed model

summary(model)

# get_config(model)

layer_name<-"dense"
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, layer_name)$output)
intermediate_output <- predict(intermediate_layer_model, iData)

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/dataFeature_MLP.mat", dataFeature_MLP=intermediate_output)
writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/tr_inf_MLP.mat", tr_inf.F1=tr_inf[[1]],tr_inf.F2=tr_inf[[2]],
         tr_inf.F3=tr_inf[[3]],tr_inf.F4=tr_inf[[4]],tr_inf.F5=tr_inf[[5]],te_accloss=score)
