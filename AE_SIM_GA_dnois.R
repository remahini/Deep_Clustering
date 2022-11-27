# Deep AE for ERP data -------------------------

# Data was prepared applying the temporal concatenating across the conditions
# for individual subjects and grand averaged.

# Note: the clustering result(labeling) should be fed on this procedure. Use MATLAB code
# in downloaded pack for clustering initialization of this DNN.

# Cite as:
# Deep Clustering Analysis for Time Window Determination of Event-Related Potential
# January 2022SSRN Electronic Journal
# DOI: 10.2139/ssrn.4068456

# Copyright:
# This code provided by Reza Mahini, University of Jyväskylä, Finland.
# If you had question or comments welcome to send me email to 
# remahini@jyu.fi


# cleaning the workspace
# rm(list = ls())
# gc()  #free up memrory and report the memory usage.

# Loading the Library -------------------------

library(keras)
library(ggplot2)
library(R.matlab) # reading/writing .mat files
library(tidyverse)
library(tensorflow)
library(caret)

library (dplyr)
K <- keras::backend()
library(MLmetrics)
library(aricode)


# loading data fro mat file  --------------------

# noise levels : 1=50dB (no_noise) 2=20B, 3=10B, 4=5dB, 5=0dB, 6=-5dB

inData1 <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat") # data without noise
iData1=as.matrix(inData1$SimDaGA.snr1[,,1]) # selecting dataset for training


inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat") # noisy data
iData=as.matrix(inData$SimDaGA.snr1[,,2]) # selecting dataset for training

# this labeling is for an assesment 
Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_GT.mat")
Lb=as.matrix(Lab$CC.GT) # no noise and ground-truth

# inData <- readMat("D:/My works/Current/Deep clustering/ERP_CORE_GA_new/inDaGA.mat") # data without noise
# iData=as.matrix(inData1$inDaGA.M1)

ep = 100L # epoch
nbcl=6 # number of expected class
d_dim=ncol(iData)
latent_dim<- c(256, 512)
BatchSize=150L

# Encoder ----------

input_layer <- layer_input(shape = c(d_dim))
encoder <- input_layer %>%
  
  
  layer_dense(units = latent_dim[1], activation = 'tanh', input_shape = c(d_dim)) %>%
  layer_batch_normalization() %>%
  # layer_layer_normalization()%>%
  layer_dropout(0.2)%>%
  
  
  layer_dense(units = latent_dim[2], activation = 'tanh') %>%
  layer_batch_normalization() %>%
  # layer_layer_normalization()%>%
  layer_dropout(0.2)%>%
  
  
  # layer_dense(units = 64, activation = 'relu') %>%
  # layer_batch_normalization() %>%
  # # layer_layer_normalization()%>%
  
  # For the classes
  layer_dense(units = nbcl) %>% #,activation = 'softmax')# %>% 
  layer_activation_leaky_relu()


# Decoder ----------
decoder <-
  encoder %>% 
  layer_dense(units = latent_dim[2], activation = 'tanh') %>%
  layer_batch_normalization() %>%
  # layer_layer_normalization()%>%
  layer_dropout(0.2)%>%
  
  
  
  layer_dense(units = latent_dim[1], activation = 'tanh') %>%
  layer_batch_normalization() %>%
  # layer_layer_normalization()%>%
  layer_dropout(0.2)%>%
  
  
  layer_dense(units = d_dim, activation = 'linear') # for the original input 'linear'


# Autoencoder model
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

# encoder model for classification
encoder <- keras_model(inputs = input_layer, encoder)


autoencoder_model %>% compile(
  loss= 'mean_squared_error', # Or try 'categorical_crossentropy', #'mean_squared_error'
  optimizer=optimizer_rmsprop(learning_rate = 0.001), # optimizer_sgd(lr = 0.001) , optimizer_adam(lr = 0.001),optimizer_rmsprop 'sgd', # or try 'rmsprop' ,
  metrics = c('accuracy')
)


summary(autoencoder_model)


# defining training and test sets -----------------------------------

# nrminmax <- function (x) {
#   (x-min(x))/(max(x)-min(x))
# }
# 
# for(i in 1:d_dim){
#   iData[,i]<-nrminmax(iData[,i])
# }

# iData=scale(iData, center = TRUE, scale = TRUE) # equal to z-scoring

#iiData=norm_minmax(iData[,1:28])
# iiData=matrix(nrow = nrow(iData),ncol = ncol(iData))



SPLT=0.8 # spliting percentage 80%
N_Samp=length(Lb) 
d_dim=ncol(iData)
b=floor(SPLT*N_Samp)

train.ind = c(1:b)
test.ind=c((b+1):N_Samp)
x_train = as.matrix(iData[train.ind, 1:d_dim])
x_train_clean = as.matrix(iData1[train.ind, 1:d_dim])
# y_train = as.matrix(Lb[train.ind])
x_test = as.matrix(iData[test.ind, 1:d_dim])
x_test_clean = as.matrix(iData1[test.ind, 1:d_dim])

# y_test= as.matrix(Lb[test.ind])


# Cross-validation ----------------------------------------------------
# Training only on train set
idx=c(1:b)

folds <- createFolds(idx, k = 5, list = F)

# for saving the results of training
info<- matrix(nrow = 4,ncol = ep)
score_acc <-c()
score_los <- c()
tr_inf=list()

for(f in unique(folds)){
  
  cat("\n Fold: ", f)
  ind <- which(folds == f) 
  train_df <- iData[-ind,]
  train_df_clean <- iData1[-ind,]
  y_train <- as.matrix(Lb[-ind])
  valid_df <- as.matrix(iData[ind,])
  valid_df_clean <- as.matrix(iData1[ind,])
  
  
  y_valid <- as.matrix(Lb[ind])
  
  # Fit the model 
  model_1 <-autoencoder_model %>% fit(
    train_df, train_df_clean, 
    epochs =ep,
    batch_size =BatchSize,
    validation_data = list(valid_df, valid_df_clean)
  )
  
  
  # The training information
  info[1,]=c(model_1$metrics$loss)
  info[2,]=c(model_1$metrics$accuracy)
  info[3,]=c(model_1$metrics$val_loss)
  info[4,]=c(model_1$metrics$val_accuracy)
  
  # saving the information
  tr_inf[f]<-list(matrix(info,nrow = 4,ncol = ep))
  
}

score <- autoencoder_model%>% keras::evaluate(x_test, x_test)
score <- autoencoder_model%>% keras::evaluate(x_test, x_test_clean)


# we don't need the rest of code (saving the results) --------------------------
# autoencoder_weights <- 
#   autoencoder_model %>%
#   keras::get_weights()
# keras::save_model_weights_hdf5(object = autoencoder_model,filepath = 'D:/My works/Current/Deep clustering/ERP_CORE_GA_new/autoencoder_weights.hdf5',overwrite = TRUE)
# 
encoder_model <- keras_model(inputs = input_layer, outputs = encoder$output)
# encoder_model %>% keras::load_model_weights_hdf5(filepath = 'D:/My works/Current/Deep clustering/ERP_CORE_GA_new/autoencoder_weights.hdf5',kip_mismatch = TRUE,by_name = TRUE)
# 

embeded_points <- encoder_model %>% 
  keras::predict_on_batch(x = x_test)


# Saving the weighted inputs for clustering -----------------------------
summary(encoder_model)

layer_name<-"dense" #"leaky_re_lu"  "dense"
intermediate_layer_model <- keras_model(inputs = encoder_model$input, outputs = get_layer(encoder_model, layer_name)$output)
intermediate_output <- predict(intermediate_layer_model, iData)


# Initialize clusters using k-means
cat('Initializing cluster centers with k-means.\n')
km <- stats::kmeans(intermediate_output, centers = nbcl, nstart = 20L )
currentPrediction <- km$cluster #fitted( km )
plot(currentPrediction)
Lb=as.vector(Lb)
cat(" ACC= ", Accuracy(Lb, currentPrediction), ", NMI= ", NMI(Lb, currentPrediction), "\n", sep = ' ' )
cat("F1_score= ", F1_Score(currentPrediction,Lb))
# cat("F1_score= ", confusionMatrix(currentPrediction,Lb))


writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/dataFeature_AE.mat", dataFeature_AE=intermediate_output)
writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/tr_inf_AE.mat", tr_inf.F1=tr_inf[[1]],tr_inf.F2=tr_inf[[2]],
         tr_inf.F3=tr_inf[[3]],tr_inf.F4=tr_inf[[4]],tr_inf.F5=tr_inf[[5]],te_accloss=score)

# End ---------------------
# End ---------------------
