# Original VAE for Clustering 

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


if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
library(tidyverse)
library(tensorflow)
K <- keras::backend()
library(MLmetrics)
library(aricode)
library(R.matlab)

# loading data fro mat file  -----------------------------------------

# noise levels : 1=50dB (no_noise) 2=20B, 3=10B, 4=5dB, 5=0dB, 6=-5dB

inData1 <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat") # data without noise
iData1=as.matrix(inData1$SimDaGA.snr1[,,1]) # selecting dataset for training


inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat") # noisy data
iData=as.matrix(inData$SimDaGA.snr1[,,2]) # selecting dataset for training


Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_GT.mat")
Lb=as.matrix(Lab$CC.GT) # no noise and ground-truth
# Lab <- readMat("C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/CC_label.mat")
# Lb=as.matrix(Lab$CC.label)

# Parameters initialization --------------------------------------------

N_Samp=length(Lb)
d_dim=ncol(iData)
original_dim <- ncol(iData)
latent_dim <- 6L
nbcl=6L
intermediate_dim <- c(128,256,256)
ep <- 100L
batchsiz=150L
epsilon_std <- 1.0

# Model definition --------------------------------------------------------

x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim[1], activation = "tanh")
h <- layer_batch_normalization(h)
h <- layer_dropout(h,0.25)

h <- layer_dense(h, intermediate_dim[2], activation = "tanh")
h <- layer_batch_normalization(h)
h <- layer_dropout(h,0.25)

h <- layer_dense(h, intermediate_dim[3], activation = "tanh")
h <- layer_batch_normalization(h)
h <- layer_dropout(h,0.25) 

# designing the distribution in the latent layer

z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

# note that "output_shape" isn't necessary with the TensorFlow backend

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling) # lambda layer

# we instantiate these layers separately so as to reuse them later
decoder_h <-layer_dense(z,units = intermediate_dim[3], activation = "tanh")
decoder_h <-layer_batch_normalization(decoder_h)
decoder_h <- layer_dropout(decoder_h,0.25)

decoder_h <-layer_dense(decoder_h,units = intermediate_dim[2], activation = "tanh")
decoder_h <-layer_batch_normalization(decoder_h)
decoder_h <- layer_dropout(decoder_h,0.25)

decoder_h<-layer_dense(decoder_h,units = intermediate_dim[1], activation = "tanh")
decoder_h <-layer_batch_normalization(decoder_h)
decoder_h <- layer_dropout(decoder_h,0.25)

x_decoded_mean <- layer_dense(decoder_h, units = original_dim, activation = 'linear')
# h_decoded <- decoder_h(z)
# x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder ****** -----------------
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
# decoder_input <- layer_input(shape = latent_dim)
# h_decoded_2 <- decoder_h(decoder_input)
# x_decoded_mean_2 <- decoder_mean(h_decoded_2)
# generator <- keras_model(decoder_input, x_decoded_mean)

vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_mean_squared_error(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(
  optimizer=optimizer_rmsprop(learning_rate = 0.001), # optimizer_sgd(lr = 0.001) , optimizer_adam(lr = 0.001),optimizer_rmsprop 'sgd', # or try 'rmsprop' ,
  loss = vae_loss,
  metrics = c('accuracy'))
# optimizer can be -> "rmsprop" 

summary(vae)

# get_config(vae)

# Data preparation --------------------------------------------------------
# get_config(vae)

x <- iData
y <- as.vector(Lb)


# Data Preparation --------------------------------------------------------
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


# Model training ----------------------------------------------------------

# Cross-validation ----------------------------------------------------

library(caret)

idx=c(1:b)

folds <- createFolds(idx, k = 5, list = F)

# to save results of training
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
  
  # Fit the model 
  # model_1 <-autoencoder_model %>% fit(
  #   train_df, train_df, 
  #   epochs =ep,
  #   batch_size = 300,
  #   validation_data = list(valid_df, valid_df)
  # )
  
  model_1 <-vae %>% fit(
    train_df, train_df_clean, 
    shuffle = TRUE,
    epochs =ep,
    batch_size =batchsiz,
    validation_data = list(valid_df, valid_df_clean)
  )
  
  # validation_split = 0.12
  
  
  # The training information
  info[1,]=c(model_1$metrics$loss)
  info[2,]=c(model_1$metrics$accuracy)
  info[3,]=c(model_1$metrics$val_loss)
  info[4,]=c(model_1$metrics$val_accuracy)
  
  # saving the information
  tr_inf[f]<-list(matrix(info,nrow = 4,ncol = ep))
  
}

summary(vae)

score <- vae%>% keras::evaluate(x_test, x_test_clean)


# ------------------------------------------------------------
layer_name1<-"dense_3"
intermediate_layer_model <- keras_model(inputs = vae$input, outputs =  get_layer(vae, layer_name1)$output)
intermediate_output <- predict(intermediate_layer_model, iData)

# Initialize clusters using k-means
cat('Initializing cluster centers with k-means.\n')
km <- stats::kmeans(intermediate_output, centers = nbcl, nstart = 20L )
currentPrediction <- km$cluster #fitted( km )
plot(currentPrediction)
Lb=as.vector(Lb)
cat(" ACC= ", Accuracy(Lb, currentPrediction), ", NMI= ", NMI(Lb, currentPrediction), "\n", sep = ' ' )

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/currentPrediction.mat", currentPrediction=currentPrediction)

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/dataFeature_VAE.mat", dataFeature_VAE=intermediate_output)
writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/tr_inf_VAE.mat", tr_inf.F1=tr_inf[[1]],tr_inf.F2=tr_inf[[2]],
         tr_inf.F3=tr_inf[[3]],tr_inf.F4=tr_inf[[4]],tr_inf.F5=tr_inf[[5]],te_accloss=score)

# End --------------------------------
