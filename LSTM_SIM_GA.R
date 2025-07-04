# LSTM DNN Demo for ERP data -------------------------

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


# loading the libraries  ------------------------
library(keras)
library(tensorflow)
library(ggplot2)
library(R.matlab)

# loading data fro mat file  --------------------


inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat")
Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_label.mat")

# noise levels : 1=50dB (no_noise) 2=20B, 3=10B, 4=5dB, 5=0dB, 6=-5dB

iData=as.matrix(inData$SimDaGA.snr1[,,2]) # selecting data noise level e.g., SimDaGA.snr1[,,1] 'no noise'
Lb=as.matrix(Lab$CC.lab) # CC labeling result provided in MATLAB, note that labels should be [0,nbc-1] 
# nbc =number of clusters



# defining training and test sets -------------------

X = as.matrix(iData[, 1:65])

Y = as.matrix(Lb)

ep =100
nbcl=6
SPLT=0.8
N_Samp=length(Lb)
loopback=ncol(iData)
b=floor(SPLT*N_Samp)
# x_train = array_reshape(X[1:b,], c(dim(X[1:b,]), 1))
x_test = array_reshape(X[(b+1):N_Samp,], c(dim(X[(b+1):N_Samp,]), 1))
# y_train =Y[1:b]
y_test = as.matrix(Y[(b+1):N_Samp])


# ---------------------------------------------------------------------------
# define and compile model

model <- keras_model_sequential()
model %>%
  
  # expected input data shape: (batch_size, timesteps, data_dim)
  # many-to-one (return_sequences=False) 
  layer_lstm(units = 64, input_shape=c(loopback, 1), return_sequences = TRUE) %>% 
  layer_batch_normalization() %>% 
  
  layer_lstm(units = 128) %>% #, return_sequences = TRUE) %>%
  layer_batch_normalization() %>% 
  
  # layer_lstm(units = 32) %>% #, return_sequences = TRUE) %>%
  # layer_dropout(0.5) %>%
  
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>% 
  
  layer_dense(units = nbcl, activation = 'softmax')


model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)


summary(model) 

# accumulated learning -----------------------

library(caret)

# Training only on train set
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
  
  # x_train = array_reshape(X[1:b,], c(dim(X[1:b,]), 1))
  # y_train =Y[1:b]
  train_df <-array_reshape(X[-ind,], c(dim(X[-ind,]), 1)) #  PMM[-ind,]
  y_train <- as.matrix(Lb[-ind,])
  valid_df <- array_reshape(X[ind,], c(dim(X[ind,]), 1))# as.matrix(PMM[ind,])
  y_valid <- as.matrix(Lb[ind,])
  
  # Fit the model 
  model_1 <-model %>% fit(
    train_df, y_train, 
    epochs = ep, 
    batch_size = 150,
    validation_data = list(valid_df, y_valid)
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

score <- model%>% keras::evaluate(x_test, y_test)

y_pred <- model %>% predict(x_test) #  why ntstsmple x nmbc ?

summary(model)

get_config(model)

layer_name<-"dense"
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, layer_name)$output)

inData= array_reshape(X, c(dim(X), 1))
intermediate_output <- predict(intermediate_layer_model, inData)

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/dataFeature_LSTM.mat", dataFeature_LSTM=intermediate_output)
writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/tr_inf_LSTM.mat", tr_inf.F1=tr_inf[[1]],tr_inf.F2=tr_inf[[2]],
         tr_inf.F3=tr_inf[[3]],tr_inf.F4=tr_inf[[4]],tr_inf.F5=tr_inf[[5]],te_accloss=score)


#End ---------------------------------------
