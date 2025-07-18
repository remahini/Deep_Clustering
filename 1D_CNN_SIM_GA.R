# 1D_CNN Demo for ERP data -------------------------

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


# loading the library --------------------------
library(keras)
library(ggplot2)
library(R.matlab)

# loading ERP mat data and consensus clustering (CC) labeling -----------------

inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimDaGA_snr1.mat")
Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_label.mat")

# noise levels : 1=50dB (no_noise) 2=20B, 3=10B, 4=5dB, 5=0dB, 6=-5dB

iData=as.matrix(inData$SimDaGA.snr1[,,1]) # selecting data noise level e.g., SimDaGA.snr1[,,1] 'no noise'
Lb=as.matrix(Lab$CC.lab) # CC labeling result provided in MATLAB 
Lb = Lb-1 


# Data Preparation ----------------------------
SPLT=0.8
N_Samp=length(Lb) 
loopback=length(iData[1,])
d_dim=ncol(iData)
b=floor(SPLT*N_Samp)

train.ind = c(1:b)
test.ind=c((b+1):N_Samp)
# x_train = as.matrix(PMM[train.ind, 1:d_dim])
# y_train = as.matrix(KMS[train.ind, 1])
x_test = as.matrix(iData[test.ind, 1:d_dim])
y_test= as.matrix(Lb[test.ind, 1])

# Defining Model ----------------------------

#Initialize model
ep = 100L 
nbcl=6
numb_s=20
sa_div=2
input_shape=length(iData[1,]) #sa_div*numb_s

model <- keras_model_sequential()

model %>% 
  layer_reshape(c(5, input_shape/5), input_shape = input_shape) %>%
  layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu', input_shape = input_shape) %>%
  layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu') %>%
  
  layer_max_pooling_1d() %>%
  layer_batch_normalization() %>% 
  
  layer_conv_1d(filters = 256, kernel_size = 1, activation = 'relu') %>%
  layer_conv_1d(filters = 256, kernel_size = 1, activation = 'relu') %>%
  layer_batch_normalization() %>% 
  
  # layer_max_pooling_1d() %>%
  
  layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu') %>%
  layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu') %>%
  
  layer_global_average_pooling_1d()%>%
  layer_batch_normalization() %>% 
  
  # layer_flatten() %>% 
  layer_dense(units = nbcl, activation = 'softmax')

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

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
  
  train_df = as.matrix(iData[-ind, 1:d_dim])
  y_train = as.matrix(Lb[-ind, 1])
  
  # train_df <- PMM[-ind,]
  # y_train <- as.matrix(KMS[-ind,])
  valid_df <- as.matrix(iData[ind, 1:d_dim]) # as.matrix(PMM[ind,])
  y_valid <- as.matrix(Lb[ind, 1]) #as.matrix(KMS[ind,])
  
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


summary(model)

get_config(model)

layer_name<-"dense"
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, layer_name)$output)
intermediate_output <- predict(intermediate_layer_model, iData)

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/dataFeature_CNN.mat", dataFeature_CNN=intermediate_output)
writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/tr_inf_CNN.mat", tr_inf.F1=tr_inf[[1]],tr_inf.F2=tr_inf[[2]],
         tr_inf.F3=tr_inf[[3]],tr_inf.F4=tr_inf[[4]],tr_inf.F5=tr_inf[[5]],te_accloss=score)


# End ----------------------------------
