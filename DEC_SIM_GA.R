# Deep embedded clustering

rm(list = ls())
gc()  #free up memrory and report the memory usage.

# loding the libray --------------------------------

library(keras)
linrK <- keras::backend()
library(ggplot2)
library(R.matlab) # reading/writing .mat files
library(tidyverse)
library(tensorflow)
library(MLmetrics)
library(aricode)

#model defination --------------------------------

# defination of clustring layer
# 1- building (input_shape), 2- call, 3- comput_output(input_shape)
# Clusterlayer converts input sample (feature) to soft label.

# Example
# ```
# model.add(ClusterLayer(n_clusters=10))
# ```
# Arguments
# n_clusters: number of clusters.
# weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
# alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.



# loading data fro mat file  -------------------------------------------------------------------

# inData <- readMat("D:/My works/Current/Deep clustering/ERP_CORE/P3_Data.mat")
# Lab <- readMat("D:/My works/Current/Deep clustering/ERP_CORE/CC_idx.mat")
# 
# iData=as.matrix(inData$outData)
# Lb=as.matrix(Lab$CC.Label)
inData1 <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimData_Gans.mat") # data without noise
iData1=as.matrix(inData1$SimData.Gans[,,7]) # selecting dataset for training

inData <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/Sim_MS3_data/SimData_Gans.mat") # noisy data
iData=as.matrix(inData$SimData.Gans[,,6]) # selecting dataset for training
# noise levels : 1=-10dB, 2=-5dB, 3=0dB, 4=5dB, 5=10dB, 6=20dB, 7=50dB

Lab <- readMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/CC_GT.mat") # ground truth labels
Lb=as.matrix(Lab$CC.GT) # no noise and ground-truth
# Lab <- readMat("C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/CC_label.mat")
# Lb=as.matrix(Lab$CC.label)

# Initialization ------------------------------------------------------------------------------
nbcl=6
batchSz=150L
ep=200L


# ---------------------------------------------------------------------------------------------------------------

# Fully connected auto-encoder model, symmetric.
# Arguments:
#   dims: list of number of units in each layer of encoder. dims[1] is input dim, dims[-1] is units in hidden layer.
# The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
# act: activation, not applied to Input, Hidden and Output layers
# return:
#   (ae_model, encoder_model), Model of autoencoder and model of encoder

createAutoencoderModel <- function( numberOfUnitsPerLayer,
                                    activation = 'tanh',
                                    initializer = 'glorot_uniform' )
{
  numberOfEncodingLayers <- as.integer(length( numberOfUnitsPerLayer ) - 1)
  # input of AE
  inputs <- layer_input( shape =c(as.integer(numberOfUnitsPerLayer[1])) )
  encoder <- inputs
  
  # internal layers in encoder
  for( i in seq_len( numberOfEncodingLayers - 1 ) )
  {
    encoder <- encoder %>%
      layer_dense( numberOfUnitsPerLayer[i+1],
                   activation = activation, kernel_initializer = initializer )
  }
  
  # hidden layer
  encoder <- encoder %>%
    layer_dense( units = tail( numberOfUnitsPerLayer, 1 ) )
  
  autoencoder <- encoder
  
  # internal layers in decoder
  for( i in seq( from = numberOfEncodingLayers, to = 2, by = -1 ) )
  {
    autoencoder <- autoencoder %>%
      layer_dense( numberOfUnitsPerLayer[i],
                   activation = activation, kernel_initializer = initializer )
  }
  
  # output
  autoencoder <- autoencoder %>%
    layer_dense( numberOfUnitsPerLayer[1], activation = "linear", kernel_initializer = initializer )
  
  return( list(
    autoencoderModel = keras_model( inputs = inputs, outputs = autoencoder ),
    encoderModel = keras_model( inputs = inputs, outputs = encoder ) ) )
}


# ------------------------------------------------------------------------------------------------------

# """
#     Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
#     sample belonging to each cluster. The probability is calculated with student's t-distribution.
#     # Example
#     ```
#         model.add(ClusteringLayer(n_clusters=10))
#     ```
#     # Arguments
#         n_clusters: number of clusters.
#         weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
#         alpha: parameter in Student's t-distribution. Default to 1.0.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.
#     in fact this function will return us the Q matrix (soft assignment) //me

ClusteringLayer <- R6::R6Class( "ClusteringLayer",
                                inherit = KerasLayer,
                                lock_objects = FALSE,
                                # I Changed nu...rs=10 to Null
                                public = list(numberOfClusters = NULL, initialClusterWeights = NULL,
                                              alpha = 1.0, name = '',
                                              
                                              initialize = function( numberOfClusters,
                                                                     initialClusterWeights = NULL, alpha = 1.0, name = '' )
                                              {
                                                self$numberOfClusters <- as.integer(numberOfClusters)
                                                self$initialClusterWeights <- initialClusterWeights
                                                self$alpha <- alpha
                                                self$name <- name
                                              },
                                              
                                              build = function( input_shape )
                                              {
                                                if( length( input_shape ) != 2 )
                                                {
                                                  stop( paste0( "input_shape is not of length 2." ) )
                                                }
                                                
                                                self$clusters <- self$add_weight(
                                                  shape = list( self$numberOfClusters, input_shape[[2]] ),
                                                  initializer = 'glorot_uniform', name = 'clusters' )
                                                
                                                if( ! is.null( self$initialClusterWeights ) )
                                                {
                                                  self$set_weights( self$initialClusterWeights )
                                                  self$initialClusterWeights <- NULL
                                                }
                                                self$built <- TRUE
                                              },
                                              
                                              call = function( inputs, mask = NULL )
                                              {
                                                # Uses Student t-distribution (same as t-SNE)
                                                # inputs are the variable containing the data, shape = ( numberOfSamples, numberOfFeatures )
                                                
                                                K <- keras::backend()
                                                
                                                q <- 1.0 / ( 1.0 + (K$sum( K$square(
                                                  K$expand_dims( inputs, axis = 1L ) - self$clusters ), axis = 2L ) / self$alpha)) 
                                                q <- q^( ( self$alpha + 1.0 ) / 2.0 )
                                                q <- K$transpose( K$transpose( q ) / K$sum( q, axis = 1L ) )
                                                
                                                return( q )
                                              },
                                              
                                              compute_output_shape = function( input_shape )
                                              {
                                                return( list( input_shape[[1]], self$numberOfClusters ) )
                                              }
                                )
)

# Layer Wrapper Function

layer_clustering <- function( object,
                              numberOfClusters, initialClusterWeights = NULL,
                              alpha = 1.0, name = '' )
{
  create_layer( ClusteringLayer, object,
                list( numberOfClusters = numberOfClusters,
                      initialClusterWeights = initialClusterWeights,
                      alpha = alpha, name = name )
  )
}

# ----------------------------------------------------------------------------------------------------------
#' Deep embedded clustering (DEC) model class

DeepEmbeddedClusteringModel <- R6::R6Class( "DeepEmbeddedClusteringModel",
                                            
                                            inherit = NULL,
                                            lock_objects = FALSE,
                                            
                                            public = list(
                                              numberOfUnitsPerLayer = NULL,
                                              numberOfClusters = nbcl,
                                              alpha = 1.0,
                                              initializer = 'glorot_uniform',
                                              convolutional = FALSE,
                                              inputImageSize = NULL,
                                              
                                              initialize = function( numberOfUnitsPerLayer,
                                                                     numberOfClusters, alpha = 1.0, initializer = 'glorot_uniform',
                                                                     convolutional = FALSE, inputImageSize = NULL )
                                              {
                                                self$numberOfUnitsPerLayer <- as.integer(numberOfUnitsPerLayer)
                                                self$numberOfClusters <- as.integer(numberOfClusters)
                                                self$alpha <- alpha
                                                self$initializer <- initializer
                                                self$convolutional <- convolutional
                                                self$inputImageSize <- as.integer(inputImageSize)
                     
                                                ae <- createAutoencoderModel( self$numberOfUnitsPerLayer,
                                                                              initializer = self$initializer )
                                                self$autoencoder <- ae$autoencoderModel
                                                self$encoder <- ae$encoderModel
                                             
                                                # prepare DEC model wrapping -----------------------------------------------
                                                
                                                clusteringLayer <- self$encoder$output %>%
                                                layer_clustering( self$numberOfClusters, name = "clustering" )
                                                
                                                self$model <- keras_model( inputs = self$encoder$input, outputs = clusteringLayer )
                                                },
                                              # here the model for reading input and deliver soft assignment is ready //me
                                  
                                              
                                              pretrain = function( x, y , optimizer=optimizer_rmsprop(learning_rate = 0.001), epochs = ep , batchSize = batchSz )
                                              {
                                                cat('.................... pretraining ....................... \n')
                                                
                                                self$autoencoder$compile( optimizer = optimizer, loss = 'mse' )
                                                self$autoencoder$fit( x, y, batch_size = batchSize, epochs = epochs)
                                                self$pretrained = TRUE
                                              },
                                              
                                              loadWeights = function( weights )
                                              {
                                                self$model$load_weights( weights )
                                              },
                                              
                                              extractFeatures = function( x )
                                              {
                                                return(self$encoder$predict( x ))
                                              },
                                              
                                              predictClusterLabels = function( x )
                                              {
                                                clusterProbabilities <- self$model$predict( x, verbose = 0 )
                                                return( max.col( clusterProbabilities ) )
                                              },
                                              
                                              targetDistribution = function( q )
                                              {
                                                weight <- q^2 / colSums( q )
                                                p <- t( t( weight ) / rowSums( weight ) )
                                                return( p )
                                              },
                                              
                                              compile = function( optimizer = 'sgd', loss =  'kld', lossWeights = NULL )
                                              {
                                                self$model$compile( optimizer = optimizer, loss = loss, loss_weights = lossWeights )
                                              },
                                              
                                              fit = function( x, y ,maxNumberOfIterations = 2e4, batchSize = batchSz , tolerance = 1e-3, updateInterval = 10 )
                                              {
                                                
                                                # Initialize clusters using k-means
                                                
                                                cat('Initializing cluster centers with k-means.\n')
                                                km <- stats::kmeans( self$encoder$predict( x ),
                                                                     centers = self$numberOfClusters, nstart = 20L )
                                                currentPrediction <- km$cluster #fitted( km )
                                                
                                                plot(currentPrediction, col="red")
                                                title(main = 'current prediction')  
                                                previousPrediction <- currentPrediction
                                                
                                                self$model$get_layer( name = 'clustering' )$set_weights( list( km$centers ) )
                                                
                                                # Deep clustering ----------------------------------------------------------
                                                
                                                loss <- 1000
                                                index <- 0
                                                indexArray <- 1:( dim( x )[1] )
                                                
                                                for( i in seq_len( maxNumberOfIterations ) )
                                                {
                                                  if( i %% updateInterval == 1 )
                                                  { 
                                                    q <- self$model$predict( x, verbose = 0 )
                                                    p <- self$targetDistribution( q ) # update the auxiliary target distribution p
                                          
                                                    # evaluate the clustering performance
                                                    currentPrediction <- max.col( q )
                                                    
                                                    plot(currentPrediction, col="blue")
                                                    title(main = 'current prediction')
                                                    
                                                   
                                                    # Met stopping criterion --------------------------------------------------
                                                    # evaluate the clustering performance
                                                    deltaLabel <- sum( currentPrediction != previousPrediction ) / length( currentPrediction )
                                                    
                                                    cat( "Itr", i, ": ( out of", maxNumberOfIterations,
                                                         "): loss = [", unlist( loss ), "], deltaLabel =", deltaLabel,", ACC= ", Accuracy(previousPrediction, currentPrediction),
                                                         ", NMI= ", NMI(previousPrediction, currentPrediction), "\n", sep = ' ' )
                                                    
                                                    previousPrediction <- currentPrediction
                                                    
                                                    # cat("Iteration ", i, ": (out of ", maxNumberOfIterations, "), loss = ", loss, ", deltaLabel = ", deltaLabel, "\n", sep = '' )
                                                    
                                                    if( i > 1 && deltaLabel < tolerance )
                                                    {
                                                      print('Reached tolerance threshold. Stopping training......')
                                                      break
                                                    }
                                                  }
                                                  
                                                  # train on batch
                                                  batchIndices <- indexArray[( index * batchSize + 1 ):min( ( index + 1 ) * batchSize, dim( x )[1] )]
                                                  
                                                  loss <- self$model$train_on_batch( x = x[batchIndices,], y = p[batchIndices,] )

                                                  if( ( index + 1 ) * batchSize + 1 <= dim( x )[1] )
                                                  {
                                                    index <- index + 1
                                                  } else {
                                                    index <- 0
                                                  }
                                                }
                                                return( currentPrediction )
                                              }

                                            )
)


# defining training and test sets ------------------

d_dim=ncol(iData)
x <- iData
y <-iData1  #as.vector(Lb)
numberOfClusters <- nbcl # length( unique( Lb ))
numberOfPixels <- d_dim 

initializer <- initializer_variance_scaling(
  scale = 1/3, mode = 'fan_in', distribution = 'uniform' )

pretrainOptimizer <- optimizer_sgd( learning_rate = 0.001, momentum = 0.9 )
initializer='glorot_uniform'

decModel <- DeepEmbeddedClusteringModel$new(
  numberOfUnitsPerLayer = c( numberOfPixels, 128, 256, 256, nbcl ),
  numberOfClusters = numberOfClusters, initializer = initializer )


decModel$pretrain( x=x, y=y , optimizer = pretrainOptimizer,
                   epochs = ep, batchSize = batchSz )


decModel$compile( optimizer = optimizer_sgd( learning_rate = 0.001, momentum = 0.9 ), loss = 'kld')

yPredicted <- decModel$fit( x, y, maxNumberOfIterations = 2000, batchSize = batchSz,
                            tolerance = 1e-2, updateInterval = 5 )


#Saving the labels

writeMat("D:/My works/Current/Deep clustering/SIM_GA_DC_NL/DEC_lb.mat", yPredicted=yPredicted)
