# Deep clustering via AE
# reference codes :
# https://statslab.eighty20.co.za/posts/autoencoders_keras_r/
# https://www.datatechnotes.com/2020/02/how-to-build-simple-autoencoder-with-keras-in-r.html

library(keras)
library(caret)
library(tidyverse)

c(c(xtrain, ytrain), c(xtest, ytest)) %<-% dataset_mnist()
xtrain = xtrain/255
xtest = xtest/255 

input_size = dim(xtrain)[2]*dim(xtrain)[3]
latent_size = 10
print(input_size) 

x_train = array_reshape(xtrain, dim=c(dim(xtrain)[1], input_size))
x_test = array_reshape(xtest, dim=c(dim(xtest)[1], input_size))

x <- rbind( x_test, x_train )/255.0

# Encoder
encoder_input = layer_input(shape = input_size)
encoder_output = encoder_input %>% 
  layer_dense(units=256, activation = "relu") %>% 
  layer_activation_leaky_relu() %>% 
  layer_dense(units=latent_size) %>% 
  layer_activation_leaky_relu()

encoderoder = keras_model(encoder_input, encoder_output)
summary(encoderoder) 

# Decoder
decoder_input = layer_input(shape = latent_size)
decoder_output = decoder_input %>% 
  layer_dense(units=256, activation = "relu") %>% 
  layer_activation_leaky_relu() %>% 
  layer_dense(units = input_size, activation = "relu") %>% 
  layer_activation_leaky_relu()

decoderoder = keras_model(decoder_input, decoder_output)
summary(decoderoder) 

# Autoencoder
autoencoderoder_input = layer_input(shape = input_size)
autoencoderoder_output = autoencoderoder_input %>% 
  encoderoder() %>% 
  decoderoder()

autoencoderoder = keras_model(autoencoderoder_input, autoencoderoder_output)
summary(autoencoderoder)

autoencoderoder %>% compile(optimizer="rmsprop", loss="binary_crossentropy")
autoencoderoder %>% fit(x_train,x_train, epochs=20, batch_size=256) 

encoderoded_imgs = encoderoder %>% predict(x_test)
decoderoded_imgs = decoderoder %>% predict(encoderoded_imgs)

# Images plot
pred_images = array_reshape(decoderoded_imgs, dim=c(dim(decoderoded_imgs)[1], 28, 28)) 

n = 10
op = par(mfrow=c(12,2), mar=c(1,0,0,0))
for (i in 1:n) 
{
  plot(as.raster(pred_images[i,,]))
  plot(as.raster(xtest[i,,]))
}


# Saving trained Net
autoencoderoder_weights <- autoencoderoder %>%
  keras::get_weights()
keras::save_model_weights_hdf5(object = autoencoderoder,filepath = '.../autoencoderoder_weights.hdf5',overwrite = TRUE)

encoderoder_model <- keras_model(inputs = encoder_input, outputs = encoderoder$output)
encoderoder_model %>% keras::load_model_weights_hdf5(filepath = ".../autoencoderoder_weights.hdf5",skip_mismatch = TRUE,by_name = TRUE)

encoderoder_model %>% compile(
  loss='mean_squared_error',
  optimizer='adam',
  metrics = c('accuracy')
)
embeded_points <- 
  encoderoder_model %>% 
  keras::predict_on_batch(x = x_train)

summary(encoderoder_model)

# Getting layer
layer_name<-"dense_1"
intermediate_layer_model <- keras_model(inputs = encoderoder_model$input, outputs = get_layer(encoderoder_model, layer_name)$output)
intermediate_output <- predict(intermediate_layer_model, x)

# Clustering latent space
km <- stats::kmeans( intermediate_output, centers = 10L, nstart = 20L )
labPrediction <- km$cluster 

plot(labPrediction)
# The End
