Ensemble Deep clustering for time window determination of grand average ERP

This demo presents using the ensemble deep clustering method based on our latest work that is under review in Brain Signal Processing and Control journal [1]. We design an ensemble clustering from popular deep clustering methods, including semi-supervised methods initialized by consensus clustering and unsupervised deep clustering methods with end-to-end architectures. Afterward, an adaptive time window determination method is used for qualifying and determining the time window for the ERP of interest. For an assessment, we apply the proposed method to ERP dataset with different strengths of additional noise (i.e., from 20 dB to -5 dB).

About this demo
We have provided a demo code including the deep clustering applied in the abovementioned research, including 6 DNN models (MLP-FC, LSTM, 1DCNN, AE, VAE, and DEC). A consensus clustering from the most popular clustering method in neuroimaging has been designed, and it is available in this demo in the form of Matlab files. The data is available for testing the materials.


Copyright info

Method and Code by Reza Mahini 
Email: r.mahini@gmail.com, remahini@jyu.fi

Please address these works if you find them useful:

[1 ]Mahini, R., Li, F., Zarei, M., K. Nandi, A., Hämäläinen, T., & Cong, F. (2022). Deep Clustering Analysis for Time Window Determination of Event-Related Potential. BSPC-D-22-00635, Available at SSRN. https://doi.org/http://dx.doi.org/10.2139/ssrn.4068456

[2] Mahini, R., Li, Y., Ding, W., Fu, R., Ristaniemi, T., Nandi, A.K., et al. (2020). Determination of the Time Window of Event-Related Potential Using Multiple-Set Consensus Clustering. Frontiers in Neuroscience 14(1047). doi: 10.3389/fnins.2020.521595

[3] Mahini, R., Xu, P., Chen, G., Li, Y., Ding, W., Zhang, L., Qureshi, N. K., Hämäläinen, T., Nandi, A. K., & Cong, F. (2022). Optimal Number of Clusters by Measuring Similarity Among Topographies for Spatio-Temporal ERP Analysis. Brain Topography. https://doi.org/10.1007/s10548-022-00903-2


Note ---------------------------------------------------------------------------------------------
Before you run this Code, you will need to run individual DNNs to get these datasets. 
To ease this, we provided the materials when additional noise meets 10dB (i.e., you will find the related materials in folder ‘Matlab_codes’). Then you will need to copy them to the Matlab workspace for executing the main MatLab file ‘ERP_EDC_SIM.m’

Therefore, open Rstudio (we suppose you already have installed Python up v3.7 and the required packages in R, especially: TensorFlow, Keras, ggplot2, ... 
See more details and the required libraries in our R codes and follow them. You will be able to change the directory address to any directory you wish.

 Otherwise, you may prepare the below files before:
1.	dataFeature_MLP.mat
2.	dataFeature_CNN.mat
3.	dataFeature_LSTM.mat
4.	dataFeature_AE.mat
5.	dataFeature_VAE.mat
6.	DEC_lb.mat

Recommended libraries for Rstudio:
1.	library(keras)
2.	library(ggplot2)
3.	library(R.matlab) 
4.	library(tidyverse)
5.	library(tensorflow)
6.	library(MLmetrics)
7.	library(aricode)
