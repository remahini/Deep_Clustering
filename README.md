Ensemble Consensus clustering for time window determination of grand average ERP

Copyright info

First version of Concenuc clustering (May 2017)
Updated at Aug 2019 Stabilization (Mahini et al. 2020)
Code by Reza Mahini r.mahini@gmail.com, remahini@jyu.fi

Plese cite this work as :

[1] Mahini, R., Li, Y., Ding, W., Fu, R., Ristaniemi, T., Nandi, A.K., et al. (2020).
Determination of the Time Window of Event-Related Potential Using Multiple-Set Consensus Clustering.
Frontiers in Neuroscience 14(1047). doi: 10.3389/fnins.2020.521595.

[2] Mahini, R., Xu, P., Chen, G., Li, Y., Ding, W., Zhang, L., Qureshi, N. K., Hämäläinen, T., Nandi, A. K., & Cong, F. (2022).
Optimal Number of Clusters by Measuring Similarity Among Topographies for Spatio-Temporal ERP Analysis.
Brain Topography. https://doi.org/10.1007/s10548-022-00903-2

[3] Mahini, R., Xu, P., Chen, G., Li, Y., Ding, W., Zhang, L., et al. (2019).
Optimal Number of Clusters by Measuring Similarity among Topographies for Spatio-temporal ERP Analysis.
arXiv preprint arXiv:1911.09415.


arXiv preprint arXiv:1911.09415.

------------------------- Before you run this Code -----------------------
We need to run your individual DNNs to get these datasets 

 Therefore

 Prepare the below files before :
   1- dataFeature_MLP.mat;
   2- dataFeature_CNN.mat;
   3- dataFeature_LSTM.mat;
   4- dataFeature_AE;
   5-dataFeature_VAE;
   6- DEC_lb.mat.  

So, Open Rstodio (we suppose you already have installed Python up v3.7 and the required packages in R,
especially: tensorflow,keras, ggplot2, ... 

Pleae see related details in R file 

