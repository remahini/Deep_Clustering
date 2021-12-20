
"""
Keras implementation for Deep Embedded Clustering (DEC) algorithm:
        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.
Usage:
    use `python DEC.py -h` for help.
Author:
    Xifeng Guo. 2017.1.30
"""
from time import time
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
#import metrics
from sklearn import metrics 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



def autoencoder(dims, act='tanh', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
        #h = BatchNormalization()(h)
        h = Dropout(0.4)(h)
    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
        #y = BatchNormalization()(y)
        y = Dropout(0.4)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

 # ------------------------------------------------------------------------------------


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    
   

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# ------------------------------------------------------------------------------------
class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=5,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='rmsprop', epochs=100, batch_size=200, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse', metrics='accuracy')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
    
        # begin pretraining
        t0 = time()
        for i in range(0, 1):            
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
            print('Pretraining time: %ds' % round(time() - t0))
        # self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        # print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
            self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 1000
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                          
                # plot
                plt.scatter(np.arange(0,len(y_pred)),y_pred)
                plt.show()
                
                if y is not None:
                    acc = np.round(metrics.accuracy_score(y_pred_last, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y_pred_last, y_pred), 5)
                    ari = np.round( metrics.adjusted_mutual_info_score(y_pred_last, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and  delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            #if ite % save_interval == 0:
                #print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                #self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        #logfile.close()
        #print('saving model to:', save_dir + '/DEC_model_final.h5')
        #self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', default='mnist',
    #                     choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
        
        
    # load dataset ------------------------------------------------------------

    import scipy.io as sio
    
    # x_data= sio.loadmat('D:/My works/Current/Deep clustering/DC_Simulated data/inDaGA_M1.mat')
    # y_label=sio.loadmat('D:/My works/Current/Deep clustering/DC_Simulated data/CC_M1_idx.mat')
    # indata=x_data['inDaGA_M1']
    # x=indata/255
    # y=np.concatenate(y_label['Clu_idx2'])
    
    
    # x_data= sio.loadmat('D:/My works/Current/S_ConClust/ERP_CC_GA_Toolbox/full/inDaGA.mat')
    # y_label=sio.loadmat('D:/My works/Current/S_ConClust/ERP_CC_GA_Toolbox/full/CC_label.mat')
    # indata=x_data['inDaGA_M1']
    # x=indata/255
    # y=np.concatenate(y_label['CC_label'])
    
    x_data= sio.loadmat('C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/SimData_Gans.mat')
    y_label=sio.loadmat('C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/CC_GT.mat')
    
    dataAll=x_data['SimData_Gans']
    x=dataAll[:,:,1]/255.0 # put them in "n-1" starts from 0 :p
    y=np.concatenate(y_label['CC_GT'])
    
           
    
    plt.scatter(np.arange(0,len(y)),y)
    plt.show()
                
    n_clusters = len(np.unique(y)) 
    init = 'glorot_uniform'
    pretrain_optimizer = 'rmsprop' #rmsprop
    update_interval = 20
    pretrain_epochs = 50 
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    
    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    # prepare the DEC model
    dec = DEC(dims=[x.shape[-1], 256, 512, 128, 6], n_clusters=n_clusters, init=init)

    dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     save_dir=args.save_dir)

    dec.model.summary()
    t0 = time()
    dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = dec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=update_interval, save_dir=args.save_dir)
    
    sio.savemat('C:/Users/Rza/Google Drive/Current/Deep clustering/SIM_GA_DC_NL/DEC_idx.mat',{'y_pred': y_pred})
    
    
    # print('acc:', metrics.normalized_mutual_info_score(y, y_pred))
    # print('clustering time: ', (time() - t0))