'''
Created on Aug 8, 2016

@author: hshi
'''
import caffe
from caffe import layers as L, params as P
import numpy as np
from numpy import ones, newaxis
import scipy.io as sio

from skimage.io import imshow
import os
import random as rd




class LipFrameBatchLoader(object):
    
    def __init__(self, videoNumPerBatch, frameNumPerVideo, dataDir, frameHeight, frameWidth):

        self.videoNumPerBatch = videoNumPerBatch
        self.frameNumPerVideo = frameNumPerVideo
        
        self.dataDir = dataDir
        
        self.frameChannel = 1
        self.frameHeight  =frameHeight
        self.frameWidth = frameWidth
        
        self.samples_all = sio.loadmat(os.path.join(self.dataDir, 'samples.mat'))['resultSamples'].transpose([0,1,4,2,3])
        self.samples_all = self.samples_all/256.0
        self.samples_all = self.samples_all.reshape([self.samples_all.shape[0] * self.samples_all.shape[1],
                                                     self.samples_all.shape[2],
                                                     self.samples_all.shape[3],
                                                     self.samples_all.shape[4]])
        
        
        self.samples_all = self.samples_all[:,:,newaxis, :, :]
        
        self.labels_all = sio.loadmat(os.path.join(self.dataDir, 'labels.mat'))['resultLabels']
        self.labels_all = self.labels_all.reshape([self.labels_all.shape[0] * self.labels_all.shape[1], 
                                                   20])
        
        self.labels_all = self.labels_all[:, :, newaxis, newaxis, newaxis]
        
        
        self.clipMarkers_all = sio.loadmat(os.path.join(self.dataDir, 'clipMarkers.mat'))['clipMarkers']
        self.clipMarkers_all = self.clipMarkers_all.reshape([self.clipMarkers_all.shape[0] * self.clipMarkers_all.shape[1], 
                                                             20])
        
        self.clipMarkers_all = self.clipMarkers_all[:, :, newaxis, newaxis, newaxis]
        
        
        
        sampleInd = np.linspace(0, self.labels_all.shape[0] - 1, self.labels_all.shape[0]).astype('int')
        import random as rd
        rd.shuffle(sampleInd)
        
        self.samples_all = self.samples_all[sampleInd, :, :, :, :]
        self.labels_all = self.labels_all[sampleInd, :, :, :, :]
        self.clipMarkers_all = self.clipMarkers_all[sampleInd, :, :, :, :]
        
        
        
        
        
        
        
        
        
        
        self.currentVideoInd = 0
        
        
#         self.data_all = list()
#         
#         for i in range(self.samples_all.shape[0]):
#             for j in range(self.samples_all.shape[1]):
#                 self.data_all.append({'sample': self.samples_all[i,j,:,:,:], 'label':self.labels_all[i,j,0], 'clipMarker': self.clipMarkers[i,j,:]})
# 
#         rd.shuffle(self.data_all)
        
        self.batchSize = self.videoNumPerBatch * self.frameNumPerVideo
        self.videoNum = self.samples_all.shape[0]
        
        
        self.sampleBatch = np.zeros(shape = (self.batchSize, 
                                             self.frameChannel, 
                                             self.frameHeight, 
                                             self.frameWidth))
        
        self.labelBatch = np.zeros(shape=(self.videoNumPerBatch,
                                          1))
        
        self.clipMarkerBatch = np.ones(shape=(self.frameNumPerVideo, 
                                               self.videoNumPerBatch))
        self.clipMarkerBatch[0,:] = 0
        
        
    def getVideoNum(self):
        return len(self.data_all)
        
    def getNextBatch(self):
        for i in range(self.videoNumPerBatch):
            
            if self.currentVideoInd == self.videoNum:
                self.currentVideoInd = 0
               
            for T in range(self.frameNumPerVideo):
                
                self.sampleBatch[i + T * self.videoNumPerBatch, :, :, :] = \
                    self.samples_all[self.currentVideoInd, T, :, :, :]

            self.labelBatch[i, 0] = self.labels_all[self.currentVideoInd, T, 0, 0, 0]


            self.currentVideoInd += 1
            
            
        return self.sampleBatch, self.labelBatch, self.clipMarkerBatch


class LipDataLayer(caffe.Layer):
    '''
    classdocs
    '''
    def setup(self, bottom, top):
        
        self.top_names = ['sample', 'label', 'clipMarker']
        params = eval(self.param_str)
        

        self.videoNumPerBatch = params['videoNumPerBatch']
        self.frameNumPerVideo = params['frameNumPerVideo']
        self.frameHeight = params['frameHeight']
        self.frameWidth = params['frameWidth']
        self.frameChannel = params['frameChannel']
        self.dataDir = params['dataDir']
        self.batchSize = self.videoNumPerBatch * self.frameNumPerVideo

        self.batchLoader = LipFrameBatchLoader(self.videoNumPerBatch,
                                               self.frameNumPerVideo,
                                               self.dataDir,
                                               self.frameHeight,
                                               self.frameWidth)
        
        
        top[0].reshape(self.batchSize,  self.frameChannel, self.frameHeight, self.frameWidth)
        top[1].reshape(self.videoNumPerBatch, 1)
        top[2].reshape(self.frameNumPerVideo, self.videoNumPerBatch)
    
    def forward(self, bottom, top):
        sampleBatch, labelBatch, clipMarkerBatch = self.batchLoader.getNextBatch()
        top[0].data[...] = sampleBatch
        top[1].data[...] = labelBatch
        top[2].data[...] = clipMarkerBatch

    def reshape(self, bottom, top):
        pass
    
    def backward(self, bottom, top):
        pass
        











def creatNet(DB_PREFIX,
             batch_size_train,
             batch_size_test,
             image_num_per_sequence,
             test_or_train,
             dataLayerParam):
    
    
    image_num_per_sequence
    sequence_num_per_batch_train = (int)(batch_size_train / image_num_per_sequence)
    sequence_num_per_batch_test = (int)(batch_size_test / image_num_per_sequence)
        

    
    
    
    net = caffe.NetSpec()
    # Input layer
    # # Data layer
    # ## Train
    #### V1

           
    [net.sample_v1, 
    net.label,
    net.cm] = L.Python(ntop = 3,
                           module = 'LSTM_Net_Define_Module',
                           layer = 'LipDataLayer',
                           param_str = str(dataLayerParam))
           
    
    # Batch norm layers
    # #V1
    net.bn_1_v1 = L.BatchNorm(net.sample_v1)

    
    
    # Scale layers 1
    # #V1
    net.scale_1_v1 = L.Scale(net.bn_1_v1)

    
    # Convolution layers 1
    # #V1
    net.con_1_v1 = L.Convolution(net.scale_1_v1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 96,
                                                    'kernel_size': 7,
                                                    'stride': 2,
                                                    'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                    'bias_filler': {'type': 'constant', 'value': 0.1}})

    
    # Pooling layer 1
    # #v1
    net.pooling_1_v1 = L.Pooling(net.con_1_v1,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 3,
                                                'stride': 2})
    # #v2

    
    
    # Barch normalizaiton layer 2
    # #V1
    net.bn_2_v1 = L.BatchNorm(net.pooling_1_v1)
    # #V2

    
    
    # Scale layers 2
    # #V1
    net.scale_2_v1 = L.Scale(net.bn_2_v1)

    
    
    # RELU layer 1
    # #V1
    net.relu_1_v1 = L.ReLU(net.scale_2_v1)

    
    # IP 1
    # #v1
    net.ip_1_v1 = L.InnerProduct(net.relu_1_v1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                      'bias_filler': {'type': 'constant', 'value': 0.1}})
    # #v2


    # Droout 1
    # #v1
    net.dropout_1_v1 = L.Dropout(net.ip_1_v1,
                                 dropout_param={'dropout_ratio': 0.6})

    
    # Reshape sample layer 1
    # # TRAIN
    if test_or_train == 'train':

        # ##v1
        net.reshape_sample_1_v1 = L.Reshape(net.dropout_1_v1,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                          sequence_num_per_batch_train,
                                                                          4096] } },
                                            include={'phase': 0})

    
    # # TEST
    else :
        # ##v1
        net.reshape_sample_1_v1 = L.Reshape(net.dropout_1_v1,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_test,
                                                                             4096] } },
                                            include={'phase': 1})
        # ##v2


    
    # LSTM LAYER 1
    # # V1
    net.lstm_1_v1 = L.LSTM(net.reshape_sample_1_v1,
                        net.cm,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'uniform', 'min':-0.01, 'max': 0.01},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})


    
    net.permutedLstm = L.Python(net.lstm_1_v1,
                           module = 'PyPermuteLayer',
                           layer = 'PyPermuteLayer',
                           param_str = str(dict(permuteIndex = [1, 0, 2])))
    

    # IP 2
    
    net.ip_3 = L.InnerProduct(net.permutedLstm,
                              param=[{'lr_mult': 1, 'decay_mult': 1},
                                     {'lr_mult': 2, 'decay_mult': 0}],
                              inner_product_param={'num_output': 10,
                                                      'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                      'bias_filler': {'type': 'constant', 'value': 0.1},
                                                      'axis':1})
    

    
    
    if test_or_train == 'train':
        net.loss = L.SoftmaxWithLoss(net.ip_3,
                                 net.label,
                                 include={'phase': 0})
    
    net.accuracy = L.Accuracy(net.ip_3,
                              net.label)
        
    
    return net
