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
        
        self.labelBatch = np.zeros(shape=(self.batchSize,
                                          1,
                                          1,
                                          1))
        
        self.clipMarkerBatch = np.zeros(shape=(self.batchSize, 
                                               1,
                                               1,
                                               1))
        #self.clipMarkerBatch[0,:] = 0
        
        
    def getVideoNum(self):
        return len(self.data_all)
        
    def getNextBatch(self):
        for i in range(self.videoNumPerBatch):
            
            if self.currentVideoInd == self.videoNum:
                self.currentVideoInd = 0
               
            for T in range(self.frameNumPerVideo):
                
                self.sampleBatch[i + T * self.videoNumPerBatch, :, :, :] = \
                    self.samples_all[self.currentVideoInd, T, :, :, :]
               
                self.labelBatch[i + T * self.videoNumPerBatch, :, :, :] = \
                    self.labels_all[self.currentVideoInd, T, :, :, :]
            
                
                self.clipMarkerBatch[i + T * self.videoNumPerBatch, :, :, :] = \
                    self.clipMarkers_all[self.currentVideoInd, T, :, :, :]

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
        top[1].reshape(self.batchSize, 1, 1, 1)
        top[2].reshape(self.batchSize, 1, 1, 1)
    
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
        
    DB_NAME_SAMPLE_TRAIN_V1 = '/SAMPLE_TRAIN_V1'
#     DB_NAME_SAMPLE_TRAIN_V2 = '/SAMPLE_TRAIN_V2'
#     DB_NAME_SAMPLE_TRAIN_V3 = '/SAMPLE_TRAIN_V3'
#     DB_NAME_SAMPLE_TRAIN_V4 = '/SAMPLE_TRAIN_V4'
#     DB_NAME_SAMPLE_TRAIN_V5 = '/SAMPLE_TRAIN_V5'
    
    DB_NAME_SAMPLE_TEST_V1 = '/SAMPLE_TEST_V1'
#     DB_NAME_SAMPLE_TEST_V2 = '/SAMPLE_TEST_V2'
#     DB_NAME_SAMPLE_TEST_V3 = '/SAMPLE_TEST_V3'
#     DB_NAME_SAMPLE_TEST_V4 = '/SAMPLE_TEST_V4'
#     DB_NAME_SAMPLE_TEST_V5 = '/SAMPLE_TEST_V5'
    
    DB_NAME_LABEL_TRAIN = '/LABEL_TRAIN'
    DB_NAME_LABEL_TEST = '/LABEL_TEST'
    DB_NAME_CLIP_MARKER_TRAIN = '/CLIP_MARKER_TRAIN'
    DB_NAME_CLIP_MARKER_TEST = '/CLIP_MARKER_TEST'

    DB_NAME_SAMPLE_INDEX_TRAIN = '/SAMPLE_INDEX_TRAIN'
    DB_NAME_SAMPLE_INDEX_TEST = '/SAMPLE_INDEX_TEST'
    
    
    
    net = caffe.NetSpec()
    # Input layer
    # # Data layer
    # ## Train
    #### V1
    
    if test_or_train == 'train':
           
        [net.sample_v1, 
         net.label,
         net.cm] = L.Python(ntop = 3,
                           module = 'LSTM_Net_Define_Module',
                           layer = 'LipDataLayer',
                           param_str = str(dataLayerParam),
                           include={'phase': 0})
           
           
#         net.sample_v1 = L.Data(batch_size=batch_size_train,
#                                backend=P.Data.LMDB,
#                                source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V1,
#                                transform_param={'scale': 0.00390625},
#                                include={'phase': 0})
#         
# 
#         net.label = L.Data(batch_size=batch_size_train,
#                            backend=P.Data.LMDB,
#                            source=DB_PREFIX + DB_NAME_LABEL_TRAIN,
#                            include={'phase': 0})
# 
#         net.cm = L.Data(batch_size=batch_size_train,
#                         backend=P.Data.LMDB,
#                            source=DB_PREFIX + DB_NAME_CLIP_MARKER_TRAIN,
#                            include={'phase': 0})
#         
# 
#         net.si = L.Data(batch_size=batch_size_train,
#                     backend=P.Data.LMDB,
#                     source=DB_PREFIX + DB_NAME_SAMPLE_INDEX_TRAIN,
#                     include={'phase': 0})   
#         


    else:   
        # ## Test
        #### V1
        net.sample_v1 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V1,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})
        

        net.label = L.Data(batch_size=batch_size_test,
                           backend=P.Data.LMDB,
                           source=DB_PREFIX + DB_NAME_LABEL_TEST,
                           include={'phase': 1})    
    # # Clip makrers layer

        net.cm = L.Data(batch_size=batch_size_test,
                        backend=P.Data.LMDB,
                        source=DB_PREFIX + DB_NAME_CLIP_MARKER_TEST,
                        include={'phase': 1})   


        net.si = L.Data(batch_size=batch_size_test,
                    backend=P.Data.LMDB,
                    source=DB_PREFIX + DB_NAME_SAMPLE_INDEX_TEST,
                    include={'phase': 1})  
    
    
    
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

    # # TRAIN
    if test_or_train == 'train':

        net.reshape_cm_1 = L.Reshape(net.cm,
                                   reshape_param={'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch_train] } },
                                   include={'phase': 0})    

    # #TEST
    else: 
        net.reshape_cm_1 = L.Reshape(net.cm,
                               reshape_param={'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch_test] } },
                               include={'phase': 1})      
    
    
    # LSTM LAYER 1
    # # V1
    net.lstm_1_v1 = L.LSTM(net.reshape_sample_1_v1,
                        net.reshape_cm_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'uniform', 'min':-0.01, 'max': 0.01},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})


    
    

    # IP 2
    
    net.ip_2 = L.InnerProduct(net.lstm_1_v1,
                              param=[{'lr_mult': 1, 'decay_mult': 1},
                                     {'lr_mult': 2, 'decay_mult': 0}],
                              inner_product_param={'num_output': 10,
                                                      'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                      'bias_filler': {'type': 'constant', 'value': 0.1},
                                                      'axis':2})
    
    # Slice train ip_2
    if test_or_train == 'train':

    # #train
        [net.slice_sample_1_1,
         net.slice_sample_1_2,
         net.slice_sample_1_3] = L.Slice(net.ip_2,
                                         ntop=3,
                                         slice_param={'axis': 1,
                                                      'slice_point': [1, 2]},
                                         include={'phase': 0})
    
    # #test
    
    
    # Reshape sliced sample
    # #train
    if test_or_train == 'train':

    # ##slice_1
        net.reshape_sample_2_1 = L.Reshape(net.slice_sample_1_1,
                                           reshape_param={'shape': { 'dim': [1, 20, 10] }},
                                           include={'phase': 0})
        
        # ##slice_2
        net.reshape_sample_2_2 = L.Reshape(net.slice_sample_1_2,
                                           reshape_param={'shape': { 'dim': [1, 20, 10] }},
                                           include={'phase': 0})
        
        # ##slice_3
        net.reshape_sample_2_3 = L.Reshape(net.slice_sample_1_3,
                                           reshape_param={'shape': { 'dim': [1, 20, 10] }},
                                           include={'phase': 0})
    
    # #test
    else:
        net.reshape_sample_2_1 = L.Reshape(net.ip_2,
                                       reshape_param={'shape': { 'dim': [1, 20, 10] }},
                                       include = {'phase': 1})
     
    
    
    # Concate 2
    # #train
    if test_or_train == 'train':

        net.concate_sample_2 = L.Concat(net.reshape_sample_2_1,
                                    net.reshape_sample_2_2,
                                    net.reshape_sample_2_3,
                                    concat_param={'axis': 0},
                                    include={'phase': 0})
    
    # #test
    else:
        net.concate_sample_2 = L.Concat(net.reshape_sample_2_1,
                                    concat_param={'axis': 0},
                                    include={'phase': 1})
    
    
    
    
    # ip_3
    net.ip_3 = L.InnerProduct(net.concate_sample_2,
                              param=[{'lr_mult': 1, 'decay_mult': 1},
                                     {'lr_mult': 2, 'decay_mult': 0}],
                              inner_product_param={'num_output': 10,
                                                      'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                      'bias_filler': {'type': 'constant', 'value': 0.1},
                                                      'axis':1})
    
    
    
    # Reshape label
    net.reshape_label_1 = L.Reshape(net.label,
                                    reshape_param={'shape': {'dim': [20, -1] } } )
    
    # slice label 1
    [net.slice_label_1_1,
     net.slice_label_1_2] = L.Slice(net.reshape_label_1,
                                    ntop=2,
                                    slice_param={'axis': 0,
                                                 'slice_point': 1})
     
    # reshape label 2
    # #train
    if test_or_train == 'train':

                                    net.reshape_label_2 = L.Reshape(net.slice_label_1_1,
                                    reshape_param={'shape': { 'dim': [3, 1] }},
                                    include={'phase': 0})
    
    # #test
    else: 
        net.reshape_label_2 = L.Reshape(net.slice_label_1_1,
                                    reshape_param={'shape': { 'dim': [1, 1] }},
                                    include={'phase': 1})
    
    
    if test_or_train == 'train':
        net.loss = L.SoftmaxWithLoss(net.ip_3,
                                 net.reshape_label_2,
                                 include={'phase': 0})
    
    net.accuracy = L.Accuracy(net.ip_3,
                              net.reshape_label_2)
        
    
    return net
