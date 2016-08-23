'''
Created on Aug 8, 2016

@author: hshi
'''
import caffe
from caffe import layers as L, params as P
import numpy as np
from numpy import ones


def creatNet(DB_PREFIX,
             batch_size_train,
             batch_size_test,
             image_num_per_sequence,
             test_or_train):
    
    
    image_num_per_sequence
    sequence_num_per_batch_train = (int)(batch_size_train / image_num_per_sequence)
    sequence_num_per_batch_test = (int)(batch_size_test / image_num_per_sequence)
        
    DB_NAME_SAMPLE_TRAIN = '/SAMPLE_TRAIN'
 
    DB_NAME_SAMPLE_TEST = '/SAMPLE_TEST'
    
    DB_NAME_LABEL_TRAIN = '/LABEL_TRAIN'
    DB_NAME_LABEL_TEST = '/LABEL_TEST'
    DB_NAME_CLIP_MARKER_TRAIN = '/CLIP_MARKER_TRAIN'
    DB_NAME_CLIP_MARKER_TEST = '/CLIP_MARKER_TEST'
    DB_NAME_LOGICAL_LABEL_TRAIN = '/LOGICAL_LABEL_TRAIN'
    DB_NAME_LOGICAL_LABEL_TEST = '/LOGICAL_LABEL_TEST'  
    DB_NAME_SAMPLE_INDEX_TRAIN = '/SAMPLE_INDEX_TRAIN'
    DB_NAME_SAMPLE_INDEX_TEST = '/SAMPLE_INDEX_TEST'
    
    
    
    net = caffe.NetSpec()
    # Input layer
    # # Data layer
    # ## Train
    #### V1
    
    if test_or_train == 'train':
           
        net.sample = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
       
    else:   
        # ## Test
        #### V1
        net.sample = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})

    
    
    
    
    # # Label layer
    if test_or_train == 'train':
        # ## Train
        net.label = L.Data(batch_size=batch_size_train,
                           backend=P.Data.LMDB,
                           source=DB_PREFIX + DB_NAME_LABEL_TRAIN,
                           include={'phase': 0})
    
    # ##Test
    else :
        net.label = L.Data(batch_size=batch_size_test,
                           backend=P.Data.LMDB,
                           source=DB_PREFIX + DB_NAME_LABEL_TEST,
                           include={'phase': 1})    
    # # Clip makrers layer
   
    # ## Train
    if test_or_train == 'train':
        net.cm = L.Data(batch_size=batch_size_train,
                        backend=P.Data.LMDB,
                           source=DB_PREFIX + DB_NAME_CLIP_MARKER_TRAIN,
                           include={'phase': 0})
    # ##Test
    else: 
        net.cm = L.Data(batch_size=batch_size_test,
                        backend=P.Data.LMDB,
                        source=DB_PREFIX + DB_NAME_CLIP_MARKER_TEST,
                        include={'phase': 1})   
    # #Sample index layer
       
    # ##Train
    if test_or_train == 'train':
        net.si = L.Data(batch_size=batch_size_train,
                    backend=P.Data.LMDB,
                    source=DB_PREFIX + DB_NAME_SAMPLE_INDEX_TRAIN,
                    include={'phase': 0})
    # ##Test
    else: 
        net.si = L.Data(batch_size=batch_size_test,
                    backend=P.Data.LMDB,
                    source=DB_PREFIX + DB_NAME_SAMPLE_INDEX_TEST,
                    include={'phase': 1})  
    
    
    
    # Batch norm layers
    net.bn_1 = L.BatchNorm(net.sample)

  
    # Scale layers 1
    net.scale_1 = L.Scale(net.bn_1)

    # Convolution layers 1
    net.con_1 = L.Convolution(net.scale_1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 96,
                                                    'kernel_size': 7,
                                                    'stride': 2,
                                                    'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                    'bias_filler': {'type': 'constant', 'value': 0.1}})

    
    # Pooling layer 1
    net.pooling_1 = L.Pooling(net.con_1,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 3,
                                                'stride': 2})
   
    
    
    # Barch normalizaiton layer 2
    net.bn_2 = L.BatchNorm(net.pooling_1)
    
    
    # Scale layers 2
    net.scale_2 = L.Scale(net.bn_2)
    
    
    
    # RELU layer 1
    net.relu_1 = L.ReLU(net.scale_2)

    
    # IP 1
    net.ip_1 = L.InnerProduct(net.relu_1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                      'bias_filler': {'type': 'constant', 'value': 0.1}})
    
    # Droout 1

    net.dropout_1 = L.Dropout(net.ip_1, dropout_param={'dropout_ratio': 0.6})
   
    
    
    # Reshape sample layer 1
    # # TRAIN
    if test_or_train == 'train':
        
        net.reshape_sample_1 = L.Reshape(net.dropout_1,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                          sequence_num_per_batch_train,
                                                                          4096] } },
                                            include={'phase': 0})
      
       
    # # TEST
    else :
     
        net.reshape_sample_1 = L.Reshape(net.dropout_1,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_test,
                                                                             4096] } },
                                            include={'phase': 1})
        
        
    # Reshape Clipmarkers layer 1
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

    net.lstm_1 = L.LSTM(net.reshape_sample_1,
                        net.reshape_cm_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'uniform', 'min':-0.01, 'max': 0.01},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})

   
    
    
    # IP 2
    
    net.ip_2 = L.InnerProduct(net.lstm_1,
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
