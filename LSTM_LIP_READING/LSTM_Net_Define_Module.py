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
        
    DB_NAME_SAMPLE_TRAIN_V1 = '/SAMPLE_TRAIN_V1'
    DB_NAME_SAMPLE_TRAIN_V2 = '/SAMPLE_TRAIN_V2'
    DB_NAME_SAMPLE_TRAIN_V3 = '/SAMPLE_TRAIN_V3'
    DB_NAME_SAMPLE_TRAIN_V4 = '/SAMPLE_TRAIN_V4'
    DB_NAME_SAMPLE_TRAIN_V5 = '/SAMPLE_TRAIN_V5'
    
    DB_NAME_SAMPLE_TEST_V1 = '/SAMPLE_TEST_V1'
    DB_NAME_SAMPLE_TEST_V2 = '/SAMPLE_TEST_V2'
    DB_NAME_SAMPLE_TEST_V3 = '/SAMPLE_TEST_V3'
    DB_NAME_SAMPLE_TEST_V4 = '/SAMPLE_TEST_V4'
    DB_NAME_SAMPLE_TEST_V5 = '/SAMPLE_TEST_V5'
    
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
           
        net.sample_v1 = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V1,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
        
        #### V2
        net.sample_v2 = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V2,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
        #### V3
        net.sample_v3 = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V3,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
        
        #### V4
        net.sample_v4 = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V4,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
        #### V5
        net.sample_v5 = L.Data(batch_size=batch_size_train,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TRAIN_V5,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 0})
    else:   
        # ## Test
        #### V1
        net.sample_v1 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V1,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})
        
        #### V2
        net.sample_v2 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V2,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})
        #### V3
        net.sample_v3 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V3,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})
        
        #### V4
        net.sample_v4 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V4,
                               transform_param={'scale': 0.00390625},
                               include={'phase': 1})
        #### V5
        net.sample_v5 = L.Data(batch_size=batch_size_test,
                               backend=P.Data.LMDB,
                               source=DB_PREFIX + DB_NAME_SAMPLE_TEST_V5,
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
#     # #V1
#     net.bn_1_v1 = L.BatchNorm(net.sample_v1)
#     # #V2
#     net.bn_1_v2 = L.BatchNorm(net.sample_v2)
#     # #V3
#     net.bn_1_v3 = L.BatchNorm(net.sample_v3)
#     # #V4
#     net.bn_1_v4 = L.BatchNorm(net.sample_v4)
#     # #V5
#     net.bn_1_v5 = L.BatchNorm(net.sample_v5)
#     
#     
#     # Scale layers 1
#     # #V1
#     net.scale_1_v1 = L.Scale(net.bn_1_v1)
#     # #V2
#     net.scale_1_v2 = L.Scale(net.bn_1_v2)
#     # #V3
#     net.scale_1_v3 = L.Scale(net.bn_1_v3)
#     # #V4
#     net.scale_1_v4 = L.Scale(net.bn_1_v4)
#     # #V5
#     net.scale_1_v5 = L.Scale(net.bn_1_v5)
#     
    # Convolution layers 1
    # #V1
    net.con_1_v1 = L.Convolution(net.sample_v1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 64,
                                                    'kernel_size': 3,
                                                    'stride': 1,
                                                    'weight_filler': {'type': 'xavier'},
                                                    'bias_filler': {'type': 'constant', 'value': 0}})
    # #v2
    net.con_1_v2 = L.Convolution(net.sample_v2,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 64,
                                                    'kernel_size': 3,
                                                    'stride': 1,
                                                    'weight_filler': {'type': 'xavier'},
                                                    'bias_filler': {'type': 'constant', 'value': 0}})
    # #v3
    net.con_1_v3 = L.Convolution(net.sample_v3,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 64,
                                                    'kernel_size': 3,
                                                    'stride': 1,
                                                    'weight_filler': {'type': 'xavier'},
                                                    'bias_filler': {'type': 'constant', 'value': 0}})
    # #v4
    net.con_1_v4 = L.Convolution(net.sample_v4,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 64,
                                                    'kernel_size': 3,
                                                    'stride': 1,
                                                    'weight_filler': {'type': 'xavier'},
                                                    'bias_filler': {'type': 'constant', 'value': 0}})
    # #v5
    net.con_1_v5 = L.Convolution(net.sample_v5,
                                 param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                 convolution_param={'num_output': 64,
                                                    'kernel_size': 3,
                                                    'stride': 1,
                                                    'weight_filler': {'type': 'xavier'},
                                                    'bias_filler': {'type': 'constant', 'value': 0}})
    
    
#         # Convolution layers 2
#     # #V1
#     net.con_2_v1 = L.Convolution(net.con_1_v1,
#                                  param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
#                                  convolution_param={'num_output': 64,
#                                                     'kernel_size': 3,
#                                                     'stride': 1,
#                                                     'weight_filler': {'type': 'xavier'},
#                                                     'bias_filler': {'type': 'constant', 'value': 0}})
#     # #v2
#     net.con_2_v2 = L.Convolution(net.con_1_v2,
#                                  param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
#                                  convolution_param={'num_output': 64,
#                                                     'kernel_size': 3,
#                                                     'stride': 1,
#                                                     'weight_filler': {'type': 'xavier'},
#                                                     'bias_filler': {'type': 'constant', 'value': 0}})
#     # #v3
#     net.con_2_v3 = L.Convolution(net.con_1_v3,
#                                  param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
#                                  convolution_param={'num_output': 64,
#                                                     'kernel_size': 3,
#                                                     'stride': 1,
#                                                     'weight_filler': {'type': 'xavier'},
#                                                     'bias_filler': {'type': 'constant', 'value': 0}})
#     # #v4
#     net.con_2_v4 = L.Convolution(net.con_1_v4,
#                                  param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
#                                  convolution_param={'num_output': 64,
#                                                     'kernel_size': 3,
#                                                     'stride': 1,
#                                                     'weight_filler': {'type': 'xavier'},
#                                                     'bias_filler': {'type': 'constant', 'value': 0}})
#     # #v5
#     net.con_2_v5 = L.Convolution(net.con_1_v5,
#                                  param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
#                                  convolution_param={'num_output': 64,
#                                                     'kernel_size': 3,
#                                                     'stride': 1,
#                                                     'weight_filler': {'type': 'xavier'},
#                                                     'bias_filler': {'type': 'constant', 'value': 0}})
    
    
    
    
    # Pooling layer 1
    # #v1
    net.pooling_1_v1 = L.Pooling(net.con_1_v1,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 2,
                                                'stride': 2})
    # #v2
    net.pooling_1_v2 = L.Pooling(net.con_1_v2,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 2,
                                                'stride': 2})
    # #v3
    net.pooling_1_v3 = L.Pooling(net.con_1_v3,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 2,
                                                'stride': 2})
    # #v4
    net.pooling_1_v4 = L.Pooling(net.con_1_v4,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 2,
                                                'stride': 2})
    # #v5
    net.pooling_1_v5 = L.Pooling(net.con_1_v5,
                                 pooling_param={'pool': P.Pooling.MAX,
                                                'kernel_size': 2,
                                                'stride': 2})
    
#     
#     # Barch normalizaiton layer 2
#     # #V1
#     net.bn_2_v1 = L.BatchNorm(net.pooling_1_v1)
#     # #V2
#     net.bn_2_v2 = L.BatchNorm(net.pooling_1_v2)
#     # #V3
#     net.bn_2_v3 = L.BatchNorm(net.pooling_1_v3)
#     # #V4
#     net.bn_2_v4 = L.BatchNorm(net.pooling_1_v4)
#     # #V5
#     net.bn_2_v5 = L.BatchNorm(net.pooling_1_v5)
#     
#     
#     # Scale layers 2
#     # #V1
#     net.scale_2_v1 = L.Scale(net.bn_2_v1)
#     # #V2
#     net.scale_2_v2 = L.Scale(net.bn_2_v2)
#     # #V3
#     net.scale_2_v3 = L.Scale(net.bn_2_v3)
#     # #V4
#     net.scale_2_v4 = L.Scale(net.bn_2_v4)
#     # #V5
#     net.scale_2_v5 = L.Scale(net.bn_2_v5)
    
    
    # RELU layer 1
    # #V1
    net.relu_1_v1 = L.ReLU(net.pooling_1_v1)
    # #v2
    net.relu_1_v2 = L.ReLU(net.pooling_1_v2)
    # #v3
    net.relu_1_v3 = L.ReLU(net.pooling_1_v3)
    # #v4
    net.relu_1_v4 = L.ReLU(net.pooling_1_v4)
    # #v5
    net.relu_1_v5 = L.ReLU(net.pooling_1_v5)
    
    # IP 1
    # #v1
    net.ip_1_v1 = L.InnerProduct(net.relu_1_v1,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    # #v2
    net.ip_1_v2 = L.InnerProduct(net.relu_1_v2,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    # #v3
    net.ip_1_v3 = L.InnerProduct(net.relu_1_v3,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    # #v4
    net.ip_1_v4 = L.InnerProduct(net.relu_1_v4,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    # #v5
    net.ip_1_v5 = L.InnerProduct(net.relu_1_v5,
                                 param=[{'lr_mult': 1, 'decay_mult': 1},
                                        {'lr_mult': 2, 'decay_mult': 0}],
                                 inner_product_param={'num_output': 4096,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0}})
    # Droout 1
    # #v1
    net.dropout_1_v1 = L.Dropout(net.ip_1_v1,
                                 dropout_param={'dropout_ratio': 0.6})
    # #v2
    net.dropout_1_v2 = L.Dropout(net.ip_1_v2,
                                 dropout_param={'dropout_ratio': 0.6})
    # #v3
    net.dropout_1_v3 = L.Dropout(net.ip_1_v3,
                                 dropout_param={'dropout_ratio': 0.6})
    # #v4
    net.dropout_1_v4 = L.Dropout(net.ip_1_v4,
                                 dropout_param={'dropout_ratio': 0.6})
    # #v5
    net.dropout_1_v5 = L.Dropout(net.ip_1_v5,
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
        # ##v2
        net.reshape_sample_1_v2 = L.Reshape(net.dropout_1_v2,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_train,
                                                                             4096] } },
                                            include={'phase': 0})
        # ##v3
        net.reshape_sample_1_v3 = L.Reshape(net.dropout_1_v3,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_train,
                                                                             4096] } },
                                            include={'phase': 0})
        # ##v4
        net.reshape_sample_1_v4 = L.Reshape(net.dropout_1_v4,
                                          reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                           sequence_num_per_batch_train,
                                                                           4096] } },
                                          include={'phase': 0})
        # ##v5
        net.reshape_sample_1_v5 = L.Reshape(net.dropout_1_v5,
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
        net.reshape_sample_1_v2 = L.Reshape(net.dropout_1_v2,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_test,
                                                                             4096] } },
                                            include={'phase': 1})
        # ##v3
        net.reshape_sample_1_v3 = L.Reshape(net.dropout_1_v3,
                                            reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                             sequence_num_per_batch_test,
                                                                             4096] } },
                                            include={'phase': 1})
        # ##v4
        net.reshape_sample_1_v4 = L.Reshape(net.dropout_1_v4,
                                          reshape_param={'shape': {'dim': [image_num_per_sequence,
                                                                           sequence_num_per_batch_test,
                                                                           4096] } },
                                          include={'phase': 1})
        # ##v5
        net.reshape_sample_1_v5 = L.Reshape(net.dropout_1_v5,
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
    # # V1
    net.lstm_1_v1 = L.LSTM(net.reshape_sample_1_v1,
                        net.reshape_cm_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'xavier'},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})

    # #v2
    net.lstm_1_v2 = L.LSTM(net.reshape_sample_1_v2,
                        net.reshape_cm_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'xavier'},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    # #v3
    net.lstm_1_v3 = L.LSTM(net.reshape_sample_1_v3,
                           net.reshape_cm_1,
                           recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'xavier'},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    # #v4
    net.lstm_1_v4 = L.LSTM(net.reshape_sample_1_v4,
                           net.reshape_cm_1,
                           recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'xavier'},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    # #v5
    net.lstm_1_v5 = L.LSTM(net.reshape_sample_1_v5,
                        net.reshape_cm_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'xavier'},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    
#     # LSTM LAYER 2
#     # # V1
#     net.lstm_2_v1 = L.LSTM(net.lstm_1_v1,
#                         net.reshape_cm_1,
#                         recurrent_param={'num_output': 256,
#                                            'weight_filler': {'type': 'xavier'},
#                                            'bias_filler': {'type': 'constant', 'value': 0 }})
# 
#     # #v2
#     net.lstm_2_v2 = L.LSTM(net.lstm_1_v2,
#                         net.reshape_cm_1,
#                         recurrent_param={'num_output': 256,
#                                            'weight_filler': {'type': 'xavier'},
#                                            'bias_filler': {'type': 'constant', 'value': 0 }})
#     # #v3
#     net.lstm_2_v3 = L.LSTM(net.lstm_1_v3,
#                            net.reshape_cm_1,
#                            recurrent_param={'num_output': 256,
#                                            'weight_filler': {'type': 'xavier'},
#                                            'bias_filler': {'type': 'constant', 'value': 0 }})
#     # #v4
#     net.lstm_2_v4 = L.LSTM(net.lstm_1_v4,
#                            net.reshape_cm_1,
#                            recurrent_param={'num_output': 256,
#                                            'weight_filler': {'type': 'xavier'},
#                                            'bias_filler': {'type': 'constant', 'value': 0 }})
#     # #v5
#     net.lstm_2_v5 = L.LSTM(net.lstm_1_v5,
#                         net.reshape_cm_1,
#                         recurrent_param={'num_output': 256,
#                                            'weight_filler': {'type': 'xavier'},
#                                            'bias_filler': {'type': 'constant', 'value': 0 }})    
    
    # # TRAIN
#     if test_or_train == 'train':
#         net.reshape_sample_3_v1 = L.Reshape(net.lstm_1_v1,
#                                          reshape_param={'shape': {'dim': [batch_size_train, 1, 256, 1] } },
#                                          include = {'phase': 0})   
#         
#         net.reshape_sample_3_v2 = L.Reshape(net.lstm_1_v2,
#                                          reshape_param={'shape': {'dim': [batch_size_train, 1, 256, 1] } },
#                                          include = {'phase': 0})   
#         
#         net.reshape_sample_3_v3 = L.Reshape(net.lstm_1_v3,
#                                          reshape_param={'shape': {'dim': [batch_size_train, 1, 256, 1] } },
#                                          include = {'phase': 0})   
#         
#         net.reshape_sample_3_v4 = L.Reshape(net.lstm_1_v4,
#                                          reshape_param={'shape': {'dim': [batch_size_train, 1, 256, 1] } },
#                                          include = {'phase': 0})   
#         
#         net.reshape_sample_3_v5 = L.Reshape(net.lstm_1_v5,
#                                          reshape_param={'shape': {'dim': [batch_size_train, 1, 256, 1] } },
#                                          include = {'phase': 0})   
#     
#     
#     else:
#         net.reshape_sample_3_v1 = L.Reshape(net.lstm_1_v1,
#                                          reshape_param={'shape': {'dim': [batch_size_test, 1, 256, 1] } },
#                                          include = {'phase': 1})   
#     
#         net.reshape_sample_3_v2 = L.Reshape(net.lstm_1_v2,
#                                          reshape_param={'shape': {'dim': [batch_size_test, 1, 256, 1] } },
#                                          include = {'phase': 1})  
#         
#         net.reshape_sample_3_v3 = L.Reshape(net.lstm_1_v3,
#                                          reshape_param={'shape': {'dim': [batch_size_test, 1, 256, 1] } },
#                                          include = {'phase': 1})  
#         
#         net.reshape_sample_3_v4 = L.Reshape(net.lstm_1_v4,
#                                          reshape_param={'shape': {'dim': [batch_size_test, 1, 256, 1] } },
#                                          include = {'phase': 1})  
#         
#         net.reshape_sample_3_v5 = L.Reshape(net.lstm_1_v5,
#                                          reshape_param={'shape': {'dim': [batch_size_test, 1, 256, 1] } },
#                                          include = {'phase': 1})  
#     
#     
#     
#     
#     # Barch normalizaiton layer 3
#     # Batch norm layers
#     # ##Train
#     if test_or_train == 'train':
#         
#         net.bn_3_v1 = L.BatchNorm(net.reshape_sample_3_v1,
#                                batch_norm_param = {'use_global_stats': False},
#                                include = {'phase': 0})
#         
#         net.bn_3_v2 = L.BatchNorm(net.reshape_sample_3_v2,
#                                batch_norm_param = {'use_global_stats': False},
#                                include = {'phase': 0})
#         
#         net.bn_3_v3 = L.BatchNorm(net.reshape_sample_3_v3,
#                                batch_norm_param = {'use_global_stats': False},
#                                include = {'phase': 0})
#         
#         net.bn_3_v4 = L.BatchNorm(net.reshape_sample_3_v4,
#                                batch_norm_param = {'use_global_stats': False},
#                                include = {'phase': 0})
#         
#         net.bn_3_v5 = L.BatchNorm(net.reshape_sample_3_v5,
#                                batch_norm_param = {'use_global_stats': False},
#                                include = {'phase': 0})
# 
#     else:
#         net.bn_3_v1 = L.BatchNorm(net.reshape_sample_3_v1,
#                                param = [{'lr_mult': 0},
#                                         {'lr_mult': 0},
#                                         {'lr_mult': 0}],
#                                batch_norm_param = {'use_global_stats': True},
#                                include = {'phase': 1})
#         
#         net.bn_3_v2 = L.BatchNorm(net.reshape_sample_3_v2,
#                                param = [{'lr_mult': 0},
#                                         {'lr_mult': 0},
#                                         {'lr_mult': 0}],
#                                batch_norm_param = {'use_global_stats': True},
#                                include = {'phase': 1})
#         
#         net.bn_3_v3 = L.BatchNorm(net.reshape_sample_3_v3,
#                                param = [{'lr_mult': 0},
#                                         {'lr_mult': 0},
#                                         {'lr_mult': 0}],
#                                batch_norm_param = {'use_global_stats': True},
#                                include = {'phase': 1})
#         
#         net.bn_3_v4 = L.BatchNorm(net.reshape_sample_3_v4,
#                                param = [{'lr_mult': 0},
#                                         {'lr_mult': 0},
#                                         {'lr_mult': 0}],
#                                batch_norm_param = {'use_global_stats': True},
#                                include = {'phase': 1})
#         
#         net.bn_3_v5 = L.BatchNorm(net.reshape_sample_3_v5,
#                                param = [{'lr_mult': 0},
#                                         {'lr_mult': 0},
#                                         {'lr_mult': 0}],
#                                batch_norm_param = {'use_global_stats': True},
#                                include = {'phase': 1})
#                            
#     # Barch normalizaiton layer 2
#     
#     
#     # Scale layers 2
#     net.scale_3_v1 = L.Scale(net.bn_3_v1,
#                           scale_param = {'bias_term': True })
#     
#     net.scale_3_v2 = L.Scale(net.bn_3_v2,
#                           scale_param = {'bias_term': True })
#     
#     net.scale_3_v3 = L.Scale(net.bn_3_v3,
#                           scale_param = {'bias_term': True })
#     
#     net.scale_3_v4 = L.Scale(net.bn_3_v4,
#                           scale_param = {'bias_term': True })
#     
#     net.scale_3_v5 = L.Scale(net.bn_3_v5,
#                           scale_param = {'bias_term': True })
#     
#     
    # RELU layer 2
    net.relu_2_v1 = L.ReLU(net.lstm_1_v1)
    
    net.relu_2_v2 = L.ReLU(net.lstm_1_v2)
    
    net.relu_2_v3 = L.ReLU(net.lstm_1_v3)
    
    net.relu_2_v4 = L.ReLU(net.lstm_1_v4)
    
    net.relu_2_v5 = L.ReLU(net.lstm_1_v5)
    
    
    net.reshape_sample_4_v1 = L.Reshape(net.relu_2_v1,
                            reshape_param={'shape': {'dim': [image_num_per_sequence, -1, 256] } })   
    
    net.reshape_sample_4_v2 = L.Reshape(net.relu_2_v2,
                            reshape_param={'shape': {'dim': [image_num_per_sequence, -1, 256] } })   
    
    net.reshape_sample_4_v3 = L.Reshape(net.relu_2_v3,
                            reshape_param={'shape': {'dim': [image_num_per_sequence, -1, 256] } })   
    
    net.reshape_sample_4_v4 = L.Reshape(net.relu_2_v4,
                            reshape_param={'shape': {'dim': [image_num_per_sequence, -1, 256] } })   
    
    net.reshape_sample_4_v5 = L.Reshape(net.relu_2_v5,
                            reshape_param={'shape': {'dim': [image_num_per_sequence, -1, 256] } })   
    
    
    
    
    # Concate sample 1
    net.concate_sample_1 = L.Concat(net.reshape_sample_4_v1,
                                    net.reshape_sample_4_v2,
                                    net.reshape_sample_4_v3,
                                    net.reshape_sample_4_v4,
                                    net.reshape_sample_4_v5,
                                    concat_param={'axis': 2})
    
    # IP 2
    
    net.ip_2 = L.InnerProduct(net.concate_sample_1,
                              param=[{'lr_mult': 1, 'decay_mult': 1},
                                     {'lr_mult': 2, 'decay_mult': 0}],
                              inner_product_param={'num_output': 10,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0},
                                                      'axis':2})
    
    # Slice train ip_2
#     if test_or_train == 'train':
# 
#     # #train
#         [net.slice_sample_1_1,
#          net.slice_sample_1_2,
#          net.slice_sample_1_3] = L.Slice(net.ip_2,
#                                          ntop=3,
#                                          slice_param={'axis': 1,
#                                                       'slice_point': [1, 2]},
#                                          include={'phase': 0})
#     
#     # #test
#     
#     
#     # Reshape sliced sample
#     # #train
#     if test_or_train == 'train':
# 
#     # ##slice_1
#         net.reshape_sample_2_1 = L.Reshape(net.slice_sample_1_1,
#                                            reshape_param={'shape': { 'dim': [1, 20, 10] }},
#                                            include={'phase': 0})
#         
#         # ##slice_2
#         net.reshape_sample_2_2 = L.Reshape(net.slice_sample_1_2,
#                                            reshape_param={'shape': { 'dim': [1, 20, 10] }},
#                                            include={'phase': 0})
#         
#         # ##slice_3
#         net.reshape_sample_2_3 = L.Reshape(net.slice_sample_1_3,
#                                            reshape_param={'shape': { 'dim': [1, 20, 10] }},
#                                            include={'phase': 0})
#     
#     # #test
#     else:
    net.reshape_sample_2_1 = L.Reshape(net.ip_2,
                                       reshape_param={'shape': { 'dim': [1, 20, 10] }})
     
    
#     
#     # Concate 2
#     # #train
#     if test_or_train == 'train':
# 
#         net.concate_sample_2 = L.Concat(net.reshape_sample_2_1,
#                                     net.reshape_sample_2_2,
#                                     net.reshape_sample_2_3,
#                                     concat_param={'axis': 0},
#                                     include={'phase': 0})
#     
#     # #test
#     else:
#         net.concate_sample_2 = L.Concat(net.reshape_sample_2_1,
#                                     concat_param={'axis': 0},
#                                     include={'phase': 1})
#     
#     
    
    
    # ip_3
    net.ip_3 = L.InnerProduct(net.reshape_sample_2_1,
                              param=[{'lr_mult': 1, 'decay_mult': 1},
                                     {'lr_mult': 2, 'decay_mult': 0}],
                              inner_product_param={'num_output': 10,
                                                      'weight_filler': {'type': 'xavier'},
                                                      'bias_filler': {'type': 'constant', 'value': 0},
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
                                    reshape_param={'shape': { 'dim': [-1, 1] }},
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
