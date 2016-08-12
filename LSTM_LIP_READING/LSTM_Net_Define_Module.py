'''
Created on Aug 8, 2016

@author: hshi
'''
import caffe
from caffe import layers as L, params as P
import numpy as np
from numpy import ones


def creatNet(data_path,
             batch_size,
             DB_NAME_SAMPLES,
             DB_NAME_LABELS,
             DB_NAME_CLIP_MARKERS,
             DB_NAME_LOGICAL_LABELS,
             DB_NAME_SAMPLE_INDEX,
             image_height,
             image_width,
             channel_num,
             image_num_per_sequence):

    
    sequence_num_per_batch = batch_size / image_num_per_sequence
    
    # Current net structure
    # samples        labels          clip_markers
    # Convolution    Reshape         Reshape
    # ReLu
    # Pooling
    # InnerProduct
    # ReLu
    # DropOut
    # Reshape
    #                       Lstm
    
    # DEFINE THE NETWORK ARCHETECTURE
    net = caffe.NetSpec()
    
    # DATA LAYER
    # SAMPLE AND LABEL
    net.samples = L.Data(batch_size=batch_size,
                         backend=P.Data.LMDB,
                         source=DB_NAME_SAMPLES,
                         transform_param = {'scale': 0.00390625})
    
 
    
    net.labels = L.Data(batch_size=batch_size,
                        backend=P.Data.LMDB,
                        source=DB_NAME_LABELS)
    
    net.clip_markers = L.Data(batch_size=batch_size,
                              backend=P.Data.LMDB,
                              source=DB_NAME_CLIP_MARKERS)
    
    net.sample_indexes = L.Data(batch_size=batch_size,
                              backend=P.Data.LMDB,
                              source=DB_NAME_SAMPLE_INDEX)
    
    # Adding layers 20160809
    # Batch Normalization
    
    net.batch_normalization_1 = L.BatchNorm(net.samples)
    
    net.scale_1 = L.Scale(net.batch_normalization_1,
                          scale_param = {'bias_term': True})
    
    
    
    net.relu_3 = L.ReLU(net.scale_1)
    
    # Adding layers 2016-08-08
    # Convolution Later
    # Bottom:samples
    
    net.convolution_1 = L.Convolution(net.relu_3,
                                      param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                                      convolution_param={'num_output': 96,
                                                           'kernel_size': 7,
                                                           'stride': 2,
                                                           'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                           'bias_filler': {'type': 'constant', 'value': 0.1}})
    
    

    
    # Pooling
    net.pooling_1 = L.Pooling(net.convolution_1,
                              pooling_param={'pool': P.Pooling.MAX,
                                               'kernel_size': 3,
                                               'stride': 2})
    
    net.batch_normalization_2 = L.BatchNorm(net.pooling_1)
    
    net.scale_2 = L.Scale(net.batch_normalization_2,
                          scale_param = {'bias_term': True})
    
    
        # ReLu
    net.relu_1 = L.ReLU(net.scale_2)

    # Inner Product
    net.inner_product_1 = L.InnerProduct(net.relu_1,
                                         param=[{'lr_mult': 1, 'decay_mult': 1}, 
                                                {'lr_mult': 2, 'decay_mult': 0}],
                                         inner_product_param={'num_output': 4096,
                                                              'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                              'bias_filler': {'type': 'constant', 'value': 0.1}})
    
    # ReLu
    net.relu_2 = L.ReLU(net.inner_product_1)
    
    
    
    net.dropout_1 = L.Dropout(net.relu_2,
                              dropout_param={'dropout_ratio': 0.9})
    
    
    
    
    
    
    
    
    net.reshape_sample_1 = L.Reshape(net.dropout_1,
                                     reshape_param={ 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch, 4096] } })
        
    
    # Reshaple lable
    net.reshape_label_1 = L.Reshape(net.labels,
                                    reshape_param={ 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch]}})
    
    # Reshape clip markers
    net.reshape_clip_markers_1 = L.Reshape(net.clip_markers,
                                           reshape_param={ 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch]}})

    
    # LSTM network
    
    net.lstm_1 = L.LSTM(net.reshape_sample_1,
                        net.reshape_clip_markers_1,
                        recurrent_param={'num_output': 256,
                                           'weight_filler': {'type': 'uniform', 'min':-0.01, 'max': 0.01},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    
    
    
    net.dropout_2 = L.Dropout(net.lstm_1,
                              dropout_param={'dropout_ratio': 0.5})
    
    
    
    # INNERPRODUCT LAYER
    net.inner_product_2 = L.InnerProduct(net.lstm_1,
                                         param=[{'lr_mult': 1, 'decay_mult': 1}, 
                                                {'lr_mult': 2, 'decay_mult': 0}],
                                         inner_product_param={'num_output': 10,                                                             
                                                              'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                                              'bias_filler': {'type': 'constant', 'value': 0},
                                                              'axis': -2})
    
    net.slice_ip12_1 = L.Slice()


#     
#     net.inner_product_3 = L.InnerProduct(net.inner_product_2,
#                                          inner_product_param={'num_output': 1,                                                             
#                                                               'weight_filler': {'type': 'gaussian', 'std': 0.01},
#                                                               'bias_filler': {'type': 'constant', 'value': 0},
#                                                               'axis': 0})
#     
#     
# 
#     
#     [net.slice_label_1, net.slice_label_2] = L.Slice(net.reshape_label_1,
#                                                      ntop = 2,
#                                                      slice_param = {'axis': 0,
#                                                                     'slice_point': 1})
#     
#     
#     net.reshape_slice_label_1 = L.Reshape(net.slice_label_1,
#                                  reshape_param={ 'shape': {'dim': [1, sequence_num_per_batch]}})
    
    
    
    
    # LOSS LAYER
    #net.loss = L.SoftmaxWithLoss(net.inner_product_3,
     #                            net.reshape_slice_label_1)
    
    # Accuracy layer
    #net.accuracy = L.Accuracy(net.inner_product_3,
    #                          net.reshape_slice_label_1)
    

    # RESHAPE LAYER
    # WHY RESHAPE?
    # Data from database looks like:
    # sample (sequence = 1, time = 1)
    # sample (sequence = 1, time = 2)
    #                .
    #                .
    #                .
    # sample (sequence = 1, time = T)
    # sample (sequence = 2, time = 1)
    #
    # Thus for feeding the LSTM, the data should like :

    # sample (s = 1, t = 1), sample (s = 2, t = 1), sample (s = 3 ,t = 1),  ...  sample (s = N, t = 1)
    # sample (s = 1, t = 2), sample (s = 2, t = 2), sample (s = 3, t = 2),  ...  sample (s = N, t = 2)
    #                                        .
    #                                        .
    # sample (s = 1, t = T), sample (s = 2, t = T), sample (s = 3, t = T),  ...  sample (s = N, t = T)                                        .
    
    # RESHAPE SHOULE BE TWICE
    # WHY?
    # Because the caffe build-in reshape is line-prioritize filled
    
    # THE 1st RESHAPE
    
    # SAMPLES RESHAPE LAYER
    # input shape (raw sample shape): a blob of (T * N) * h * w
    # desired output shape: a blob of 
    return net
