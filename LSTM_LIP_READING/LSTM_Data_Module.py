'''
Created on Jul 27, 2016

@author: henglinshi
'''

import scipy.io as sio
import numpy as np
from numpy import newaxis
import caffe
import os
import lmdb
import random as rd
import shutil


def loadData(data_path):
    samples = sio.loadmat(os.path.join(data_path, 'samples.mat'))['resultSamples']
    samples = samples.transpose((0,1,4,2,3))
    
    labels = sio.loadmat(os.path.join(data_path, 'labels.mat'))['resultLabels']
    
    logical_labels = sio.loadmat(os.path.join(data_path, 'labelsInLogic.mat'))['resultLabelsInLogic']
    
    clip_markers = sio.loadmat(os.path.join(data_path, 'clipMarkers.mat'))['clipMarkers']
    return {'samples': samples, 'labels': labels, 'logical_labels': logical_labels, 'clip_markers': clip_markers}    
    
    
    
def prepareData_LMDB(data_path, 
                     batch_size,
                     DB_NAME_SAMPLES_TRAIN,
                     DB_NAME_SAMPLES_TEST,
                     DB_NAME_LABLES_TRAIN,
                     DB_NAME_LABELS_TEST,
                     DB_NAME_CLIP_MARKERS_TRAIN,
                     DB_NAME_CLIP_MARKERS_TEST,
                     DB_NAME_LOGICAL_LABLES_TRAIN,
                     DB_NAME_LOGICAL_LABELS_TEST,
                     DB_NAME_SAMPLE_INDEXES_TRAIN,
                     DB_NAME_SAMPLE_INDEXES_TEST):
     
    # Why batch_size is needed?
    
    data = loadData(data_path)
    
    samples = data['samples'].astype('uint8')
    labels = data['labels'].astype('int')
    logical_labels = data['logical_labels'].astype('int')
    clip_markers = data['clip_markers'].astype('int')
    
    [speech_num_per_person, person_num, frame_num_per_speech, frame_height, frame_width,] = samples.shape
        
    # Shuffling persons and dividing to training set and testing set
    person_index = np.linspace(0, person_num - 1, person_num).astype('int')
    
    person_index_shuffled = rd.shuffle(person_index)
    person_index_shuffled = person_index
    
    person_index_train = person_index_shuffled[0: (np.ceil(0.8 * person_num)).astype('int')]
    person_index_test = person_index_shuffled[np.ceil(0.8 * person_num).astype('int'): person_num]

    
    samples_train = samples[ :, person_index_train, :, :, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech, frame_height, frame_width))
    samples_test = samples[ :, person_index_test, :, :, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech, frame_height, frame_width))
    
    clip_markers_train = clip_markers[ :, person_index_train, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech))
    clip_markers_test = clip_markers[ :, person_index_test, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech))
    
    label_types_num = 10
    logical_labels_train = logical_labels[ :, person_index_train, :, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech, label_types_num))
    logical_labels_test = logical_labels[ :, person_index_test, :, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech, label_types_num))

    labels_train = labels[ :, person_index_train, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech))
    labels_test = labels[ :, person_index_test, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech))

    # Create a local data identifier 
    # Create a local samples identifirer
    # Create a local sample identifier 
    # Create a local sample identifier 
    
    # Size speech_num_per_person*person_index*frame_num_per_speech*
    
    sample_indexes_train = labels[ :, person_index_train, : ]
    
    for i in range(0, sample_indexes_train.shape[1]) :
        for j in range(0, sample_indexes_train.shape[0]) :
            for k in range(0, sample_indexes_train.shape[2]) :
                sample_indexes_train[j,i,k] = 10000 * person_index_train[i] + 100 * j + k
                 
    sample_indexes_test = labels[ :, person_index_test, : ]
    
    for i in range(0, sample_indexes_test.shape[1]) :
        for j in range(0, sample_indexes_test.shape[0]) :
            for k in range(0, sample_indexes_test.shape[2]) :
                sample_indexes_test[j,i,k] = 10000 * person_index_test[i] + 100 * j + k
                
                
    sample_indexes_train = sample_indexes_train.reshape(labels_train.shape)
    sample_indexes_test = sample_indexes_test.reshape(labels_test.shape)
    
    
    
    # Shuffling the data
    
    sequence_index_train = np.linspace(0, labels_train.shape[0] - 1, labels_train.shape[0]).astype('int')
    sequence_index_test = np.linspace(0, labels_test.shape[0] - 1, labels_test.shape[0]).astype('int')
    
    rd.shuffle(sequence_index_train)
    rd.shuffle(sequence_index_test)
    
    sample_indexes_train = sample_indexes_train[sequence_index_train,:]
    sample_indexes_test = sample_indexes_test[sequence_index_test,:]
    
    labels_train = labels_train[sequence_index_train,:]
    labels_test = labels_test[sequence_index_test,:]
    
    clip_markers_train = clip_markers_train[sequence_index_train,:]
    clip_markers_test = clip_markers_test[sequence_index_test,:]
    
    logical_labels_train = logical_labels_train[sequence_index_train,:,:]
    logical_labels_test = logical_labels_test[sequence_index_test,:,:]
    
    samples_train = samples_train[sequence_index_train,:,:,:]
    samples_test = samples_test[sequence_index_test,:,:,:]

    
    insert_data_to_DB(sample_indexes_train[:,:,newaxis, newaxis, newaxis], batch_size, DB_NAME_SAMPLE_INDEXES_TRAIN)  
    insert_data_to_DB(sample_indexes_test[:,:,newaxis, newaxis, newaxis], 20, DB_NAME_SAMPLE_INDEXES_TEST)     
   
    insert_data_to_DB(samples_train[:,:,newaxis,:,:], batch_size, DB_NAME_SAMPLES_TRAIN)  
    insert_data_to_DB(samples_test[:,:,newaxis,:,:], 20, DB_NAME_SAMPLES_TEST) 
     
    insert_data_to_DB(clip_markers_train[:,:,newaxis, newaxis, newaxis], batch_size, DB_NAME_CLIP_MARKERS_TRAIN)  
    insert_data_to_DB(clip_markers_test[:,:,newaxis, newaxis, newaxis], 20, DB_NAME_CLIP_MARKERS_TEST)  
    
    insert_data_to_DB(logical_labels_train[:,:,newaxis,newaxis,:], batch_size, DB_NAME_LOGICAL_LABLES_TRAIN)  
    insert_data_to_DB(logical_labels_test[:,:,newaxis,newaxis,:], 20, DB_NAME_LOGICAL_LABELS_TEST)  
    
    insert_data_to_DB(labels_train[:,:,newaxis, newaxis, newaxis], batch_size, DB_NAME_LABLES_TRAIN)  
    insert_data_to_DB(labels_test[:,:,newaxis, newaxis, newaxis], 20, DB_NAME_LABELS_TEST)  
    
def insert_data_to_DB (data, batch_size, DB_NAME):  
    
    if os.path.isdir(DB_NAME) == True:
        shutil.rmtree(DB_NAME)
    
    # input shape
    # sequence_num, frame_num_per_sequence, channel, frame_height, frame_width
    #      N                   T               K          H             W
    [sequence_num, 
     frame_num_per_sequence,
     channel,    # fake 
     frame_height, 
     frame_width] = data.shape
    
    

    
    
    sequence_num_per_batch = batch_size / frame_num_per_sequence
    if frame_height == 1:
        DB_MAP_SIZE = sequence_num * frame_height * frame_width * frame_num_per_sequence * 64
    else :    
        DB_MAP_SIZE = sequence_num * frame_height * frame_width * frame_num_per_sequence * 16
    
    DB_ENV = lmdb.Environment(DB_NAME, DB_MAP_SIZE)
    
    DB_TXN = DB_ENV.begin(write = True, buffers = True)
    
    # prerequesit 
    # (sequence_num / seuqnce_num_per_batch == 0 ) == true
    batch_num = sequence_num / sequence_num_per_batch
    
    record_key = 0
    
    for i in range(0, batch_num):
        tmp_data_block = data[(i * sequence_num_per_batch) : (i * sequence_num_per_batch + sequence_num_per_batch), :, :, :, :]
        #tmp_data_block + tmp_data_block.transpose((1 , 0 , 2 , 3))
        
        for T in range(0, tmp_data_block.shape[1]) :
            
            for N in range(0, tmp_data_block.shape[0]) :
                
                datum = caffe.io.array_to_datum(tmp_data_block[N, T, :, :, :])
                
                record_key = record_key + 1
                
                key = '{:08}'.format(record_key)
                
                DB_TXN.put(key.encode('ascii'), datum.SerializeToString())
    
    print(record_key)
    DB_TXN.commit()
    DB_ENV.close()   
    