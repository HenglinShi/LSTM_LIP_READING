'''
Created on Aug 17, 2016

@author: hshi
'''
import caffe
import os
import scipy.io as sio
import random as rd
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
from LSTM_Data_Module import prepareData_LMDB
from LSTM_Net_Define_Module import creatNet
from pylab import *
import shutil
import lmdb

def loadData(data_path):
    samples = sio.loadmat(os.path.join(data_path, 'samples.mat'))['resultSamples']
    samples = samples.transpose((0,1,4,2,3))
    
    labels = sio.loadmat(os.path.join(data_path, 'labels.mat'))['resultLabels']
    
    logical_labels = sio.loadmat(os.path.join(data_path, 'labelsInLogic.mat'))['resultLabelsInLogic']
    
    clip_markers = sio.loadmat(os.path.join(data_path, 'clipMarkers.mat'))['clipMarkers']
    
    
    return {'samples': samples, 'labels': labels, 'logical_labels': logical_labels, 'clip_markers': clip_markers}    

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
    
    
def createSolver(solver_path, net_path, base_lr = 0.001):
    solver = caffe_pb2.SolverParameter()
    
    solver.net = net_path
  
    solver.lr_policy = 'step'
    
    solver.gamma = 0.1
    
    solver.base_lr = base_lr
    
    solver.stepsize = 10000
    
    solver.display = 2000
    
    solver.max_iter = 50000
    
    solver.test_iter.append(150)
    
    solver.test_interval = 2000
    
    solver.momentum = 0.9
    
    solver.weight_decay = 0.005
    
    solver.snapshot = 5000
    
    solver.snapshot_prefix = 'snapshot_lstm_lip_reading'
    
    solver.solver_mode = caffe_pb2.SolverParameter().GPU
    
    solver.random_seed = 1701
    
    solver.average_loss = 1000
    
    solver.clip_gradients = 5
    
    return solver



def main ():
    
    data_path = "Data"
    

    batch_size_train = 60
    batch_size_test = 20 
    folds_CV = 10      
          
    DB_PREFIX = './Experiment/'      
    _DB_NAME_SAMPLE_TRAIN = '/SAMPLE_TRAIN'
    _DB_NAME_SAMPLE_TEST = '/SAMPLE_TEST'
    _DB_NAME_LABEL_TRAIN = '/LABEL_TRAIN'
    _DB_NAME_LABEL_TEST = '/LABEL_TEST'
    _DB_NAME_CLIP_MARKER_TRAIN = '/CLIP_MARKER_TRAIN'
    _DB_NAME_CLIP_MARKER_TEST = '/CLIP_MARKER_TEST'
    _DB_NAME_LOGICAL_LABEL_TRAIN = '/LOGICAL_LABEL_TRAIN'
    _DB_NAME_LOGICAL_LABEL_TEST = '/LOGICAL_LABEL_TEST'  
    _DB_NAME_SAMPLE_INDEX_TRAIN = '/SAMPLE_INDEX_TRAIN'
    _DB_NAME_SAMPLE_INDEX_TEST = '/SAMPLE_INDEX_TEST'

    
    
    
    
    
    
    
    
    # Load the data
    data = loadData(data_path)
    
    # Create splitting index
    samples = data['samples'].astype('uint8')
    labels = data['labels'].astype('int')
    logical_labels = data['logical_labels'].astype('int')
    clip_markers = data['clip_markers'].astype('int')
    
    label_types_num = len(np.unique(labels))
    
    [speech_num_per_person, person_num, frame_num_per_speech, frame_height, frame_width,] = samples.shape

    # Shuffling persons and dividing to training set and testing set
    person_index = np.linspace(0, person_num - 1, person_num).astype('int') 
    # person_index = range(pers)
    rd.shuffle(person_index)
    
    step_CV = np.floor(person_num / folds_CV)
    #sample_shape_train = [speech_num_per_person * (len(person_index) - step_CV), frame_num_per_speech, frame_height, frame_width]
    #sample_shape_test = [speech_num_per_person * step_CV, frame_num_per_speech, frame_height, frame_width]
    
    
    # Initializing tmp variable


    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    solver_max_iter = 50000
    solver_test_interval = 2000
    solver_test_iter = 150
    test_times = (int)(np.ceil(float32(solver_max_iter) / float32(solver_test_interval)))
    
    test_acc = zeros((folds_CV, test_times))
      
    
    
    for ite_folds in range(folds_CV):
        
        nn = ite_folds + 1     
                       
        DB_NAME_SAMPLE_TRAIN = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_SAMPLE_TRAIN
           
        DB_NAME_SAMPLE_TEST = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_SAMPLE_TEST
        DB_NAME_LABEL_TRAIN = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_LABEL_TRAIN
        DB_NAME_LABEL_TEST = DB_PREFIX + str((ite_folds + 1))  + _DB_NAME_LABEL_TEST
        DB_NAME_CLIP_MARKER_TRAIN = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_CLIP_MARKER_TRAIN
        DB_NAME_CLIP_MARKER_TEST = DB_PREFIX + str((ite_folds + 1))  + _DB_NAME_CLIP_MARKER_TEST
        DB_NAME_LOGICAL_LABEL_TRAIN = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_LOGICAL_LABEL_TRAIN
        DB_NAME_LOGICAL_LABEL_TEST = DB_PREFIX + str((ite_folds + 1))  + _DB_NAME_LOGICAL_LABEL_TEST
        DB_NAME_SAMPLE_INDEX_TRAIN = DB_PREFIX + str((ite_folds + 1))  + _DB_NAME_SAMPLE_INDEX_TRAIN
        DB_NAME_SAMPLE_INDEX_TEST = DB_PREFIX + str((ite_folds + 1)) + _DB_NAME_SAMPLE_INDEX_TEST
        

        
        person_index_test = person_index[(step_CV * ite_folds):(step_CV * ite_folds + step_CV)]  
        person_index_train = np.setdiff1d(person_index, person_index_test)
        
        rd.shuffle(person_index_train)
        
        #person_index[[0:(step_CV * ite_folds)], [(step_CV * ite_folds + step_CV) : person_num]]
            
        
        
        samples_train = samples[ :, person_index_train, :, :, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech, frame_height, frame_width))
        samples_test = samples[ :, person_index_test, :, :, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech, frame_height, frame_width))
                
        logical_labels_train = logical_labels[ :, person_index_train, :, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech, label_types_num))
        logical_labels_test = logical_labels[ :, person_index_test, :, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech, label_types_num))
            
        clip_markers_train = clip_markers[ :, person_index_train, : ].reshape((speech_num_per_person * len(person_index_train), frame_num_per_speech))
        clip_markers_test = clip_markers[ :, person_index_test, : ].reshape((speech_num_per_person * len(person_index_test), frame_num_per_speech))
    
        labels_train = labels[ :, person_index_train, : ].reshape(clip_markers_train.shape)
        labels_test = labels[ :, person_index_test, : ].reshape(clip_markers_test.shape)
           
        #Create Sample Index
        sample_indexes_train = labels[ :, person_index_train, : ]
        
        for i in range(0, sample_indexes_train.shape[1]) :
            for j in range(0, sample_indexes_train.shape[0]) :
                for k in range(0, sample_indexes_train.shape[2]) :
                    sample_indexes_train[j,i,k] = 10000 * (person_index_train[i] + 1) + 100 * (j + 1) + k + 1
                     
        sample_indexes_test = labels[ :, person_index_test, : ]
        
        for i in range(0, sample_indexes_test.shape[1]) :
            for j in range(0, sample_indexes_test.shape[0]) :
                for k in range(0, sample_indexes_test.shape[2]) :
                    sample_indexes_test[j,i,k] = 10000 * (person_index_test[i] + 1) + 100 * (j + 1) + k + 1
                    
                    
        sample_indexes_train = sample_indexes_train.reshape(labels_train.shape)
        sample_indexes_test = sample_indexes_test.reshape(labels_test.shape)
    
        
        # Shuffling the data separately for traing set and testing set
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
        
        insert_data_to_DB(sample_indexes_train[:,:,newaxis, newaxis, newaxis], batch_size_train, DB_NAME_SAMPLE_INDEX_TRAIN)  
        insert_data_to_DB(sample_indexes_test[:,:,newaxis, newaxis, newaxis], batch_size_test, DB_NAME_SAMPLE_INDEX_TEST)     
       
        insert_data_to_DB(samples_train[:,:,newaxis,:,:], batch_size_train, DB_NAME_SAMPLE_TRAIN)  
        insert_data_to_DB(samples_test[:,:,newaxis,:,:], batch_size_test, DB_NAME_SAMPLE_TEST) 
         
        insert_data_to_DB(clip_markers_train[:,:,newaxis, newaxis, newaxis], batch_size_train, DB_NAME_CLIP_MARKER_TRAIN)  
        insert_data_to_DB(clip_markers_test[:,:,newaxis, newaxis, newaxis], batch_size_test, DB_NAME_CLIP_MARKER_TEST)  
        
        insert_data_to_DB(logical_labels_train[:,:,newaxis,newaxis,:], batch_size_train, DB_NAME_LOGICAL_LABEL_TRAIN)  
        insert_data_to_DB(logical_labels_test[:,:,newaxis,newaxis,:], batch_size_test, DB_NAME_LOGICAL_LABEL_TEST)  
        
        insert_data_to_DB(labels_train[:,:,newaxis, newaxis, newaxis], batch_size_train, DB_NAME_LABEL_TRAIN)  
        insert_data_to_DB(labels_test[:,:,newaxis, newaxis, newaxis], batch_size_test, DB_NAME_LABEL_TEST)  
        
        
        
        solver_path = './Experiment/' + str(ite_folds + 1) + '/solver.prototxt'
        net_path = './Experiment/' + str(ite_folds + 1) + '/train_test_net.prototxt'
        
        solver = None
        solver = createSolver(solver_path, net_path, 0.01)
        with open(solver_path, 'w') as f:    
            f.write(str(solver))
    
        solver = None
        solver = caffe.SGDSolver(solver_path)
        rest_train_iter = solver_max_iter
        test_index = 0

        while rest_train_iter > 0 :
            
            if (int)(rest_train_iter / solver_test_interval) > 0:
                solver.step(solver_test_interval)
                       
            else :
                solver.step(rest_train_iter)
            
            rest_train_iter = rest_train_iter - solver_test_interval
            
            correct = 0
            for iter_test in range(solver_test_iter):
                
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['accuracy'].data)

            test_acc[ite_folds, test_index] = correct / solver_test_iter
            
            test_index = test_index + 1 
    
    
    sio.savemat('./Output/acc.mat', {'acc':test_acc})
            
if __name__ == '__main__':
    main()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    