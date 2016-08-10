'''
Created on Jul 27, 2016

@author: henglinshi
'''
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
from LSTM_Data_Module import prepareData_LMDB
from LSTM_Net_Define_Module import creatNet
from pylab import *

def createSolver(solver_path, train_net_path, test_net_path, base_lr = 0.001):
    solver = caffe_pb2.SolverParameter()
    
    solver.train_net = train_net_path
    
    solver.test_net.append(test_net_path)
    
    solver.test_iter.append(100)
    
    solver.test_interval = 100
    
    solver.lr_policy = 'step'
    
    solver.gamma = 0.1
    
    solver.base_lr = base_lr
    
    solver.stepsize = 10000
    
    solver.display = 200
    
    solver.max_iter = 10000
    
    solver.momentum = 0.9
    
    solver.weight_decay = 0.005
    
    solver.snapshot = 5000
    
    solver.snapshot_prefix = 'snapshot_lstm_lip_reading'
    
    solver.solver_mode = caffe_pb2.SolverParameter().CPU
    
    solver.random_seed = 1701
    
    solver.average_loss = 1000
    
    solver.clip_gradients = 5
    
    return solver

    
def main():
    
    # Prepare data
    data_path = 'Data'
    batch_size = 60
    DB_NAME_SAMPLES_TRAIN = './LMDB_DB/SAMPLES_TRAIN'
    DB_NAME_SAMPLES_TEST = './LMDB_DB/SAMPLES_TEST'
    DB_NAME_LABELS_TRAIN = './LMDB_DB/LABELS_TRAIN'
    DB_NAME_LABELS_TEST = './LMDB_DB/LABELS_TEST'
    DB_NAME_CLIP_MARKERS_TRAIN = './LMDB_DB/CLIP_MARKERS_TRAIN'
    DB_NAME_CLIP_MARKERS_TEST = './LMDB_DB/CLIP_MARKERS_TEST'
    DB_NAME_LOGICAL_LABELS_TRAIN = './LMDB_DB/LOGICAL_LABELS_TRAIN'
    DB_NAME_LOGICAL_LABELS_TEST = './LMDB_DB/LOGICAL_LABELS_TEST'  
    DB_NAME_SAMPLE_INDEX_TRAIN = './LMDB_DB/SAMPLE_INDEX_TRAIN'
    DB_NAME_SAMPLE_INDEX_TEST = './LMDB_DB/SAMPLE_INDEX_TEST'
    
    # put the data to the lmdb
    
    prepareData_LMDB(data_path,batch_size,DB_NAME_SAMPLES_TRAIN,DB_NAME_SAMPLES_TEST,DB_NAME_LABELS_TRAIN,DB_NAME_LABELS_TEST,DB_NAME_CLIP_MARKERS_TRAIN,DB_NAME_CLIP_MARKERS_TEST,DB_NAME_LOGICAL_LABELS_TRAIN, DB_NAME_LOGICAL_LABELS_TEST, DB_NAME_SAMPLE_INDEX_TRAIN, DB_NAME_SAMPLE_INDEX_TEST)
    
    
    # create train net
    train_net_path = 'train_net.prototxt'
    test_net_path = 'test_net.prototxt'
 
    train_net = creatNet(data_path, 
                   batch_size,
                   DB_NAME_SAMPLES_TRAIN,
                   DB_NAME_LABELS_TRAIN,
                   DB_NAME_CLIP_MARKERS_TRAIN,
                   DB_NAME_LOGICAL_LABELS_TRAIN,    
                   DB_NAME_SAMPLE_INDEX_TRAIN,   
                   image_height = 40,
                   image_width = 50,
                   channel_num = 1,
                   image_num_per_sequence = 20)

    #save net
    with open(train_net_path, 'w') as f:
        f.write(str(train_net.to_proto()))
    
    
    test_net = creatNet(data_path, 
                       20,
                       DB_NAME_SAMPLES_TEST,
                       DB_NAME_LABELS_TEST,
                       DB_NAME_CLIP_MARKERS_TEST,
                       DB_NAME_LOGICAL_LABELS_TEST,       
                       DB_NAME_SAMPLE_INDEX_TEST,
                       image_height = 40,
                       image_width = 50,
                       channel_num = 1,
                       image_num_per_sequence = 20)
    
    
        #save net
    with open(test_net_path, 'w') as f:
        f.write(str(test_net.to_proto()))
    
    # create solver
    solver_path = 'solver.prototxt'
    solver = createSolver(solver_path, train_net_path, test_net_path, 0.01)
    
    #save solver
    with open(solver_path, 'w') as f:
        f.write(str(solver))
    # Load net
    
    # starting training
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    solver = None
    solver = caffe.SGDSolver(solver_path)
    
    [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    solver.net.forward()
    solver.test_nets[0].forward()


    solver_max_iter = 100000
    solver_test_interval = 2000
    solver_test_iter = 100
    train_loss = zeros(solver_max_iter)
    test_acc = zeros(int(np.ceil(solver_max_iter / solver_test_interval)+1))
    output = zeros((solver_max_iter, batch_size, 10))
    
    tmp_sample_indexes = zeros((solver_max_iter, batch_size))
    tmp_labels=zeros((solver_max_iter,batch_size))
    tmp_clip_markers=zeros((solver_max_iter,batch_size))
    # the main solver loop
    for it in range(solver_max_iter):
        
        solver.step(1)  # SGD by Caffe
        
        #  print(np.unique(solver.net.blobs['labels'].data))
        
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        tmp_sample_indexes [it,:] = solver.net.blobs['sample_indexes'].data[:,0,0,0].reshape((1, batch_size))
        tmp_labels[it,:]=solver.net.blobs['labels'].data.reshape((1,batch_size))
        tmp_clip_markers[it,:]=solver.net.blobs['clip_markers'].data.reshape((1,batch_size))
        if it % solver_test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(solver_test_iter):
                
                solver.test_nets[0].forward()
                
                correct += sum(solver.test_nets[0].blobs['accuracy'].data)
                print (solver.test_nets[0].blobs['accuracy'].data)
                
            test_acc[it // solver_test_interval] = correct / solver_test_iter
    
    
    tmp_acc = test_acc/(int(np.ceil(solver_max_iter / solver_test_interval)))
            
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(solver_max_iter), train_loss)
    ax2.plot(solver_test_interval * arange(len(test_acc)), test_acc, 'r')
    
    import scipy.io as sio
    sio.savemat('./Output/inter_output.mat',{'sample_indexes':tmp_sample_indexes,'labels':tmp_labels,'clip_markers':tmp_clip_markers})
    sio.savemat('./Output/loss.mat',{'loss':train_loss})
    sio.savemat('./Output/acc.mat', {'acc':test_acc})
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))


if __name__ == '__main__':
    main()