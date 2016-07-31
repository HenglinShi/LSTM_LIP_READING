'''
Created on Jul 27, 2016

@author: henglinshi
'''
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
from LSTM_Data_Module import prepareData_LMDB
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
    
def creatNet(data_path, 
             batch_size,
             DB_NAME_SAMPLES,
             DB_NAME_LABELS,
             DB_NAME_CLIP_MARKERS,
             DB_NAME_LOGICAL_LABELS,       
             image_height,
             image_width,
             channel_num,
             image_num_per_sequence):

    
    sequence_num_per_batch = batch_size / image_num_per_sequence
    
    
    # DEFINE THE NETWORK ARCHETECTURE
    net = caffe.NetSpec()
    
    # DATA LAYER
    # SAMPLE AND LABEL
    net.samples = L.Data(batch_size = batch_size,
                         backend = P.Data.LMDB,
                         source = DB_NAME_SAMPLES)
    
 
    
    net.labels = L.Data(batch_size = batch_size,
                        backend = P.Data.LMDB,
                        source = DB_NAME_LABELS)
    
    net.clip_markers = L.Data(batch_size = batch_size,
                              backend = P.Data.LMDB,
                              source = DB_NAME_CLIP_MARKERS)
    
    
    
    net.reshape_sample_1 = L.Reshape(net.samples,
                                     reshape_param = { 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch, image_height * image_width] } })
        
    
    # Reshaple lable
    net.reshape_label_1 = L.Reshape(net.labels,
                                    reshape_param = { 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch]}})
    
    # Reshape clip markers
    net.reshape_clip_markers_1 = L.Reshape(net.clip_markers,
                                           reshape_param  = { 'shape': {'dim': [image_num_per_sequence, sequence_num_per_batch]}})

    
    # LSTM network
    
    net.lstm_1 = L.LSTM(net.reshape_sample_1, 
                        net.reshape_clip_markers_1,
                        recurrent_param = {'num_output': 256,
                                           'weight_filler': {'type': 'uniform', 'min': -0.01, 'max': 0.01},
                                           'bias_filler': {'type': 'constant', 'value': 0 }})
    
    # INNERPRODUCT LAYER
    net.inner_product_1 = L.InnerProduct(net.lstm_1,
                                         inner_product_param = {'num_output': 10,
                                                                'weight_filler': {'type': 'gaussian', 'std': 0.1},
                                                                'bias_filler': {'type': 'constant'},
                                                                'axis': 2})
    
    
    # LOSS LAYER
    net.loss = L.SoftmaxWithLoss(net.inner_product_1,
                                 net.reshape_label_1, 
                                 softmax_param = {'axis': 2})
    
    # Accuracy layer
    net.accuracy = L.Accuracy(net.inner_product_1, 
                              net.reshape_label_1,
                              accuracy_param = {'axis': 2})
    
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

    
def main():
    
    # Prepare data
    data_path = 'Data'
    batch_size = 80
    DB_NAME_SAMPLES_TRAIN = 'SAMPLES_TRAIN'
    DB_NAME_SAMPLES_TEST = 'SAMPLES_TEST'
    DB_NAME_LABELS_TRAIN = 'LABELS_TRAIN'
    DB_NAME_LABELS_TEST = 'LABELS_TEST'
    DB_NAME_CLIP_MARKERS_TRAIN = 'CLIP_MARKERS_TRAIN'
    DB_NAME_CLIP_MARKERS_TEST = 'CLIP_MARKERS_TEST'
    DB_NAME_LOGICAL_LABELS_TRAIN = 'LOGICAL_LABELS_TRAIN'
    DB_NAME_LOGICAL_LABELS_TEST = 'LOGICAL_LABELS_TEST'  
    
    
    # put the data to the lmdb
    # prepareData_LMDB(data_path,batch_size,DB_NAME_SAMPLES_TRAIN,DB_NAME_SAMPLES_TEST,DB_NAME_LABELS_TRAIN,DB_NAME_LABELS_TEST,DB_NAME_CLIP_MARKERS_TRAIN,DB_NAME_CLIP_MARKERS_TEST,DB_NAME_LOGICAL_LABELS_TRAIN, DB_NAME_LOGICAL_LABELS_TEST)
    
    
    # create train net
    train_net_path = 'train_net.prototxt'
    test_net_path = 'test_net.prototxt'
 
    train_net = creatNet(data_path, 
                   batch_size,
                   DB_NAME_SAMPLES_TRAIN,
                   DB_NAME_LABELS_TRAIN,
                   DB_NAME_CLIP_MARKERS_TRAIN,
                   DB_NAME_LOGICAL_LABELS_TRAIN,       
                   image_height = 40,
                   image_width = 50,
                   channel_num = 1,
                   image_num_per_sequence = 20)

    #save net
    with open(train_net_path, 'w') as f:
        f.write(str(train_net.to_proto()))
    
    
    test_net = creatNet(data_path, 
                       batch_size,
                       DB_NAME_SAMPLES_TEST,
                       DB_NAME_LABELS_TEST,
                       DB_NAME_CLIP_MARKERS_TEST,
                       DB_NAME_LOGICAL_LABELS_TEST,       
                       image_height = 40,
                       image_width = 50,
                       channel_num = 1,
                       image_num_per_sequence = 20)
    
    
        #save net
    with open(test_net_path, 'w') as f:
        f.write(str(test_net.to_proto()))
    
    # create solver
    solver_path = 'solver.prototxt'
    solver = createSolver(solver_path, train_net_path, test_net_path, 0.1)
    
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


    solver_max_iter = 10000
    solver_test_interval = 100
    solver_test_iter = 100
    train_loss = zeros(solver_max_iter)
    test_acc = zeros(int(np.ceil(solver_max_iter / solver_test_interval)))
    output = zeros((solver_max_iter, batch_size, 10))

    # the main solver loop
    for it in range(solver_max_iter):
        
        solver.step(1)  # SGD by Caffe
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
    
        if it % solver_test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(solver_test_iter):
                
                solver.test_nets[0].forward()
                
                correct += sum(solver.test_nets[0].blobs['accuracy'].data)
                
            test_acc[it // solver_test_interval] = correct / 100
    



if __name__ == '__main__':
    main()