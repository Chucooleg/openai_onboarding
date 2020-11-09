import os
import numpy as np
from datetime import datetime
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
# import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    """
    Toy dataset for multiclass classification.
    """

    def __init__(self, data_dir):
        # shape (m, nx)
        self.X = np.load(os.path.join(data_dir, 'features.npy'))
        # shape (m, ny=1)
        self.y = np.load(os.path.join(data_dir, 'labels.npy'))
        

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            X = torch.from_numpy(self.X[idx, :]).type(torch.FloatTensor)
            y = torch.tensor(self.y[idx], dtype=torch.long)
            sample = {'X': X, 'y': y}

        return sample

    
class MLPLazy(nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(self, nx, hidden_layer_dims, ny):
        super(MLPLazy, self).__init__()
        self.hidden_layer_dims = hidden_layer_dims
        
        linear_layers = []
        last_dim = nx
        for next_dim in hidden_layer_dims:
            linear_layer = nn.Linear(last_dim, next_dim)
            linear_layers.append(linear_layer)
            last_dim = next_dim
        # GPU: should push to ModuleList so that params stay on cuda
        self.linear_layers = nn.ModuleList(linear_layers)
        
        self.scorer = nn.Linear(last_dim, ny)

    def forward(self, X):
        '''
        X has shape (m, nx)
        '''
        last_X = X
        for i, linear_layer in enumerate(self.linear_layers):
            # shape (m, self.hidden_layer_dims[i])
            last_X = linear_layer(last_X)
            # shape (m, self.hidden_layer_dims[i])
            last_X = torch.relu(last_X)
        # shape (m, ny)
        z = self.scorer(last_X)
        # shape (m, ny)
        a = torch.softmax(z, dim=1)
        return z, a
    
    
def print_process_gradients_and_params(epoch_i, batch_i, gpu, rank, model):
    print('Gradients for : epoch {} batch{} gpu {}, rank {}\n'.format(
        epoch_i, batch_i, gpu, rank), next(model.parameters()).grad.data)
    print('Param vals for : epoch {} batch{} gpu {}, rank {}\n'.format(
        epoch_i, batch_i, gpu, rank), next(model.parameters()).data)        
    
    
def run_train(gpu, args, model, train_loader, valid_loader, loss_criterion, optimizer):
    '''
    Train model and report losses on train and dev sets per epoch
    '''
    
    # DDP: keep track of where we are in stdout
    rank = args.nr * args.gpus + gpu
    print("Running run_train inside gpu {}, rank {}".format(gpu, rank))
    
    history = {
        'train_losses': [],
        'valid_losses': [],        
        'valid_accuracy': [],
    }
    
    for epoch_i in range(args.epochs):
        
        # train
        model.train()
        sum_batch_losses = torch.tensor([0.], dtype=torch.float, device=gpu)
        
        for batch_i, batch_data in enumerate(train_loader):
            
            # GPU: used pinned memory (uncomment non_blocking)
            batch_X = batch_data['X'].cuda(gpu)#, non_blocking=True)
            batch_y = batch_data['y'].cuda(gpu)#, non_blocking=True)
            logits, activations = model(batch_X)
            
            # DDP: this is local device loss
            loss = loss_criterion(logits, batch_y)
            
            optimizer.zero_grad()
           
            # DDP: averages the gradient when this is called
            # DDP: This is a synchronization point
            loss.backward()
            
            # Debug DDP BEFORE Optim Step: check if gradients are params are synced
            if epoch_i == args.epochs - 1 and batch_i == 0:
                print_process_gradients_and_params(epoch_i, batch_i, gpu, rank, model)
                
            # To check local gradients instead of global averaged:
            # https://discuss.pytorch.org/t/distributeddataparallel-gradient-print/49767/2
                
            optimizer.step()
            
            # Debug DDP AFTER Optim Step: check if gradients are params are synced
            if epoch_i == args.epochs - 1 and batch_i == 0:
                print_process_gradients_and_params(epoch_i, batch_i, gpu, rank, model)
                 
            sum_batch_losses += loss
            
            # Debug DDP: report local device loss
            print('Rank {}, Epoch {}, batch {}, loss={}'.format(rank, epoch_i, batch_i, loss.item()))
            
        num_batches = batch_i + 1.
        history['train_losses'].append(sum_batch_losses/num_batches)

        # validate on rank 0 device
        if rank == 0:
            val_sum_batch_losses, val_sum_batch_accuracies, val_num_batches = \
                run_pred(gpu, args, model, valid_loader, loss_criterion)
            history['valid_losses'].append(val_sum_batch_losses / val_num_batches)
            history['valid_accuracy'].append(val_sum_batch_accuracies / val_num_batches)
    
    # GPU: Use pinned memory in dataloader for faster data transfer
    # do not add synchronization point until training loop has ended
    # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
    # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    itemize = lambda x: [tensor_val.item() for tensor_val in x]
    history['train_losses'] = itemize(history['train_losses'])
    history['valid_losses'] = itemize(history['valid_losses'])
    history['valid_accuracy'] = itemize(history['valid_accuracy'])
    
    return history


@torch.no_grad()
def run_pred(gpu, args, model, test_loader, loss_criterion):
    '''Propogate forward on dev or test set, report loss and accuracy.'''
    
    # evaluate
    model.eval()
    sum_batch_losses = torch.tensor([0.], dtype=torch.float, device=gpu)
    sum_batch_accuracies = torch.tensor([0.], dtype=torch.float, device=gpu)
    for batch_i, batch_data in enumerate(test_loader):
        batch_X = batch_data['X'].cuda(gpu, non_blocking=True)
        batch_y = batch_data['y'].cuda(gpu, non_blocking=True)
        logits, activations = model(batch_X)
        # Question: is this just computing the local loss?
        loss = loss_criterion(logits, batch_y)
        sum_batch_losses += loss
        _, max_index = torch.max(logits, dim=1)
        accuracy = torch.mean(max_index.eq(batch_y).type(torch.FloatTensor))
        sum_batch_accuracies += accuracy
    num_batches_computed = batch_i + 1.
    
    return sum_batch_losses, sum_batch_accuracies, num_batches_computed


def train(gpu, args):
    
    ################################################################
    
    # DDP: keep track of where we are in stdout
    rank = args.nr * args.gpus + gpu
    print("Running train inside gpu {}, rank {}".format(gpu, rank))
    print_cuda_device_memory(gpu)
    
    # DDP: initiate process group, once for each process
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=args.world_size, rank=rank) 
    
    ################################################################

    torch.manual_seed(args.seed)
  
    # load dataset
    training_set = ToyDataset(data_dir=os.path.join(args.data_dir, args.dataset_dir, 'train'))
    validation_set = ToyDataset(data_dir=os.path.join(args.data_dir, args.dataset_dir, 'valid'))

    # DDP: different process should get a different slice of data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_set,
        num_replicas=args.world_size,
        rank=rank
    )

    # DDP: Use sampler and don't shuffle the usual way
    # Note: with batch size, num_workers=2 is slower.
    training_generator = torch.utils.data.DataLoader(
        dataset=training_set, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2, 
        pin_memory=True,
        sampler=train_sampler
    )
    validation_generator = torch.utils.data.DataLoader(
        dataset=validation_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
#         sampler=valid_sampler
    )   
    
    
    nx = training_set.X.shape[1]
    ny = max(training_set.y) + 1
    
    print('Train set X shape:', training_set.X.shape)
    print('Train set y shape:', training_set.y.shape)
    print('Valid set X shape:', validation_set.X.shape)
    print('Valid set y shape:', validation_set.y.shape)
    
    ################################################################
    
    # GPU: put model to GPU
    model = MLPLazy(nx, args.hidden_layer_dims, ny)
    torch.cuda.set_device(gpu)
    model.to(device=gpu)
    
    # GPU: needs loss criterion is also a layer with parameters
    loss_criterion = nn.CrossEntropyLoss(reduction='mean').cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)   
    
    # DDP: Wrap model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    
    ################################################################
    
    start = datetime.now()
    history = run_train(gpu, args, model, training_generator, validation_generator, loss_criterion, optimizer)
    
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    
    print_cuda_device_memory(gpu)
    
    print(history)
    return history


def print_cuda_device_memory(gpu):
    cuda = torch.device('cuda:{}'.format(gpu)) 
    print(torch.cuda.memory_summary(cuda))
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    
    # TODO: merge into arguments when necesscary
    args.data_dir = '/datadrive'
    args.dataset_dir = 'toy_mlp_1'
    args.seed = 123
    args.batch_size = 500
    # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
#     args.hidden_layer_dims = [10000, 10000, 10000, 10000, 10000]
    args.hidden_layer_dims = [10, 10]
    args.lr = 0.01

    #########################################################
    args.world_size = args.gpus * args.nodes                
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '29500'
    print("mp is spawning processes...")
    mp.spawn(train, nprocs=args.gpus, args=(args,))         
    #########################################################

if __name__ == '__main__':
    main()
