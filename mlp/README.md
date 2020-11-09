## Multi Layer Perceptron on GPU

Simple MLP CPU
- simple_mlp_CPU.ipynb
- simple mlp with variable number of layers
- generate multi-class dataset

Simple MLP GPU Single
- simple_mlp_GPU_single.ipynb
- put model and layers to CUDA device
- remove synchronization points in training loop to load host arrays from pin_memory directly

Simple MLP DDP with Wrapper
- simple_mlp_DDP_with_wrapper.py
- simple_mlp_DDP_with_wrapper_output.txt
- dist.init_process_group with nccl backend for each process
- use DistributedSampler on dataloaders to make sure each process gets a different slice of the current training batch
- wrap model with DDP wrapper
- check that parameters are synced for all processes
- check that gradients are averaged and synced after loss.backward() is called
- note that local loss is different across processes, only gradients are sent across network
- run validation on entire validation set only on rank 0 process



## Readings

- Standard wrapper, wrap optimizer and model in special class. 
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
- torch.distributed.init_process_group() documentation
- Distributed All Reduce; only sync operations that I need
- Distributed Package and Functions. Backend types.
- https://pytorch.org/docs/stable/distributed.html
- Manually assign tensors to individual GPUs (CUDA semantics). How to record time. Memory management. PIN memory. Environmental Variable Init.
- https://pytorch.org/docs/stable/notes/cuda.html

- Overview
- https://pytorch.org/tutorials/beginner/dist_overview.html
- Data Parallel
- https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html
- https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
- nn.parallel primitives
- https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
- Distributed Data Parallel DDP explanation. Include pointers to source code
- https://pytorch.org/docs/stable/notes/ddp.html
- https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
- DDP, Multi-processing example. Distributed Sampler
- https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
