# Zero Stability Well Predicts Performance of Convolutional Neural Networks

Official Pytorch implementation of paper "Zero Stability Well Predicts Performance of Convolutional Neural Networks"


### Run ZeroSNet44_Opt on CIFAR-10: 

```Bash
CUDA_VISIBLE_DEVICES=0 python train_cifar_ZeroSNet.py --arch ZeroSNet44_Opt --dataset cifar10
```

### Run ZeroSNet56_Tra on CIFAR-100: 

```Bash
CUDA_VISIBLE_DEVICES=0 python train_cifar_ZeroSNet.py --arch ZeroSNet56_Tra --dataset cifar100
```

### Run ZeroSNet_Opt on ImageNet: 

```Bash
CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 main_ZeroSNet_IN.py --arch zerosnet18_in -bs 128 --lr 0.2 --opt_level O2 --data <your data set path>  --workers 8 --given_coe 0.3333333 0.5555556 0.1111111 1.77777778 
```


