# Does Federated Learning Really Need Backpropagation?

Here is the official implementation of the model BAFFLE in paper [Does Federated Learning Really Need Backpropagation?](https://arxiv.org/abs/2301.12195)

## Setup
### Install Package Dependencies
```
python >= 3.6
torch >= 1.2.0
torchvision >= 0.4.0
wandb
ast
torch_ema
```

## BAFFLE For MNIST with IID Participation ($C=10,\sigma=10^{-4}$)

### Evaluate BAFFLE with Different K

```
# For LeNet
python main.py --K 100 --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam --ema --normalization NoNorm 
# For WRN-10-2
python main.py --K 100 --net-name wideresnet --depth 10 --width 2 --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam  --ema
```

### Evaluate BAFFLE with Different Guidelines

```
# w/o EMA
python main.py --K 100 --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam --normalization NoNorm
# ReLU 
python main.py --K 100 --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam --normalization NoNorm --ema --activation "ReLU"
# SELU
python main.py --K 100 --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam --normalization NoNorm --ema --activation "SELU"
# Central Scheme
python main.py --K 50 --fd-format center  --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam  --normalization NoNorm --ema 
```

### Evaluate BAFFLE with Epoch-level Communications

```
# One Epoch Communication
python main.py --K 100 --fedopt FedAvg --dataset MNIST --epochs 20 --lr 1e-2 --optimizer Adam  --ema --normalization NoNorm 
```
