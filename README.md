# Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks

Official implementation of [Ternary Spike AAAI2024](https://arxiv.org/pdf/2312.06372.pdf).

## Introduction

The Spiking Neural Network (SNN), as one of the biologically inspired neural network infrastructures, has drawn increasing attention recently. It adopts binary spike activations to transmit information, thus the multiplications of activations and weights can be substituted by additions, which brings high energy efficiency. However, in the paper, we theoretically and experimentally prove that the binary spike activation map cannot carry enough information, thus causing information loss and resulting in accuracy decreasing. To handle the problem, we propose a ternary spike neuron to transmit information. The ternary spike neuron can also enjoy the event-driven and multiplication-free operation advantages of the binary spike neuron but will boost the information capacity. 

### Dataset

ImageNet.

## Get Started

```
cd Ternary Spike
python -m torch.distributed.launch --nproc_per_node 8 --nnode 1 --master_port=25641 Train.py --spike --step 4
```

## Citation

```bash
@article{guo2023ternary,
  title={Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks},
  author={Guo, Yufei and Chen, Yuanpei and Liu, Xiaode and Peng, Weihang and Zhang, Yuhan and Huang, Xuhui and Ma, Zhe},
  journal={arXiv preprint arXiv:2312.06372},
  year={2023}
}
```
