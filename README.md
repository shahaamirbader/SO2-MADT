The code is still being organized.🚧

# SO2-MADT

This is the code repository for our paper titled "Safe Offline-to-Online Multi-Agent Decision Transformer: A Safety
Conscious Sequence Modeling Approach", published in the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024.

Our paper can be accessed via the link : 

## Overview 

SO2-MADT is an innovative framework that revolutionizes safety considerations in Multi-agent Reinforcement Learning (MARL) through a novel sequence modeling approach. This codebase contains scripts to reproduce experiments. However, the SMAC dataset and MAMujoco datasets will need to be downloaded and installed as per the instructions contained in their respective repos. See following sections for more details. 

![Overview_4](https://github.com/user-attachments/assets/87cdb95f-aafa-43cc-aec9-2616a5f4769e)



## Instructions

We provide code in two sub-directories: `SMAC` containing code for Starcraft II experiments and `MAMujoco`containing code for Multi-Agent Mujoco experiments. 

The implementation is based on [PyMARL](https://github.com/oxwhirl/pymarl) framework. PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning. 

Setup the environment for StarCraft II Multi-agent Challenge as per the instructions provided in [SMAC](https://github.com/oxwhirl/smac) repo. While for Multi-Agent Mujoco, setup the environment as per the instructions provided in [MAMujoco](https://github.com/schroederdewitt/multiagent_mujoco) repo. 

## Getting Started

It is recommended to setup the environment and relevant libraries in Conda. 

Start by first cloning the repo:
```bash
git clone https://github.com/shahaamirbader/SO2-MADT && cd SO2-MADT
```
Then run the code as per the instructions mentioned in the following sections for SMAC and MAMujoco respectively: 

### SMAC
```
pip install -r SMAC/requirements.txt
```
### MAMujoco
```
pip install -r MAMujoco/requirements.txt
```

## Acknowledgement

Our code is inspired by following code bases:

Decision Transformer : https://github.com/kzl/decision-transformer

MADT : https://github.com/ReinholdM/Offline-Pre-trained-Multi-Agent-Decision-Transformer

SECA : https://github.com/DarkDawn233/SeCA

OSRL : https://github.com/liuzuxin/OSRL

CORL : https://github.com/corl-team/CORL

## Citation

Please cite our paper as:

```
@article{chen2021decisiontransformer,
  title={Safe Offline-to-Online Multi-Agent Decision Transformer: A Safety Conscious Sequence Modeling Approach},
  author={Aamir Bader Shah, Yu Wen, Jiefu Chen, Xuqing Wu and Xin Fu},
  journal={IROS},
  year={2024}
}
```

## License

MIT


