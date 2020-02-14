# Query-efficient Meta Aattack to Deep Neural Networks
This repository contains the code for reproducing the experimental results of attacking mnist, cifar10, tiny-imagenet models, of our submission: Query-efficient Meta Aattack to Deep Neural Networks ([openreview](https://openreview.net/forum?id=Skxd6gSYDS)). The paper can be cited as follows:
```
@inproceedings{
Du2020Query-efficient,
title={Query-efficient Meta Attack to Deep Neural Networks},
author={Jiawei Du and Hu Zhang and Joey Tianyi Zhou and Yi Yang and Jiashi Feng},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Skxd6gSYDS}
}
```

## Reproducing the results
### Requirements
* Pytorch (`torch`, `torchvision`) packages
### For generating gradients for meta model training (take cifar10 for example)
`cd gen_grad/cifar_gen_grad_for_meta`

`python cifar_main.py`
### For training meta model to attack
`cd meta_training/cifar_meta_training`

`python cifar_train.py`
### For query-efficient attack
The results can be reproduced (with the default hyperparameters) with the following command:

`cd meta_attack/meta_attack_cifar`

`python test_all.py`
