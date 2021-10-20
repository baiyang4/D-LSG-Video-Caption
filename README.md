# Discriminative Latent Semantic Graph for Video Captioning
This repository is the implementation of "Discriminative Latent Semantic Graph for Video Captioning" ACM MM 2021.


## Prepare 
1. Create two empty folders, `data` and `caption-eval`
2. Download visual and text features of [MSVD](https://rec.ustc.edu.cn/share/f9335ba0-ba07-11ea-9198-9366c81a1928) 
and [MSR-VTT](https://rec.ustc.edu.cn/share/26685ac0-ba08-11ea-866f-6fc664dfaa3b), and put them in `data` folder. (from [RMN](https://github.com/tgc1997/RMN))
3. Download [evaluation tool](https://www.dropbox.com/sh/1h7jguu8z33a5a8/AAClLvIP-cxsiitvAVOPAG_ha?dl=0), and put them in `caption-eval` folder.

## Train 
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python \-m torch.distributed.launch \--nproc_per_node=4 train_debug.py
```

## D-LSD loss
![image](https://github.com/baiyang4/D-LSG-Video-Caption/blob/main/dlsg_loss.png)
The figure above shows the training loss of the proposed D-LSG model. After the generator loss drops at the mid-stage of training, the caption loss and all evaluation metrics received better performance. This shows the effectiveness of using semantic information of a given sentence as discriminative information works on the video captioning task.
The optimal setting setting is no uploaded yet (forgot to save the optimal setting, and will update with the optimal setting soon.)

## Acknowledgement
Our code is based on https://github.com/tgc1997/RMN. Thanks for their great works!
