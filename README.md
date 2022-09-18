# PLL-DARLING
## 采用PACS，VACS数据集进行操作

## 主函数在 `python train97.py` 中,进行PACS的偏标签学习
running 
```
CUDA_VISIBLE_DEVICES=0 python -u train97.py 
--num-class 7  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed 
--world-size 1 --rank 0 --seed 123  --arch resnet18 --moco_queue 8192 --prot_start 1 
--lr 0.01 --wd 1e-3 --cosine --epochs 800  --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.1
```
## 损失采用偏标签损失与域标签损失的结合
