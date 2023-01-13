
ESTI: An Action Recognition Network with Enhanced Spatio-Temporal Information
---------------------
By Zhiyu Jiang, Yi Zhang



Introduction
----------------
Action recognition is an active topic in video understanding, which aims to recognize human actions in videos. The critical step is to model the spatio-temporal information and extract key action clues. To this end, we propose a simple and efficient network (dubbed ESTI) which consists of two core modules. The Local Motion Extraction module highlights the short-term temporal context. While the Global Multi-scale Feature Enhancement module strengthens the spatio-temporal and channel features to model long-term information. By appending ESTI to a 2D ResNet backbone, our network is capable of reasoning different kinds of actions with various amplitudes in videos. Our network is developed under two Geforce RTX 3090 using Python3.7/Pytorch1.8. Extensive experiments have been conducted on 5 mainstream datasets to verify the effectiveness of our network, in which ESTI outperforms most of the state-of-the-arts methods in terms of accuracy, computational cost and network scale.

Prepare data
----------
```
  cd /ESTI
  mkdir -p data
  ln -s /path_to_dataset ./data
```

Training
--------------
For example,
```shell
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
                main.py  something RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
                --lr_scheduler step --lr_steps  30 40 45 --epochs 50 --batch-size 32 \
                --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 8 --npb
```



Testing
-----------
```shell
CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
--archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
--test_crops=1 --batch_size=16  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0
```
```shell
python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>  
```

Results
---------
| Dataset                | Frames | Top-1 |
|------------------------|--------|-------|
| Something-Something-V1 | 8      | 50.2  |
| Something-Something-V2 | 8      | [63.7](https://pan.baidu.com/s/11Q0gN5HrESSGrNodBh9EDw?pwd=26zn 提取码：26zn)  |
| Jester                 | 8      | 96.9  |
| Diving48               | 8      | 37.2  |
| UCF101                 | 8      | 87.2  |


License
--------
This project is released under the [Apache 2.0 license](LICENSE)
