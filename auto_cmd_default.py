#-*-coding:utf-8-*-
cmds = [
    # "python trainval_net.py --save_dir weights_cag --dataset fake_sim10k --net vgg16  --bs 2 --nw 0 --lr 0.001 --cuda --cag",
    "CUDA_VISIBLE_DEVICES=0 python train.py --dataset real_cityscapes --batch_size 8 --lr 1e-4",
    "CUDA_VISIBLE_DEVICES=0 python train.py --dataset fake_sim10k --batch_size 8 --lr 1e-4",
    "CUDA_VISIBLE_DEVICES=0 python train.py --dataset fake_cityscapes --batch_size 8 --lr 1e-4",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
