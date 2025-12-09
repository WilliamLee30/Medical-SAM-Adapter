#!/bin/bash

exp_name="test_lucchi"
sam_ckpt_path="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/model/sam_vit_b_01ec64.pth"
data_path="/mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/data/Lucchi++_crop512"

cd /mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/code/Medical-SAM-Adapter

python -u train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name $exp_name \
    -sam_ckpt $sam_ckpt_path \
    -image_size 512 \
    -b 1 \
    -dataset Lucchi \
    -vis 1 \
    -gpu_device 3\
    -data_path $data_path

# nohup bash /mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/code/Medical-SAM-Adapter/scripts/train_2D_MILab.sh > /mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/experiment_results/log/segmentation/MedSA_test_lucchi.log 2>&1 &