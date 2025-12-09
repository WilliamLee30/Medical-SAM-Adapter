#!/bin/bash

exp_name="test_lucchi"
sam_ckpt_path="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/model/sam_vit_b_01ec64.pth"
# data_path="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/data/ISIC"
data_path="/mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/data/Lucchi++_crop512"

cd /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/code/Medical-SAM-Adapter

python -u train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name $exp_name \
    -sam_ckpt $sam_ckpt_path \
    -image_size 1024 \
    -b 2 \
    -dataset isic \
    -vis 1 \
    -data_path $data_path

# nohup bash /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/code/Medical-SAM-Adapter/scripts/train_2D.sh > /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/experiment_results/logs/test_isic.log 2>&1 &