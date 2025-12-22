#!/bin/bash

project_name="mitochondria_segmentation"
exp_name="lucchi_single_point_0logit"
sam_ckpt_path="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/model/sam_vit_b_01ec64.pth"
data_path="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/data/Lucchi++_crop512"


cd /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/code/Medical-SAM-Adapter

python -u train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name $exp_name \
    -sam_ckpt $sam_ckpt_path \
    -image_size 512 \
    -out_size 512 \
    -b 2 \
    -dataset Lucchi \
    -vis 1 \
    -gpu_device 0 \
    -data_path $data_path

# nohup bash /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/code/Medical-SAM-Adapter/scripts/train_2D_Tencent.sh > /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic_video/user_williamvvli/pre_experiment/mitochondria_segmentation/experiment_results/segmentation/lucchi_single_point_0logit.log 2>&1 &