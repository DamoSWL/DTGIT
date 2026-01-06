#!/bin/bash


model_path=Meta-Llama-3-8B-Instruct
data_path=
graph_data_path=${data_path}
pretra_gnn=temporal
dataset=Amazon_movies

# wandb offline
for neighbor in  48
    output_model=${dataset}_checkpoint_es/${dataset}_checkpoint_${neighbor}
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=20000 \
        graphgpt/train/train_mem.py \
        --model_name_or_path ${model_path} \
        --version llama3 \
        --negative_sample entity \
        --data_path ${data_path} \
        --use_dataset ${dataset} \
        --sample_neighbor_size ${neighbor} \
        --seed 42 \
        --graph_content ./arxiv_ti_ab.json \
        --graph_data_path ${graph_data_path} \
        --graph_tower ${pretra_gnn} \
        --tune_graph_mlp_adapter True \
        --graph_select_layer -2 \
        --use_graph_start_end False \
        --bf16 True \
        --output_dir ${output_model} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 6 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "no" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --lazy_preprocess True \
        --report_to none

done



