#!/bin/bash

dataset=$1
sweep_name=$2
sweep_id=$3
sweep_count=$4

CUDA_VISIBLE_DEVICES=0 python main.py -mode sweep -config configs/"$dataset"/sweep.yaml -sweep_name "$sweep_name"  -sweep_id "$sweep_id" -sweep_count "$sweep_count" &
CUDA_VISIBLE_DEVICES=0 python main.py -mode sweep -config configs/"$dataset"/sweep.yaml -sweep_name "$sweep_name"  -sweep_id "$sweep_id" -sweep_count "$sweep_count" &

for i in 1 2 3 4 5 6 7
do
  CUDA_VISIBLE_DEVICES=$i python main.py -mode sweep -config configs/"$dataset"/sweep.yaml -sweep_name "$sweep_name"  -sweep_id "$sweep_id"  -sweep_count "$sweep_count" &
  CUDA_VISIBLE_DEVICES=$i python main.py -mode sweep -config configs/"$dataset"/sweep.yaml -sweep_name "$sweep_name"  -sweep_id "$sweep_id"  -sweep_count "$sweep_count" &
   #CUDA_VISIBLE_DEVICES=$i wandb agent "$sweep_id"&
done

#CUDA_VISIBLE_DEVICES=7 python main.py -mode sweep -config tmp/"$dataset"/"$sweep_name".yaml -sweep_name "$sweep_name"  -sweep_id "$sweep_id" -data_round "$data_round" -sweep_count "$sweep_count" -num_iterations "$num_iterations" -evaluation_method "$evaluation_method" -ratio_easy_data "$ratio_easy_data"
wait