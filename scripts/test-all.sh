#!/bin/bash

dataset=$1
seed=$2
gpu=$3

CUDA_VISIBLE_DEVICES="$gpu" python main.py -config configs/"$dataset"/trial"$seed"/encoder-decoder.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python main.py -config configs/"$dataset"/trial"$seed"/encoder-decoder-64.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python main.py -config configs/"$dataset"/trial"$seed"/base.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python main.py -config configs/"$dataset"/trial"$seed"/smooth.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python regress.py -config configs/"$dataset"/trial"$seed"/regress-smooth-filtered.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python regress.py -config configs/"$dataset"/trial"$seed"/regress-base-filtered.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python main.py -config configs/"$dataset"/trial"$seed"/smooth-noAnnealing.yaml -mode test
CUDA_VISIBLE_DEVICES="$gpu" python regress.py -config configs/"$dataset"/trial"$seed"/regress-smooth.yaml -mode test