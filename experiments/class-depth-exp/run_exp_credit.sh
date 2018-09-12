#!/usr/bin/env bash

set -e
dataset=credit
gpu=0

for i in `seq 0 2`;
do
    make -j4 DATASET=$dataset FOLD=$i GPU=$gpu blm lsuv uninformative heuristic xavier-normal orthogonal
done
#make -j DATASET=$dataset FOLD=0 GPU=$gpu blm lsuv uninformative heuristic xavier-normal orthogonal
#make -j DATASET=$dataset FOLD=1 GPU=$gpu
#make -j DATASET=$dataset FOLD=2 GPU=$gpu
#make -j DATASET=$dataset FOLD=3 GPU=$gpu
#make -j DATASET=$dataset FOLD=4 GPU=$gpu

#mv work work-$dataset
