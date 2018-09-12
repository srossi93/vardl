#!/usr/bin/env bash

set -e
dataset=mnist

for i in `seq 0 2`;
do
    make -j4 DATASET=$dataset FOLD=$i GPU=$gpu blm lsuv uninformative heuristic xavier-normal orthogonal
done

#make -j DATASET=$dataset FOLD=0 GPU=2
#make -j DATASET=$dataset FOLD=1 GPU=2
#make -j DATASET=$dataset FOLD=2 GPU=2
#make -j DATASET=$dataset FOLD=3 GPU=2
#make -j DATASET=$dataset FOLD=4 GPU=2

#mv work work-$dataset
