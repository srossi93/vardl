#!/usr/bin/env bash

dataset=credit


make -j DATASET=$dataset FOLD=0
make -j DATASET=$dataset FOLD=1
make -j DATASET=$dataset FOLD=2
make -j DATASET=$dataset FOLD=3
make -j DATASET=$dataset FOLD=4

#mv work work-$dataset
