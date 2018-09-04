#!/usr/bin/env bash

## FOLD 0

python3 ./regr-depth-exp.py with fold=0 init_strategy=blm &
PID1=$!
python3 ./regr-depth-exp.py with fold=0 init_strategy=uninformative &
PID2=$!
python3 ./regr-depth-exp.py with fold=0 init_strategy=xavier-normal &
PID3=$!
wait $PID1 $PID2 $PID3

python3 ./regr-depth-exp.py with fold=0 init_strategy=orthogonal &
PID1=$!
python3 ./regr-depth-exp.py with fold=0 init_strategy=lsuv &
PID2=$!
python3 ./regr-depth-exp.py with fold=0 init_strategy=heuristic &
PID3=$!
wait $PID1 $PID2 $PID3

## FOLD 1

python3 ./regr-depth-exp.py with fold=1 init_strategy=blm &
PID1=$!
python3 ./regr-depth-exp.py with fold=1 init_strategy=uninformative &
PID2=$!
python3 ./regr-depth-exp.py with fold=1 init_strategy=xavier-normal &
PID3=$!
wait $PID1 $PID2 $PID3

python3 ./regr-depth-exp.py with fold=1 init_strategy=orthogonal &
PID1=$!
python3 ./regr-depth-exp.py with fold=1 init_strategy=lsuv &
PID2=$!
python3 ./regr-depth-exp.py with fold=1 init_strategy=heuristic &
PID3=$!
wait $PID1 $PID2 $PID3


## FOLD 2

python3 ./regr-depth-exp.py with fold=2 init_strategy=blm &
PID1=$!
python3 ./regr-depth-exp.py with fold=2 init_strategy=uninformative &
PID2=$!
python3 ./regr-depth-exp.py with fold=2 init_strategy=xavier-normal &
PID3=$!
wait $PID1 $PID2 $PID3

python3 ./regr-depth-exp.py with fold=2 init_strategy=orthogonal &
PID1=$!
python3 ./regr-depth-exp.py with fold=2 init_strategy=lsuv &
PID2=$!
python3 ./regr-depth-exp.py with fold=2 init_strategy=heuristic &
PID3=$!
wait $PID1 $PID2 $PID3


## FOLD 3

python3 ./regr-depth-exp.py with fold=3 init_strategy=blm &
PID1=$!
python3 ./regr-depth-exp.py with fold=3 init_strategy=uninformative &
PID2=$!
python3 ./regr-depth-exp.py with fold=3 init_strategy=xavier-normal &
PID3=$!
wait $PID1 $PID2 $PID3

python3 ./regr-depth-exp.py with fold=3 init_strategy=orthogonal &
PID1=$!
python3 ./regr-depth-exp.py with fold=3 init_strategy=lsuv &
PID2=$!
python3 ./regr-depth-exp.py with fold=3 init_strategy=heuristic &
PID3=$!
wait $PID1 $PID2 $PID3



## FOLD 4

python3 ./regr-depth-exp.py with fold=4 init_strategy=blm &
PID1=$!
python3 ./regr-depth-exp.py with fold=4 init_strategy=uninformative &
PID2=$!
python3 ./regr-depth-exp.py with fold=4 init_strategy=xavier-normal &
PID3=$!
wait $PID1 $PID2 $PID3

python3 ./regr-depth-exp.py with fold=4 init_strategy=orthogonal &
PID1=$!
python3 ./regr-depth-exp.py with fold=4 init_strategy=lsuv &
PID2=$!
python3 ./regr-depth-exp.py with fold=4 init_strategy=heuristic &
PID3=$!
wait $PID1 $PID2 $PID3



