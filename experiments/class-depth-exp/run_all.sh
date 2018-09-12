#!/usr/bin/env bash

set -e

#./run_exp_mnist.sh
#mv work work-mnist

./run_exp_eeg.sh &
PID1=$!
#mv work work-eeg

./run_exp_spam.sh &
PID2=$!
#mv work work-credit

wait $PID2 $PID1

./run_exp_credit.sh &
PID3=$!
#mv work work-spam

