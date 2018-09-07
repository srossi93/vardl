#!/usr/bin/env bash

set -e

./run_exp_mnist.sh
mv work work-mnist

./run_exp_eeg.sh
mv work work-eeg

./run_exp_credit.sh
mv work work-credit

./run_exp_spam.sh
mv work work-spam

