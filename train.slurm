#!/bin/bash

#SBATCH --job-name=pre
#SBATCH --output=./running_log/eterna_train_v2_unlimited.out
#SBATCH --mem=80G

./train.sh