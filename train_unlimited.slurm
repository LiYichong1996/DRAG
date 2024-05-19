#!/bin/bash

#SBATCH --job-name=pre
#SBATCH --output=./running_log/eterna_train_v2_unlimited.out
#SBATCH --mem=30G

./train_unlimited.sh