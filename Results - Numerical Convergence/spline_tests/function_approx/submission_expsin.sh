#!/bin/sh
#$ -N out_expsin
#$ -cwd
#$ -l h_rt=16:00:00
#$ -l h_vmem=4G

. /etc/profile.d/modules.sh

module load anaconda
source activate py310

python main.py exp_sin
