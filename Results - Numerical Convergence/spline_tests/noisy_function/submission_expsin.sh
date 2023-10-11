#!/bin/sh
#$ -N outn_expsin
#$ -cwd
#$ -l h_rt=14:00:00
#$ -l h_vmem=4G

. /etc/profile.d/modules.sh

module load anaconda
source activate py310

python main.py exp_sin
