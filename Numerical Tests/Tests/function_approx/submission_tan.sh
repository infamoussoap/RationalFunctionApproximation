#!/bin/sh
#$ -N output_tan
#$ -cwd
#$ -l h_rt=03:00:00
#$ -l h_vmem=1G

. /etc/profile.d/modules.sh

module load anaconda
source activate py310

python tan_main.py
