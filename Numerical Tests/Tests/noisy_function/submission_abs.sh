#!/bin/sh
#$ -N output_abs
#$ -cwd
#$ -l h_rt=03:30:00
#$ -l h_vmem=1G

. /etc/profile.d/modules.sh

module load anaconda
source activate py310

python abs_main.py 0.1
python abs_main.py 0.01
