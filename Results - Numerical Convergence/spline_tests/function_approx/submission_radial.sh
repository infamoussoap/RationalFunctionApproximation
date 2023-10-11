#!/bin/sh
#$ -N out_radial
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l h_vmem=4G

. /etc/profile.d/modules.sh

module load anaconda
source activate py310

python main.py radial
