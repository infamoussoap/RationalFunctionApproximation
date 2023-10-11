import os

os.system(r'for file in *.sh; do qsub $file; done')
