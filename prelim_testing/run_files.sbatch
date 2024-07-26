#!/bin/bash
#
###############################################################################                                                    
#SBATCH --job-name="demo"
#SBATCH --output="demo.%j.%N.out"
#SBATCH --error="demo.%j.%N.err"
#SBATCH --partition=gpux1
#SBATCH --time=4                              
###############################################################################
# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

#cd ${SLURM_SUBMIT_DIR}




#Load Anaconda modulefile 
module load anaconda/2022-May/3
module load cuda

mkdir -p /home/anushaghosh/scratch/mypython3
#pip3 install -t /home/anushaghosh/scratch/mypython3 -r requirements.txt
export PYTHONPATH=/home/anushaghosh/scratch/mypython3:${PYTHONPATH}

# Run/execute the python code/scipt
python main.py --epochs=25 --model=vgg11 --batch_size=32 --attack=pgd 2> cur_out/pgd_python_err.txt 1> cur_out/pgd_python_out.txt
