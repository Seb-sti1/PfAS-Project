module load python3/3.10.14
export PATH=$PATH:~/.local/bin
export WORKON_HOME=/work3/pnha/student_scratch/Envs
mkdir -p $WORKON_HOME
source virtualenvwrapper.sh
workon llm_trajectory_prediction
LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64