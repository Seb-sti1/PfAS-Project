CUDA_VISIBLE_DEVICES=1 python exercise_5_HPC.py #run code on gpu
 #interactive node for the class
nvidia-smi #check available gpus
cd /dtu/datasets1/02514/ #course datasets
getquota_zhome.sh #check storage space
bsub -app c02516_1g.10gb < jobscript.sh   #run bash script in this course, memory limited to 10 GB, not much more is needed
bsub -app c02516_2g.20gb < jobscript.sh
bsub -app c02516_4g.40gb < jobscript.sh
#BSUB -R "select[gpu32gb]" #request 32 GB gpu only on gpuv100
bsub -q gpua100      -J "deep learning in computer vision training"      -n 4      -gpu "num=1"      -W 06:00      -R "rusage[mem=16GB]"      -B      -N      -o gpu_%J.out      -e gpu_%J.err6< run.sh #run command for different queue
a100sh #GPU interactive node
sxm2sh #GPU interactive node
voltash #GPU interactive node


Jupyter setup:
remote: 
export PATH=$PATH:~/.local/bin
jupyter notebook --no-browser --port=8888 --ip=$HOSTNAME
local:
ssh USER@login2.hpc.dtu.dk -g -LXXXXX:n-00-00-00:XXXXX –N

Browser:
http://127.0.0.1:XXXXX/?token=xyxyxyxyxyxyxyxyxyxyxyxyyxyxyxyxyxxy

02516sh
source .venv_deep_learning/bin/activate
export PATH=$PATH:~/.local/bin
jupyter notebook --no-browser --port=8888 --ip=$HOSTNAME