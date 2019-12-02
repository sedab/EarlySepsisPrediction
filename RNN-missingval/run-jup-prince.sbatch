#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=char_jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=36:00:00

module purge
module load jupyter-kernels/py3.5
module load pillow/python3.5/intel/4.2.1
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
pip install -U --user nilearn
pip3 install -U --user pandas==0.24.1


port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port





