#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH -J batcjob
#SBATCH -o batcjob.%J.out
#SBATCH -e batcjob.%J.err



export CODE_SERVER_CONFIG=~/.config/code-server/config.yaml
export XDG_CONFIG_HOME=$HOME/tmpdir
node=$(/bin/hostname)
port=1717
user=$(niuk0a)  
submit_host=${SLURM_SUBMIT_HOST} 

if [ -f ${CODE_SERVER_CONFIG} ] ; then
 rm ${CODE_SERVER_CONFIG}
fi

echo "bind-addr: ${node}:${port}" >> $CODE_SERVER_CONFIG 
echo "auth: password" >> config
echo "password: testpass" >> $CODE_SERVER_CONFIG
echo "cert: false" >> config

echo "Copy the following line in a new terminal to create a secure SSH tunnel between your computer and Ibex compute node."
echo "ssh -L localhost:${port}:${node}:${port} ${user}@${submit_host}.ibex.kaust.edu.sa"

code-server --auth=password --verbose


# source /home/niuk0a/anaconda3/bin/activate /home/niuk0a/anaconda3/bin/code-server