#!/usr/bin/env python
#=========================================================================
# This is OPEN SOURCE SOFTWARE governed by the Gnu General Public
# License (GPL) version 3, as described at www.opensource.org.
# Copyright (C)2023 William H. Majoros <bmajoros@alumni.duke.edu>
#=========================================================================
from __future__ import (absolute_import, division, print_function, 
   unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
   chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
# The above imports should allow this program to run in both Python 2 and
# Python 3.  You might need to update your version of module "future".
import sys
import random
import ProgramName
from SlurmWriter import SlurmWriter

#=========================================================================
#                               GLOBALS                                    
#=========================================================================
CONDA_ENV="/hpc/home/bmajoros/lab/conda/TF5"
GIT="/hpc/group/igvf/hyper/git"
RUN_DIR=GIT
addSbatchLines="#SBATCH --exclusive\n"   +\
    "#SBATCH --gres=gpu:RTXA5000:1\n" #,gpu:RTX6000:1\n"
MAX_PARALLEL=300
JOB_NAME="BlueSTARR"
MEMORY=100000


#=========================================================================
#                     ARCHITECTURE / PARAMETER SPACE                      
#=========================================================================
NUM_LAYERS=10
NUM_REPS=10
PARM_SPACE={}
PARM_SPACE["NumKernels"]=(1024,512,256,128,64,32,32,32,32,32)
PARM_SPACE["KernelSizes"]=(8,16,32,64,128,128,128,128,128,128)
PARM_SPACE["AttentionKeyDim"]=(5,10,20,30) # Enformer uses 64
# Everything below here is fixed
PARM_SPACE["AttentionHeads"]=(8,) # Enformer uses 8
PARM_SPACE["ConvDropout"]=(1,)
PARM_SPACE["DropoutRate"]=(0.5,)
PARM_SPACE["DilationFactor"]=(0,)
PARM_SPACE["AttentionResidualSkip"]=(0,)
PARM_SPACE["BatchSize"]=(128,)
PARM_SPACE["ConvPad"]=("same",)
PARM_SPACE["ConvPoolSize"]=(1,)
PARM_SPACE["ConvResidualSkip"]=(1,)
PARM_SPACE["DenseSizes"]=(0,)
PARM_SPACE["EarlyStop"]=(10,)
PARM_SPACE["Epochs"]=(200,)
PARM_SPACE["GlobalAvePool"]=(1,)
PARM_SPACE["GlobalMaxPool"]=(0,)
PARM_SPACE["LearningRate"]=(0.002,)
PARM_SPACE["MaxTest"]=(999999999,)
PARM_SPACE["MaxTrain"]=(3000000,)
PARM_SPACE["NumDense"]=(0,)
PARM_SPACE["RevComp"]=(0,)
PARM_SPACE["ShouldTest"]=(1,)
PARM_SPACE["Tasks"]=("K562",)
PARM_SPACE["TaskWeights"]=(1,)
PARM_SPACE["UseCustomLoss"]=(0,)
PARM_SPACE["Verbose"]=(2,)



#=========================================================================
#                                 main()
#=========================================================================
def main(SLURM_DIR,QUEUE,NUM_JOBS,DATA_DIR,MODEL_DIR):
    slurm=SlurmWriter()
    KEY_DIMS=PARM_SPACE["AttentionKeyDim"]
    jobNum=0
    for i in range(len(KEY_DIMS)):
        for numConv in range(10):
            for rep in range(NUM_REPS):
                parms=[NUM_LAYERS,numConv,KEY_DIMS[i]]
                nextJob(slurm,SLURM_DIR,jobNum,DATA_DIR,MODEL_DIR,parms)
                jobNum+=1
    slurm.mem(MEMORY)
    slurm.setQueue(QUEUE)
    slurm.writeArrayScript(SLURM_DIR,JOB_NAME,MAX_PARALLEL,addSbatchLines)



#=========================================================================
#                                FUNCTIONS
#=========================================================================
def nextJob(slurm,subdir,jobNum,dataDir,modelDir,parms):
    configFile=subdir+"/"+str(jobNum)+".config"
    modelFile=modelDir+"/model"+str(jobNum)
    writeConfig(configFile,parms)
    cmd="cd "+RUN_DIR+"\n"+\
        "source ~/.bashrc\n"+\
        "conda activate /hpc/home/bmajoros/lab/conda/TF5\n"+\
        "hostname\n"+\
        "echo $SLURMD_NODENAME\n"+\
        "nvidia-smi\n"+\
        GIT+"/BlueSTARR-Transformer.py "+configFile+" "+dataDir+" "+\
        modelFile+"\n"
    slurm.addCommand(cmd)

def writeConfig(filename,parms):
    OUT=open(filename,"wt")
    keys=PARM_SPACE.keys()
    assigned={}
    for key in keys:
        assigned[key]=PARM_SPACE[key][0]

    # Generate list parms
    (numLayers,numConv,keyDimension)=parms
    numAttn=numLayers-numConv
    HEADS=PARM_SPACE["AttentionHeads"][0]
    numKernels=PARM_SPACE["NumKernels"][:numConv]
    kernelSizes=PARM_SPACE["KernelSizes"][:numConv]
    attnKeyDim=[keyDimension]*numAttn
    attnHeads=[HEADS]*numAttn
    denseSizes=PARM_SPACE["DenseSizes"]
    numDense=len(denseSizes)
    if(numConv==0):
        kernelSizes=[0]; numKernels=[0]
    if(numAttn==0):
        attnHeads=[0]; attnKeyDim=[0]
    if(numDense==0): denseSizes=[0]
    assigned["KernelSizes"]=",".join([str(x) for x in kernelSizes])
    assigned["NumKernels"]=",".join([str(x) for x in numKernels])
    assigned["AttentionHeads"]=",".join([str(x) for x in attnHeads])
    assigned["AttentionKeyDim"]=",".join([str(x) for x in attnKeyDim])
    assigned["DenseSizes"]=",".join([str(x) for x in denseSizes])
    for key in assigned:
        print(key+" = "+str(assigned[key]),file=OUT)
    OUT.close()
    
#=========================================================================
#                              command line
#=========================================================================
if(len(sys.argv)!=6):
    exit(ProgramName.get()+" <slurm-dir> <data-dir> <model-dir> <queue> <num-jobs>\n")
(SLURM_DIR,DATA_DIR,MODEL_DIR,QUEUE,NUM_JOBS)=sys.argv[1:]
NUM_JOBS=int(NUM_JOBS)
main(SLURM_DIR,QUEUE,NUM_JOBS,DATA_DIR,MODEL_DIR)





