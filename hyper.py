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
#                                 GLOBALS                                 
#=========================================================================
GIT="/hpc/group/igvf/hyper/git"
RUN_DIR=GIT
addSbatchLines="#SBATCH --exclusive\n"  # +\
    #"#SBATCH --gres=gpu:RTXA5000:1\n" #,gpu:RTX6000:1\n"
MAX_PARALLEL=300
JOB_NAME="BlueSTARR"
MEMORY=20000 #20000 = not enough if RevComp=1



#=========================================================================
#                     ARCHITECTURE / PARAMETER SPACE                      
#=========================================================================
PARM_SPACE={}
PARM_SPACE["AttentionHeads"]=(0,) #(5,10,20,40,60)
PARM_SPACE["AttentionKeyDim"]=(0,) #(5,10,20,30,40) Now used by TransfEncoder
PARM_SPACE["AttentionResidualSkip"]=(0,) #(1,)
PARM_SPACE["BatchSize"]=(128,)
PARM_SPACE["ConvDropout"]=(0,1)
PARM_SPACE["ConvPad"]=("same",)
PARM_SPACE["ConvPoolSize"]=(1,) #(1, 2, 5, 10, 20)
PARM_SPACE["ConvResidualSkip"]=(0,)
PARM_SPACE["DenseSizes"]=(0,) #(20,50,100,200)
PARM_SPACE["DilationFactor"]=(1,) #(1, 2, 3)
PARM_SPACE["DropoutRate"]=(0.5,) #(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4) # 0.3 - 0.9
PARM_SPACE["EarlyStop"]=(10,)
PARM_SPACE["Epochs"]=(200,)
PARM_SPACE["GlobalAvePool"]=(1,)
PARM_SPACE["GlobalMaxPool"]=(0,)
PARM_SPACE["KernelSizes"]=(8,16,32,64,128)
PARM_SPACE["LearningRate"]=(0.002,) #(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01)
PARM_SPACE["MaxTest"]=(999999999,)
PARM_SPACE["MaxTrain"]=(3000000,)
PARM_SPACE["NumAttentionLayers"]=(0,) #(0,1,2)
PARM_SPACE["NumConvLayers"]=(5,10,15,20,25,30) #(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30)# 6-12
PARM_SPACE["NumDense"]=(0,) #(0,1,2)
#PARM_SPACE["NumKernelsFirstLayer"]=(500,1000) #(250, 500, 1000, 2000)
#PARM_SPACE["NumKernelsLaterLayers"]=(25, 50, 100, 200) #(10, 20, 30, 50)
#PARM_SPACE["NumRestarts"]=(1,)
PARM_SPACE["NumKernels"]=(1024,512,256,128,64)
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
    for i in range(NUM_JOBS):
        jobNum=i+1
        nextJob(slurm,SLURM_DIR,jobNum,DATA_DIR,MODEL_DIR)
    slurm.mem(MEMORY)
    slurm.setQueue(QUEUE)
    slurm.writeArrayScript(SLURM_DIR,JOB_NAME,MAX_PARALLEL,addSbatchLines)



#=========================================================================
#                                FUNCTIONS
#=========================================================================
def nextJob(slurm,subdir,jobNum,dataDir,modelDir):
    configFile=subdir+"/"+str(jobNum)+".config"
    modelFile=modelDir+"/model"+str(jobNum)
    writeConfig(configFile)
    cmd="cd "+RUN_DIR+"\n"+\
        "source ~/.bashrc\n"+\
        "conda activate /hpc/home/bmajoros/lab/conda/TF4\n"+\
        "hostname\n"+\
        "echo $SLURMD_NODENAME\n"+\
        "nvidia-smi\n"+\
        GIT+"/BlueSTARR-Transformer.py "+configFile+" "+dataDir+" "+modelFile+"\n"
    slurm.addCommand(cmd)

def sample(values):
    n=len(values)
    i=random.randint(0,n-1)
    value=values[i]
    return value

def writeConfig(filename):
    OUT=open(filename,"wt")
    keys=PARM_SPACE.keys()
    assigned={}
    for key in keys:
        value=sample(PARM_SPACE[key])
        assigned[key]=value

    # Generate list parms
    numConv=assigned["NumConvLayers"]
    numAttn=assigned["NumAttentionLayers"]
    numDense=assigned["NumDense"]
    kernelSizes=[]; numKernels=[]; attnHeads=[]; attnKeyDim=[]; denseSizes=[]
    for i in range(numConv):
        kernelSizes.append(sample(PARM_SPACE["KernelSizes"]))
        if(i==0):
            numKernels.append(sample(PARM_SPACE["NumKernelsFirstLayer"]))
        else:
            numKernels.append(sample(PARM_SPACE["NumKernelsLaterLayers"]))
    for i in range(numAttn):
        attnHeads.append(sample(PARM_SPACE["AttentionHeads"]))
        attnKeyDim.append(sample(PARM_SPACE["AttentionKeyDim"]))
    for i in range(numDense):
        denseSizes.append(sample(PARM_SPACE["DenseSizes"]))
    if(numConv==0):
        kernelSizes=[0]; numKernels=[0]
    if(numAttn==0):
        attnHeads=[0]; attnKeyDim=[0]
    if(numDense==0):
        denseSizes=[0]
    assigned["KernelSizes"]=",".join([str(x) for x in kernelSizes])
    assigned["NumKernels"]=",".join([str(x) for x in numKernels])
    assigned["AttentionHeads"]=",".join([str(x) for x in attnHeads])
    assigned["AttentionKeyDim"]=",".join([str(x) for x in attnKeyDim])
    assigned["DenseSizes"]=",".join([str(x) for x in denseSizes])

    for key in assigned:
        if(key=="NumKernelsFirstLayer" or key=="NumKernelsLaterLayers"):
            continue
        print(key+" = "+str(assigned[key]),file=OUT)
    
    # Global pool: max or ave or neither
    #numDense=assigned["NumDense"]
    #globalPoolType=random.randint(0,2 if numDense>0 else 1)
    #if(globalPoolType==0):
    #    print("GlobalMaxPool = 1",file=OUT)
    #    print("GlobalAvePool = 0",file=OUT)
    #elif(globalPoolType==1):
    #    print("GlobalMaxPool = 0",file=OUT)
    #    print("GlobalAvePool = 1",file=OUT)
    #else:
    #    print("GlobalMaxPool = 0",file=OUT)
    #    print("GlobalAvePool = 0",file=OUT)
    OUT.close()
    
#=========================================================================
#                              command line
#=========================================================================
if(len(sys.argv)!=6):
    exit(ProgramName.get()+" <slurm-dir> <data-dir> <model-dir> <queue> <num-jobs>\n")
(SLURM_DIR,DATA_DIR,MODEL_DIR,QUEUE,NUM_JOBS)=sys.argv[1:]
NUM_JOBS=int(NUM_JOBS)
main(SLURM_DIR,QUEUE,NUM_JOBS,DATA_DIR,MODEL_DIR)





