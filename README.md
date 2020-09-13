# Dysarthric-Speech-Recognition
This is my master's dissertation in the University of Sheffield.
Many thanks to my supervisor's Phd and Post-Phd: Gerardo, Feifei, Zhenjun, Bahman. 
And thanks to my supervisor. 

This project is based on Kaldi. So you need to download and make kaldi firstly.

This code can run in sharc.

You definitely should run it on Sharc. If you run in local, you need to fix some problem.

If you meet the issues, you can email me with email germany-tum@qq.com

Because my email in UoS was already deleted.

## The tutorial to make Kaldi in HPC ShARC. 
So you can modify the progress because your hpc is another one.

And sharc is my hpc's name. You can install Kaldi in Sharc but you have to do it in your own environment, and you have to do it in your /data directory.

1. open an interactive session

2. load module and create python env
```
 - module load apps/python/conda
 - conda create -n xxxxx python=3.7
 - source activate xxxxx
 - module load libs/CUDA/9.0.176/binary
 - module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
 - module load dev/gcc/5.4
 - module load libs/icu/58.2/gcc-4.9.4
 - PATH=$PATH:/usr/local/packages/libs/icu/58.2/gcc-4.9.4/bin
 - module load libs/intel-mkl/2019.3/binary 
```
I recommend to add this info：
```
if [ $SGE_CLUSTER_NAME == "sharc" ]; then
module load apps/python/conda
source activate xxxxx
module load libs/CUDA/9.0.176/binary
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
module load dev/gcc/5.4
module load libs/icu/58.2/gcc-4.9.4
PATH=$PATH:/usr/local/packages/libs/icu/58.2/gcc-4.9.4/bin
module load libs/intel-mkl/2019.3/binary 
module load libs/libsndfile/1.0.28/gcc-4.9.4
fi
```
into this file：
```
/home/<user_id>/.bashrc
```

3. You need to install sox 
    
    download sox from https://sourceforge.net/projects/sox/files/sox/
    extract in a directory, e.g. /home/<used_id>/tools/sox
    cd to /home/<used_id>/tools/sox
    ./configure --prefix=/home/<used_id>/tools/sox
    make -s
    make install
    add in .bashrc:
    
    ```
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/tools/sox/lib
    PATH=$PATH:$HOME/tools/sox/bin
    ```
4. Download the Kaldi
```
    cd /data/<user_id>
    git clone https://github.com/kaldi-asr/kaldi.git
```
5. cd kaldi/tools

6. make

7  You will get a note that SRILM language model is not installed by default anymore.
    
    - download srilm package from http://www.speech.sri.com/projects/srilm/download.html.
    - Rename the package as “srilm.tgz”.
    - Copy the package to kaldi /tools
    - in kaldi/tools run ./install_srilm.sh

8. cd /kaldi/src

#### This is the most important progress to let your kaldi system can work on CUDA GPU!!!!

9. 
```
./configure --cudatk-dir=/usr/local/packages/libs/CUDA/9.0.176/binary/cuda --mkl-root=/usr/local/packages/dev/intel-ps-xe-ce/2019.3/binary/compilers_and_libraries_2019.3.199/linux/mkl
```
10. make depend

11 make

    After you finished the installation, you can add the following in the recipe

```
in file run.sh (adjust for your case)
#!/bin/bash
#$ -V
#$ -l h_rt=96:00:00
#$ -l rmem=50G
#$ -j y
#$ -o log.txt
#$ -e log.txt
#$ -m bea
#$ -M groadabike1@sheffield.ac.uk
#$ -N DSing1_a
```

    in cmd.sh
```
export train_cmd="run.pl"
export decode_cmd="run.pl"
if [[ "$SGE_CLUSTER_NAME" == "sharc"]]; then
    export train_cmd="queue.pl -V --mem 6G -l h_rt=08:00:00 "
    export decode_gmm="queue.pl -V --mem 10G -l h_rt=08:00:00"
    export decode_cmd="queue.pl -V --mem 8G -l h_rt=08:00:00"
fi
```

in path.sh

```
export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
```

    in conf/queue.conf

```
command qsub -v PATH -cwd -S /bin/bash -j y -m a -M groadabike1@sheffield.ac.uk
option mem=* -l rmem=$0 -j y
option mem=0          # Do not add anything to qsub_opts
option num_threads=* -pe smp $0
option num_threads=1  # Do not add anything to qsub_opts
option max_jobs_run=* -tc $0
default gpu=0
option gpu=0
option gpu=* -l gpu=$0 -P rse -q rse.q
```

    I have an extra next run.sh, cmd.sh and path.sh file called

    setup_env.sh

```
if [[ "$SGE_CLUSTER_NAME" == "sharc"]]; then
    module load apps/python/conda
    module load libs/CUDA/9.0.176/binary
    module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
    module load dev/gcc/5.4
    module load libs/icu/58.2/gcc-4.9.4
    PATH=$PATH:/usr/local/packages/libs/icu/58.2/gcc-4.9.4/bin
    module load libs/intel-mkl/2019.3/binary

    # Add SOX PATH
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/acp13gr/apps/sox/lib
    PATH=$PATH:/home/acp13gr/apps/sox/bin
    source activate xxxx
fi
```

    xxxx is your Conda virtual environment's name

    and in run.sh i called

```
. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./setup_env.sh
```

Hope it helps.
