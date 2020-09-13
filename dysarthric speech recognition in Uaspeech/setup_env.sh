    # module load apps/python/conda
    # source activate myenv
    # module load libs/CUDA/9.0.176/binary
    # module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
    # module load dev/gcc/5.4
    # module load libs/icu/58.2/gcc-4.9.4
    # PATH=$PATH:/usr/local/packages/libs/icu/58.2/gcc-4.9.4/bin
    # module load libs/intel-mkl/2019.3/binary
    # module load libs/libsndfile/1.0.28/gcc-4.9.4
if [[ "$SGE_CLUSTER_NAME" == "sharc" ]]; then
  module load apps/python/conda	    
  module load libs/CUDA/9.0.176/binary		    
  module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
  module load dev/gcc/5.4
  module load libs/icu/58.2/gcc-4.9.4
  PATH=$PATH:/usr/local/packages/libs/icu/58.2/gcc-4.9.4/bin
  module load libs/intel-mkl/2019.3/binary

  # Add SOX PATH
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/act19yz/tools/sox/lib
  PATH=$PATH:/home/act19yz/tools/sox/bin
  source activate success
  conda install pandas
  conda install xlrd
fi

