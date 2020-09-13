# Setting local system jobs (local CPU - no external clusters)
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl

if [[ "$SGE_CLUSTER_NAME" == "sharc" ]]; then
    export train_cmd="queue.pl -V --mem 6G -l h_rt=08:00:00" 
    export decode_gmm="queue.pl -V --mem 6G -l h_rt=08:00:00"
    export decode_cmd="queue.pl -V --mem 6G -l h_rt=08:00:00"
    export mkgraph_cmd="queue.pl -V --mem 6G -l h_rt=08:00:00"
    export cuda_cmd="queue.pl -V --mem 10G -l h_rt=15:00:00 --gpu 1"
    export cmd="queue.pl -V --mem 8G -l h_rt=08:00:00"
fi

