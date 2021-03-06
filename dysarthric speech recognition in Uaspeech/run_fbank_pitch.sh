#!/bin/bash
#$ -V
#$ -l h_rt=96:00:00
#$ -l rmem=20G
#$ -j y
#$ -o output_log_pitch_augmentation_MFCC.txt
#$ -e error_log.txt
#$ -m bea
#$ -M yzhou132@sheffield.ac.uk
#$ -cwd # current working directory

# the code above is to use in sharc hpc, so if you run the code in local you can delete the prefix with #$ code
# Yanghao Zhou(yzhou132@sheffield.ac.uk) @ 2020, SPandH, University of Sheffield
# please contact Yanghao Zhou

# specially for UASPEECH corpus
# please ensure that you downloaded the latest version of uaspeech dataset

# Begin configuration section.
nj=15  # probably max 13 for ctl and max 15 for dys due to limited speakers
decode_nj=15  # 20
thread_nj=1  # 4
stage=0

#I actually dont want to compare the control and dysarthric speech , I just mix them up 
# trainset="dys" 	# ctl: training with control speech data, or "dys": training with speech from dysarthric speakers
# # End configuration section


. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./setup_env.sh || exit 1
. utils/parse_options.sh || exit 1

# set -e # exit on error


# path to the global dataset
# change to your corpus folder
home_dir=/data/username/kaldi/egs/ADSR
flist_dir=$home_dir/local/flist 	# check local dir for some specific scripts for uaspeech

# path to data, feature and exp
# you also can change the corpus folder name
data_dir=$home_dir/audio


exp_dir=$home_dir/exp_DNN_pitch_aug_mfcc

# parameters

# you can change your training set name or testing set name
trset="train_mix_pitch"
etset="test_mix_pitch"

lang=$data_dir/lang
boost_sil=1.25
scoring_opts="--word-ins-penalty 0.0"
cmvn_opts="--norm-means=false --norm-vars=false"  	# set both false if online mode
numLeavesTri1=1000
numGaussTri1=10000
numLeavesMLLT=1000
numGaussMLLT=10000
numLeavesSAT=1000
numGaussSAT=15000
nndepth=7
rbm_lrate=0.1
rbm_iter=3  	# smaller datasets should have more iterations!
hid_dim=2048	# according to the total pdfs (gmm-info tri3/final.mdl)
learn_rate=0.002
acwt=0.1 	# only affects pruning (scoring is on lattices)

#====================================GMM-HMM================================================
if [ $stage -le 1 ]; then
  
  # generate the data files for training and test, in data/train*,test*
  #  rm -rf mfcc 
   rm -r $data_dir/train_sp
   rm -r $data_dir/train_mix_pitch
   rm -r $data_dir/test_mix_pitch
   mkdir -p $data_dir/train_mix_pitch
   mkdir -p $data_dir/train_sp
   mkdir -p $data_dir/test_mix_pitch
   echo ==============================================
   echo "---Prepaing UAspeech data and directories---"
   echo ==============================================
   #contain the control data and dysarthric speech: SA training 
   # python3 $home_dir/local/prepare_ua_data_mix.py
   python3 $home_dir/local/prepare_ua_data_mix.py
   utils/utt2spk_to_spk2utt.pl $data_dir/train_mix_pitch/utt2spk > $data_dir/train_mix_pitch/spk2utt        
   utils/utt2spk_to_spk2utt.pl $data_dir/test_mix_pitch/utt2spk > $data_dir/test_mix_pitch/spk2utt

fi

if [ $stage -le 2 ]; then
  echo ====================================
  echo "---Stage 2:prepare_uaspeech_lang---"
  echo ====================================

  # generate uni grammer for UASPEECH in data/lang
  if [ ! -f $data_dir/lang/G.fst ]; then
   local/prepare_uaspeech_lang.sh $flist_dir $data_dir || exit 1;
  fi
fi


# ================================================================================
# feature calculation
if [ $stage -le 3 ]; then
  # mfcc
	echo
	echo "---Stage 3 feature calculation like mfcc and cmvn---"
	echo
	echo
	echo "===== FEATURES EXTRACTION ====="
	echo
	# on the current location of dysarthric speech folder
	# mfccdir=mfcc_mix
	rm -rf $home_dir/mfcc_pitch
	feadir=mfcc_pitch 
	
	for x in train_mix_pitch test_mix_pitch ; do
		utils/validate_data_dir.sh $data_dir/$x
		utils/fix_data_dir.sh $data_dir/$x          
	done
	# change the speech from 1.0 to 1.1
	
	local/perturb_data_dir_speed_1way.sh $data_dir/$trset $data_dir/train_sp
	utils/fix_data_dir.sh $data_dir/train_sp
	utils/validate_data_dir.sh --no-feats $data_dir/train_sp
	
	echo ===========================================================
	echo "this removes the augmentation applied to healthy speakers"
	echo ===========================================================
	for x in wav.scp text utt2spk reco2dur spk2utt utt2dur utt2uniq ; do
		sed -i '/sp1.1-C/d' $data_dir/train_sp/$x
	done
	
	# transforming the feature of train/test/dev sets
	for x in train_mix_pitch test_mix_pitch; do
      
		utils/fix_data_dir.sh $data_dir/$x
		utils/validate_data_dir.sh --no-feats $data_dir/$x
		# steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data_dir/$x $exp_dir/make_mfcc/$x $mfccdir
		steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" $data_dir/$x $exp_dir/make_mfcc_pitch/$x $feadir

		# # Making cmvn.scp files on the MFCCs
		# steps/compute_cmvn_stats.sh $data_dir/$x $exp_dir/make_mfcc/$x $mfccdir
	
		# Making cmvn.scp files on the Fbank_Pitch
		steps/compute_cmvn_stats.sh $data_dir/$x $exp_dir/make_mfcc_pitch/$x $feadir

  done
fi
echo "changing the trset "

# ================================================================================
# GMM-HMM training
if [ $stage -le 4 ]; then
    echo =========================================
    echo "---GMM-HMM training on training data---"
    echo =========================================

    if [ ! -f $exp_dir/$trset/tri2/final.mdl ]; then
    # Starting basic training on fbank+pitch features for dysarthric data based on GMM/HMM
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
		$data_dir/$trset $lang $exp_dir/$trset/mono
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
		$data_dir/$trset $lang $exp_dir/$trset/mono $exp_dir/$trset/mono_ali

    #delta+delta-delta triphone training: Tri1
    steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
		$numLeavesTri1 $numGaussTri1 $data_dir/$trset $lang $exp_dir/$trset/mono_ali $exp_dir/$trset/tri1
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
		$data_dir/$trset $lang $exp_dir/$trset/tri1 $exp_dir/$trset/tri1_ali

    #LDA+MLLT training Tri2
    steps/train_lda_mllt.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
		$numLeavesMLLT $numGaussMLLT $data_dir/$trset $lang $exp_dir/$trset/tri1_ali $exp_dir/$trset/tri2
    fi

    if [ ! -f $exp_dir/$trset/tri3/final.mdl ]; then 
    # SAT training Tri3
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
		$data_dir/$trset $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2_ali

    steps/train_sat.sh --cmd "$train_cmd" --boost-silence $boost_sil \
		$numLeavesSAT $numGaussSAT $data_dir/$trset $lang $exp_dir/$trset/tri2_ali $exp_dir/$trset/tri3

    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
		$data_dir/$trset $lang $exp_dir/$trset/tri3 $exp_dir/$trset/tri3_ali

    #usage: steps/align_fmllr.sh <data-dir> <lang-dir> <src-dir> <align-dir>
    #e.g.:  steps/align_fmllr.sh data/train data/lang exp/tri1 exp/tri1_ali
    fi
fi

if [ $stage -le 5 ]; then
    echo
    echo "---Decoding---"
    echo
    # decode 
    #decoding monophone
    utils/mkgraph.sh $lang $exp_dir/$trset/mono $exp_dir/$trset/mono/graph
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
		$exp_dir/$trset/mono/graph $data_dir/$etset $exp_dir/$trset/mono/decode_test

    #decoding Tri1
    utils/mkgraph.sh $lang $exp_dir/$trset/tri1 $exp_dir/$trset/tri1/graph
    steps/decode.sh --config conf/decode.config --nj $decode_nj --cmd "$decode_cmd" --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
		$exp_dir/$trset/tri1/graph $data_dir/$etset $exp_dir/$trset/tri1/decode_test

    #decoding Tri2
    utils/mkgraph.sh $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2/graph
    if [ ! -f $exp_dir/$trset/tri2/decode_$etset/scoring_kaldi/best_wer ]; then
        steps/decode.sh --config conf/decode.config --nj $decode_nj --cmd "$decode_cmd" --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
		    $exp_dir/$trset/tri2/graph $data_dir/$etset $exp_dir/$trset/tri2/decode_test
    fi

    # decode + SAT
    utils/mkgraph.sh $lang $exp_dir/$trset/tri3 $exp_dir/$trset/tri3/graph
    if [ ! -f $exp_dir/$trset/tri3/decode_$etset/scoring_kaldi/best_wer ]; then
        steps/decode_fmllr.sh --config conf/decode.config --nj $decode_nj --cmd "$decode_cmd" --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
			$exp_dir/$trset/tri3/graph $data_dir/$etset $exp_dir/$trset/tri3/decode_test
    fi
    #print results:
    grep WER $exp_dir/$trset/tri*/decode_*/scoring_kaldi/best_wer
fi


# ================================================================================
# DNN training with mfcc-lda-mllt-fmllr features
gmmdir=$exp_dir/$trset/tri3
dnndir=$exp_dir/$trset/dnn
data_fmllr=$dnndir/fmllr-tri3

if [ $stage -le 6 ]; then
# dump fmllr features for regular DNN training with Kaldi nnet, do not need cmvn!
echo
echo " dump fmllr features for regular DNN training with Kaldi nnet, do not need cmvn! "
echo
for dset in $etset; do
  steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" --transform-dir $gmmdir/decode_test \
    $data_fmllr/$dset $data_dir/$dset $gmmdir $data_fmllr/$dset/log $data_fmllr/$dset/data || exit 1
done

# for training data
if [ ! -f $${gmmdir}_ali/ali.1.gz ]; then
 steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/$trset $lang ${gmmdir} ${gmmdir}_ali || exit 1;
fi

steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" --transform-dir ${gmmdir}_ali \
    $data_fmllr/$trset $data_dir/$trset $gmmdir $data_fmllr/$trset/log $data_fmllr/$trset/data || exit 1
# split the data : 90% train 10% cross-validation (held-out)
utils/subset_data_dir_tr_cv.sh $data_fmllr/${trset} $data_fmllr/${trset}_tr90 $data_fmllr/${trset}_cv10 || exit 1 
fi


if [ $stage -le 7 ]; then
# Pre-train DBN, i.e. a stack of RBMs
echo 
echo "Pre-train DBN based on a stack of RBMs"
echo
dir=$dnndir/pretrain
if [ ! -f $dir/${nndepth}.dbn ]; then
 (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
 $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth $nndepth --hid-dim $hid_dim --rbm-lrate $rbm_lrate --rbm-iter $rbm_iter \
				$data_fmllr/${trset} $dir || exit 1;
fi

# Train the DNN optimizing per-frame cross-entropy.
if [ $stage -le 8 ]; then
echo "Start training DNN "
dir=$dnndir
ali=${gmmdir}_ali
feature_transform=$dir/pretrain/final.feature_transform
dbn=$dir/pretrain/${nndepth}.dbn
(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
# Train
#--network-type (dnn,cnn1d,cnn2d,lstm)  # type of neural network"

$cuda_cmd $dir/log/train_nnet.log \
  steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate $learn_rate $data_fmllr/${trset}_tr90 $data_fmllr/${trset}_cv10 $lang $ali $ali $dir || exit 1;
fi
echo "# Train the DNN optimizing per-frame cross-entropy."
fi
if [ $stage -le 9 ]; then
# Decode (reuse HCLG graph)
for dset in $etset; do
  if [ ! -f $dnndir/decode_test/scoring_kaldi/best_wer ]; then
   steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
	--scoring-opts "$scoring_opts" --stage 0 \
     	$gmmdir/graph $data_fmllr/$dset $dnndir/decode_test || exit 1;
  fi
done
fi
#print results:
grep WER $dnndir/decode_*/scoring_kaldi/best_wer

echo "the above result is the WER of DNN-HMM"






