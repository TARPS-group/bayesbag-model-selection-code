#!/bin/bash -l

sparsity=$1
reg=$2
D=$3
N=$4
a=$5
s=$6
scale=$7
other=$8

set -x

python code/linear_regression_feature_selection.py synth-corr$scale-${sparsity}sparse-$reg-gaussian -D $D -N $N -a $a -r 50 -B 50 -s $s --a0 2 --sigma0 .25 --results /projectnb/bayesij/bayesbag-experiments/updated-results --figures updated-figures $other

set +x

