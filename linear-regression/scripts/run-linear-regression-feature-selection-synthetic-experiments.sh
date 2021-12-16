#!/usr/bin/bash

s=$1
scale=$2

echo "1-sparse"
for N in 50 500 5000 50000 50000; do
  for reg in linear nonlinear; do
    for a in .55 .75 .95 1; do 
	   echo "N = $N, reg = $reg, a = $a"
	   qsub scripts/linear-regression-feature-selection-synthetic-data.sh 1 $reg 10 $N $a $s $scale
    done
  done
done

echo "2-sparse"
for N in 100 1000 10000 100000; do
  for reg in linear nonlinear; do
    for a in .75 .95 1; do 
      echo "N = $N, reg = $reg, a = $a"
	    qsub scripts/linear-regression-feature-selection-synthetic-data.sh 2 $reg 20 $N $a $s $scale
    done
  done
done
