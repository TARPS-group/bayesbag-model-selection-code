#!/usr/bin/bash


echo "1-sparse"
for N in 50 500 5000 50000; do 
  for reg in linear nonlinear; do
      echo "N = $N, reg = $reg"
      source scripts/linear-regression-feature-selection-synthetic-data.sh 1 $reg 10 $N "1 .95 .75 .55" 1
      source scripts/linear-regression-feature-selection-synthetic-data.sh 1 $reg 10 $N "1 .95 .75 .55" 2
  done
done

echo "2-sparse"
for N in 100 1000 10000 100000; do
  for reg in linear nonlinear; do
      echo "N = $N, reg = $reg"
      source scripts/linear-regression-feature-selection-synthetic-data.sh 2 $reg 20 $N "1 .95 .75 .55" 2 "" "-e 1 2 3 16 17 18 19 20"
      source scripts/linear-regression-feature-selection-synthetic-data.sh 2 $reg 20 $N "1 .95 .75" 2 4 "-e 1 2 3 16 17 18 19 20"
  done
done
