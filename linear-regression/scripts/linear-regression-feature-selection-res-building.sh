#!/bin/bash -l

ALIST=('1' '0.95' '0.75')

echo "res-building"
for a in ${ALIST[@]}; do
    echo "a = $a"
    set -x
    python code/linear_regression_feature_selection.py res-building -s 3 -k 1 -B 100 -a $a --a0 2 
    python code/linear_regression_feature_selection.py res-building -s 3 -k 3 -B 100 -a $a --a0 2 --include-full-results
    set +x
done
echo "a = ${ALIST[@]}"
set -x
python code/linear_regression_feature_selection.py res-building -s 3 -k 3 -B 100 -a ${ALIST[@]} --a0 2 --include-full-results
set +x
