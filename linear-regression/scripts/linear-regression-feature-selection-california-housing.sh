#!/bin/bash -l

ALIST=('1' '0.95' '0.75')

echo "california-housing"
for a in ${ALIST[@]}; do
    echo "a = $a"
    set -x
    python scripts/linear_regression_feature_selection.py california-housing -k 1 -B 100 -a $a --a0 2
    python scripts/linear_regression_feature_selection.py california-housing -k 5 -B 100 -a $a --a0 2 --include-full-results
    set +x
done
echo "a = ${ALIST[@]}"
set -x
python scripts/linear_regression_feature_selection.py california-housing -k 5 -B 100 -a ${ALIST[@]} --a0 2 --include-full-results
set +x

