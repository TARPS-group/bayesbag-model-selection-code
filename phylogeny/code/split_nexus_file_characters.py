import os
import sys
import shutil
import argparse
import re

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nexfile')
    parser.add_argument('-k', type=int, default=2, help='number of splits')
    parser.add_argument('--split-on-codon', action='store_true')
    parser.add_argument('--equal-size', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    nexfile = args.nexfile
    nexfile_base = os.path.splitext(nexfile)[0]
    split_nexfiles = ['{}-split-{}-of-{}.nex'.format(nexfile_base, i+1,args.k)
                            for i in range(args.k)]

    with open(nexfile, 'r') as f:
        try:
            split_nex_fs = []
            for fname in split_nexfiles:
                split_nex_fs.append(open(fname, 'w'))
            in_matrix = False
            indices = []
            nchar = None
            for line in f:
                writeline = True
                if in_matrix and line.startswith(';'):
                    in_matrix = False
                elif in_matrix:
                    writeline = False
                    species, data = line.strip().split(maxsplit=1)
                    data = re.sub('\s', '', data)
                    if len(data) != nchar:
                        sys.exit('unexpected number of characters for species '
                                 + species)
                    for i, sf in enumerate(split_nex_fs):
                        sf.write(species + '\t')
                        sf.write(data[indices[i]:indices[i+1]])
                        sf.write('\n')
                elif 'DIMENSIONS' in line:
                    writeline = False
                    m = re.search(r'NCHAR=(?P<nchar>\d+)', line)
                    nchar = int(m.group('nchar'))
                    nchars_per_split = nchar // args.k
                    if args.split_on_codon:
                        nchars_per_split -= (nchars_per_split % 3)
                    split_nchars = [nchars_per_split]*args.k
                    if not args.split_on_codon and not args.equal_size:
                        for i in range(args.k):
                            if sum(split_nchars) == nchar:
                                break
                            split_nchars[i] += 1
                        assert sum(split_nchars) == nchar
                    else:
                        print('discarding', nchar - sum(split_nchars),
                              'characters')
                    print(nchar, split_nchars)
                    indices = [0] + np.cumsum(split_nchars).tolist()
                    replace_str = 'NCHAR=' + m.group('nchar')
                    for snchar, sf in zip(split_nchars, split_nex_fs):
                        sline = re.sub(replace_str,
                                       'NCHAR={}'.format(snchar),
                                       line)
                        sf.write(sline)
                elif line.startswith('MATRIX'):
                    if nchar is None:
                        sys.exit("didn't find data info in nexus file")
                    in_matrix = True
                if writeline:
                    for sf in split_nex_fs:
                        sf.write(line)
        finally:
            for sf in split_nex_fs:
                sf.close()


if __name__ == '__main__':
    main()
