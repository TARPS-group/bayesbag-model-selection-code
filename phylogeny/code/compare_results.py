import os
import sys
import shutil
import subprocess
import argparse
import json
import itertools
import re
from collections import defaultdict
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.stats import hmean
from nexus import NexusReader, NexusWriter

from config import CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='+',
                        help='output directories of results to compare')
    parser.add_argument('--results', default='results',
                        help='results directory to save comparison')
    parser.add_argument('--different-nex', action='store_true')
    parser.add_argument('--compare-to-first', action='store_true')
    parser.add_argument('-B', type=int, default=0,
                        help='if positive, enforce # of bootstrap samples')
    parser.add_argument('--bootstrap', type=int, default=1,
                        help='bootstrap to get confidence intervals on overlaps')
    return parser.parse_args()


def get_name(config, different_nex):
    if different_nex:
        s = config['baseoutfile'] + '-'
    else:
        s = ''
    s += config['model']
    if 'B' in config:
        s += '-c-{c}-a-{a}'.format(**config)
    return s


def get_base_filepath(odir, config, B):
    filename = config['baseoutfile']
    if 'B' in config:
        if B > 0:
            config['B'] = B
        filename += '.{}bootstraps'.format(config['B'])
    return os.path.join(odir, filename)


def get_base_bootstrap_filepaths(odir, config, B):
    filename = config['baseoutfile']
    if B > 0:
        config['B'] = B
    filename += '.bootstrap{}'
    return [os.path.join(odir, filename.format(b+1)) for b in range(config['B'])]


def load_bipartitions(base_filename):
    with open(base_filename+'.parts', 'r') as f:
        # discard name line
        f.readline()
        id_dict = dict()
        for line in f:
            id_str, bipart_str = line.split()
            id_dict[id_str] = bipart_str
    bipart_probs = dict()
    with open(base_filename+'.tstat', 'r') as f:
        # discard ID and column name lines
        f.readline()
        f.readline()
        for line in f:
            id_str, _, prob_str, _ = line.split(maxsplit=3)
            bipart_probs[id_dict[id_str]] = float(prob_str)
    return bipart_probs


def min_element(t):
    if isinstance(t, tuple):
        return np.min([min_element(st) for st in t])
    else:
        return t


def sort_tree(t):
    if isinstance(t, tuple):
        mins = [min_element(st) for st in t]
        sorted_inds = np.argsort(mins)
        return tuple(sort_tree(t[i]) for i in sorted_inds)
    else:
        return t


def load_tree_probs(base_filename, hdp=0.95):
    tree_probs = dict()
    total_prob = 0.0
    line_re = re.compile(r'.* \[&W (?P<prob>.*?)\] (?P<tree>\(.*\));')
    # tree_re = re.compile('^[()\d]*$')

    with open(base_filename+'.trprobs', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('tree'):
                m = line_re.search(line)
                prob = float(m.group('prob'))
                tree_str = m.group('tree')
                tree = str(sort_tree(literal_eval(tree_str)))
                if total_prob + prob > hdp:
                    tree_probs[tree] = hdp - total_prob
                    break
                else:
                    tree_probs[tree] = prob
                    total_prob += prob
    return tree_probs


def disagreement(x, y, eps=1e-3):
    return (x-y) * np.log((x+eps)/(y+eps))


def analyze_bipartitions(biparts):
    df = pd.DataFrame.from_dict(dict(biparts))
    df = df.fillna(0.0)

    max_possible_disag = disagreement(0,1)
    disagreements = []
    for n1, n2 in itertools.combinations(df, 2):
        disag = disagreement(df[n1].values, df[n2].values) / max_possible_disag
        top = np.sort(disag)[-1:-7:-1]
        res = np.mean(top)
        disagreements.append((n1,n2,res,top))
    disagreements.sort(key=lambda t: t[2], reverse=True)
    print('***significant disagreement if > {:.3f}***'.format(1/max_possible_disag))
    print('Disagreement:')
    for n1, n2, res, top in disagreements:
        print('{:22} vs. {:22} {:.3f}   {}'.format(
            n1, n2, res, '(' + ', '.join(['{:.3f}'.format(t) for t in top])) + ')')



def analyze_tree_probs(tree_probs, start_ind, compare_to_first):
    df = pd.DataFrame.from_dict(dict(tree_probs))
    df = df.fillna(0)
    overlaps = []
    overlap_lists = defaultdict(list)
    if compare_to_first:
        names = list(zip(*tree_probs))[0]
        pairs = zip([names[0]]*(len(names)-1), names[1:])
    else:
        pairs = itertools.combinations(df, 2)
    for n1, n2 in pairs:
        min_probs = df.loc[:,(n1, n2)].min(axis=1)
        overlap = min_probs.sum() + 1e-10
        n_overlap = np.count_nonzero(min_probs.values)
        overlaps.append((n1, n2, overlap, n_overlap))
        overlap_lists[n1].append(overlap)
        overlap_lists[n2].append(overlap)

    print('99% HPD tree region overlap:')
    for n1, n2, overlap, n_overlap in overlaps:
        print('{:22} vs. {:22} {:.3f}   {:d}'.format(n1[start_ind:], n2[start_ind:], overlap, n_overlap))


def find_start_ind(args, names):
    start_ind = 0
    if args.different_nex:
        subname = names[0][0]
        while np.all([n.startswith(subname) for n in names]):
            start_ind += 1
            subname = names[0][:start_ind+1]
    return start_ind


def standard_analysis(args, config_info):
    biparts = []
    tree_probs = []
    names = []
    for i, (odir, config) in enumerate(zip(args.output, config_info)):
        name = get_name(config, args.different_nex)
        names.append(name)
        base_filename = get_base_filepath(odir, config, args.B)
        tree_probs.append((name, load_tree_probs(base_filename)))

    start_ind = find_start_ind(args, names)
    analyze_tree_probs(tree_probs, start_ind, args.compare_to_first)


def bootstrap_analysis(args, config_info, hdp=.99):
    df = pd.DataFrame()
    names = []
    print('loading data...')
    for odir, config in zip(args.output, config_info):
        name = get_name(config, args.different_nex)
        print(name)
        names.append(name)
        if 'B' in config:
            base_paths = get_base_bootstrap_filepaths(odir, config, args.B)
        else:
            base_paths = [get_base_filepath(odir, config, args.B)]
        for base_path in base_paths:
            tree_probs = load_tree_probs(base_path, hdp=1.0)
            tree_probs['name'] = name
            df = df.append(tree_probs, ignore_index=True)
    df = df.fillna(0)
    # print(df)
    p_overlaps = defaultdict(list)
    n_overlaps = defaultdict(list)
    print('bootstrapping...')
    for i in range(args.bootstrap):
        if args.bootstrap == 1:
            boot_df = df.groupby('name').mean()
        else:
            boot_df = df.groupby('name').apply(lambda g: g.sample(frac=1., replace=True)).drop('name', axis=1).groupby('name').mean()
        for _, s in boot_df.iterrows():
            total_prob = 0
            all_mass = False
            for i in np.argsort(s.values)[::-1]:
                prob = s[i]
                if all_mass:
                    s[i] = 0
                elif total_prob + prob >= hdp:
                    s[i] = hdp - total_prob
                    all_mass = True
                else:
                    total_prob += prob
        if args.compare_to_first:
            pairs = zip([names[0]]*(len(names)-1), names[1:])
        else:
            pairs = itertools.combinations(names, 2)
        for n1, n2 in pairs:
            min_probs = boot_df.loc[(n1, n2),:].min(axis=0)
            p_overlap = min_probs.sum() + 1e-10
            n_overlap = np.count_nonzero(min_probs.values)
            p_overlaps[(n1,n2)].append(p_overlap)
            n_overlaps[(n1,n2)].append(n_overlap)

    start_ind = find_start_ind(args, names)

    for (n1, n2), p_overlap_list in p_overlaps.items():
        p_ci = np.percentile(p_overlap_list, (10, 90))
        median_n = int(np.median(n_overlaps[(n1,n2)]))
        print('{:22} vs. {:22}  ({:.3f}, {:.3f}) ({})'.format(
            n1[start_ind:], n2[start_ind:], p_ci[0], p_ci[1], median_n))


def main():
    args = parse_args()
    if len(args.output) < 2:
        sys.exit('need at least two outputs to compare')
    print(args.output)
    config_info = []
    for odir in args.output:
        with open(os.path.join(odir, 'experiment.config'), 'r') as f:
            config_info.append(json.load(f))
    if not args.different_nex:
        if np.any([c['nexfile'] != config_info[0]['nexfile'] for c in config_info]):
            sys.exit('all outputs must use the same nexus file')

    if args.bootstrap <= 0:
        standard_analysis(args, config_info)
    else:
        bootstrap_analysis(args, config_info)

if __name__ == '__main__':
    main()
