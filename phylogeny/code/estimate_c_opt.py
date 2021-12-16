import os
import sys
import shutil
import subprocess
import argparse
import json
import itertools
import re
from collections import defaultdict

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
from scipy.special import logit
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',
                        help='output directory of standard bayes results')
    parser.add_argument('--figures',
                        help='figures directory')

    return parser.parse_args()

def transform_param(s):
    if s.name.startswith('pi(') or s.name.startswith('pinvar'):
        t = logit
    else:
        t = np.log
    return s.apply(t)


def load_variances(paths):
    df_list = [pd.read_csv(path, sep='\s+', skiprows=1, index_col=0, dtype=np.float32) for path in paths]
    df_list = [df.iloc[int(.25*df.shape[0]):] for df in df_list]
    df = pd.concat(df_list, ignore_index=True)
    # print(df.head())
    del df['LnL']
    del df['LnPr']
    for param in list(df.columns):
        if param in ['LnL', 'LnPr'] or param.startswith('gtrsubmodel'):
            del df[param]
    df_trans = df.apply(transform_param)
    return df_trans.var(), df_trans


def load_config_info(dir):
    with open(os.path.join(dir, 'experiment.config'), 'r') as f:
        return json.load(f)



def plot_density(base_path, param, b_s, bb_s):
    plt.figure()
    sns.kdeplot(b_s, label='Standard')
    sns.kdeplot(bb_s, label='BayesBag')
    sns.despine()
    plt.legend()
    plt.savefig(base_path+'-'+param+'.pdf', bbox_inches='tight')
    plt.close()


def create_figures(base_path, b_df, bb_df):
    for param in b_df:
        param_clean = re.sub(r'[{}<>]', '-', param)
        plot_density(base_path, param_clean, b_df[param], bb_df[param])


def main():
    args = parse_args()

    # Load Bayes
    b_config = load_config_info(args.output)
    basename = b_config['baseoutfile']
    b_stat_file_template = os.path.join(args.output, basename+'.run{}.p')
    b_stat_files = [b_stat_file_template.format(i+1) for i in range(2)]
    b_vars, b_df = load_variances(b_stat_files)

    # Load BayesBag
    bb_output = args.output + '-c-1.0-a-1.0'
    bb_config = load_config_info(bb_output)
    B = bb_config['B']
    bb_stat_file_template = os.path.join(bb_output,
                                         basename+'.bootstrap{}.run{}.p')
    bb_stat_files = [bb_stat_file_template.format(b+1, i+1)
                        for b in range(B) for i in range(2)]
    bb_vars, bb_df = load_variances(bb_stat_files)

    if args.figures:
        sns.set_style('white')
        sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2})
        os.makedirs(args.figures, exist_ok=True)
        base_path = os.path.join(args.figures, b_config['baseoutfile']+'-'+b_config['model'])
        create_figures(base_path, b_df, bb_df)

    c_opts = bb_vars / (bb_vars - b_vars)
    print(os.path.basename(args.output))
    if np.any(c_opts < 0):
        print('BayesBag variance sometimes smaller than standard Bayes variances. Unable to estimate optimal c value')
        print(b_vars)
        print(bb_vars)
        print(c_opts)
        sys.exit()
    else:
        c_opt = c_opts.min()
        print('c_opt = {:.3f}'.format(c_opt))
        print('model mismatch = {:.3f}'.format(2/c_opt - 1))


if __name__ == '__main__':
    main()
