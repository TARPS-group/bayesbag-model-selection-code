import os
import sys
import shutil
import subprocess
import argparse
from itertools import chain
import json

import numpy as np

from nexus import NexusReader, NexusWriter
from config import CONFIGS

CHECKPOINT_HEADER_TEMPLATE = """#NEXUS
[ID: {}]
[generation: {}]
"""

MB_FILE_CONTENT_TEMPLATE = """begin MrBayes;
   set autoclose=yes nowarn=yes;
   execute {nexfile};
   outgroup {outgroup};
   {modelconfig}
   mcmcp nchains=4 ngen={ngen} samplefreq=1 printfreq=1 file={baseoutfile} append=yes;
   sumt;
   sump;
end;"""


SCC_BATCH_FILE_TEMPLATE = """#!/bin/bash -l
#$ -cwd
#$ -N {jobname}
#$ -pe omp 1
#$ -v OMP_NUM_THREADS=1
#$ -l h_rt=00:30:00
#$ -l mem_per_core=3G
#$ -l avx
module load openmpi/3.1.4
module load mrbayes/3.2.7
mpirun -c 1 mb {mbfile}

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()

def update_t_gen(t, i):
   pieces = t.split(maxsplit=3)
   pieces[1] = 'gen.{}'.format(i)
   return ' '.join(pieces)

def update_p_gen(p, i):
   pieces = p.split(maxsplit=2)
   pieces[0] = str(i)
   return '\t'.join(pieces)


def create_t_file(output, basename, B, run, id_num):
    trees_list = []
    for b in range(B):
        t_filename = '{}.bootstrap{}.run{}.t'.format(basename, b+1, run)
        r = NexusReader(os.path.join(output, t_filename))
        trees_list.append(r.trees.trees)
    lengths = [len(t) for t in trees_list]
    print('t files:', lengths)
    max_len = np.max(lengths)
    print('short files:', np.nonzero(np.array(lengths) < max_len)[0] + 1)
    all_trees = list(chain.from_iterable(zip(*trees_list)))
    w = NexusWriter()
    w.trees = [update_t_gen(t, i) for i, t in enumerate(all_trees)]
    w.add_comment('ID: {}'.format(id_num))
    w.add_comment('Param: tree')
    t_filename = '{}.{}bootstraps.run{}.t'.format(basename, B, run)
    w.write_to_file(os.path.join(output, t_filename))
    return len(all_trees)


def create_p_file(output, basename, B, run, id_num):
    p_lists = []
    for b in range(B):
        p_filename = '{}.bootstrap{}.run{}.p'.format(basename, b+1, run)
        with open(os.path.join(output, p_filename), 'r') as f:
            f.readline() # discard ID comment
            heading = f.readline()
            p_lists.append([line for line in f])
            line_lengths = np.array([len(line) for line in p_lists[-1]])
            short_lines = line_lengths < .5 * np.mean(line_lengths)
            if np.any(short_lines):
                print('run = {}, b = {} short lines:'.format(run, b), np.nonzero(short_lines)[0] + 3)
    all_p = list(chain.from_iterable(zip(*p_lists)))
    p_filename = '{}.{}bootstraps.run{}.p'.format(basename, B, run)
    print('p files:', [len(p) for p in p_lists])
    with open(os.path.join(output, p_filename), 'w') as f:
        f.write('[ID: {}]\n'.format(id_num))
        f.write(heading)
        for i, p in enumerate(all_p):
            f.write(update_p_gen(p, i))
    return len(all_p)


def create_ckp_file(output, basename, B, id_num, ngen):
    ckp_in_filename = '{}.bootstrap{}.ckp'.format(basename, B)
    ckp_out_filename = '{}.{}bootstraps.ckp'.format(basename, B)
    with open(os.path.join(output, ckp_in_filename), 'r') as f_in, \
         open(os.path.join(output, ckp_out_filename), 'w') as f_out:
        # discard header
        f_in.readlines(3)
        # write new header
        f_out.write(CHECKPOINT_HEADER_TEMPLATE.format(id_num, ngen))
        # copy the rest
        for line in f_in:
            f_out.write(line)

def create_mcmc_file(output, basename, B, id_num, ngen):
    final_stats = []
    for b in range(B):
        mcmc_filename = '{}.bootstrap{}.mcmc'.format(basename, b+1)
        header = ''
        reading_header = True
        with open(os.path.join(output, mcmc_filename), 'r') as f:
            # discard ID
            f.readline()
            for line in f:
                if reading_header:
                    header += line
                    if line.startswith('Gen'):
                        reading_header = False
                else:
                    last_line = line
            final_stats.append(last_line.split()[1:])
    all_stats = np.array(final_stats, dtype=float)
    mean_stats = np.mean(all_stats, axis=0)
    mcmc_filename = '{}.{}bootstraps.mcmc'.format(basename, B)
    with open(os.path.join(output, mcmc_filename), 'w') as f:
        f.write('[ID: {}]\n'.format(id_num))
        f.write(header)
        fmt = '{}\t' + '\t'.join(['{:.6f}']*mean_stats.size)
        f.write(fmt.format(ngen, *mean_stats))


def main():
    args = parse_args()
    with open(os.path.join(args.output, 'experiment.config'), 'r') as f:
        config_info = json.load(f)
    basename = config_info['baseoutfile']
    B = config_info['B']
    model = config_info['model']
    nexfile = config_info['nexfile']
    outgroup = config_info['outgroup']

    id_num = np.random.randint(np.iinfo(np.int32).max)

    print('creating t files...')
    t_lens = []
    for run in [1,2]:
        t_lens.append(create_t_file(args.output, basename, B,
                                    run, id_num))
    if len(set(t_lens)) != 1:
        sys.exit('sample lengths for t files are inconsistent!')

    print('creating p files...')
    p_lens = []
    for run in [1,2]:
        p_lens.append(create_p_file(args.output, basename, B,
                                    run, id_num))
    if len(set(p_lens)) != 1:
        sys.exit('sample lengths for p files are inconsistent!')
    if t_lens[0] != p_lens[0]:
        print(t_lens, p_lens)
        sys.exit('sample lengths for p and t files are inconsistent!')

    print('creating ckp and mcmc files...')
    ngen = t_lens[0]
    create_ckp_file(args.output, basename, B, id_num, ngen)
    create_mcmc_file(args.output, basename, B, id_num, ngen)

    print('creating scripts...')
    baseoutfile = basename + '.{}bootstraps'.format(B)
    mb_file_content = MB_FILE_CONTENT_TEMPLATE.format(
        nexfile=nexfile, outgroup=outgroup,
        modelconfig=CONFIGS[model], ngen=ngen, baseoutfile=baseoutfile)
    mb_filename = baseoutfile + '.mb'
    with open(os.path.join(args.output, mb_filename), 'w') as f:
        f.write(mb_file_content)

    scc_batch_file_content = SCC_BATCH_FILE_TEMPLATE.format(
        jobname=baseoutfile, mbfile=mb_filename)
    scc_batch_file = 'run_combine_bootstrap_results.sh'
    with open(os.path.join(args.output, scc_batch_file), 'w') as f:
        f.write(scc_batch_file_content)

    batch_cmd = 'qsub {}'.format(scc_batch_file)
    if args.run:
        print(batch_cmd)
        res = subprocess.run(batch_cmd, cwd=args.output, shell=True, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print('***batch command not run successfully***')
    else:
        print('cd', args.output + ';', batch_cmd + ';', 'cd -')


if __name__ == '__main__':
    main()
