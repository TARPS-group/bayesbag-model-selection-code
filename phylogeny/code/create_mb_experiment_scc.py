import os
import sys
import shutil
import subprocess
import argparse
import json

import numpy as np
from nexus import NexusReader, NexusWriter

from config import CONFIGS


MB_FILE_CONTENT_TEMPLATE = """begin MrBayes;
   set autoclose=yes nowarn=yes;
   execute {nexfile};
   outgroup {outgroup};
   {modelconfig}
   mcmc nchains=4 ngen={ngen} samplefreq={samplefreq} printfreq={printfreq} file={baseoutfile}{mcmcargs};
   sumt;
   sump;
end;"""


SCC_BATCH_FILE_TEMPLATE = """#!/bin/bash -l
#$ -cwd
#$ -N {jobname}
#$ -pe omp 4
#$ -v OMP_NUM_THREADS=4
#$ -l h_rt={h}:00:00
#$ -l mem_per_core=3G
#$ -l avx
module load openmpi/3.1.4
module load mrbayes/3.2.7
if [ "$1" == "--append" ]; then
    mpirun -c 4 mb {mbappendfile}
else
    mpirun -c 4 mb {mbfile}
fi

"""

SUBMIT_MISSING_JOBS_TEMPLATE = """#!/bin/bash
for b in {{1..{B}}}; do
    if [ ! -f {jobname}.bootstrap$b.con.tre ]; then
        echo "running bootstrap #$b"
        if [ -f {jobname}.bootstrap$b.ckp ] && [ "$1" != "--rerun" ]; then
            qsub -t $b {batchfilename} --append
        else
            qsub -t $b {batchfilename}
        fi
    fi
done
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nexfile')
    parser.add_argument('outgroup')
    parser.add_argument('-m', '--model', default='JC')
    parser.add_argument('--results', default='results',
                        help='results directory')
    parser.add_argument('-B', type=int, default=0,
                        help='number of bootstrap datasets')
    parser.add_argument('-c', type=float, default=1.0)
    parser.add_argument('-a', type=float,  default=1.0,
                        help='bootstrap dataset size is c*N^a')
    parser.add_argument('--hours', type=int, default=1,
                        help='runtime (hours)')
    parser.add_argument('--ngen', type=int, default=1000000)
    parser.add_argument('--by-codon-pos', action='store_true',
                        help='bootstrap should preserve codon position')
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()



def generate_bootstrap_nexus_files(nex_filepath, nex_outpathbase, B, c, a, bycodonpos):
    outputfiles = [nex_outpathbase + '.bootstrap{}.nex'.format(b+1) for b in range(B)]
    nexorig = NexusReader(nex_filepath)
    nchar = nexorig.data.nchar
    ncharboot = max(1, int(c*nchar**a))
    if bycodonpos:
        if nchar % 3 != 0:
            sys.exit('sequence length must be divisible by three to use --by-codon-pos')
        ncharboot = 3*max(1, ncharboot//3)
    for b in range(B):
        np.random.seed(341+b)
        if bycodonpos:
            inds = 3*np.random.choice(nchar//3, ncharboot) + (np.arange(ncharboot) % 3)
        else:
            inds = np.random.choice(nchar, size=ncharboot)
        nexboot = NexusReader(nex_filepath)
        for taxon in nexboot.data.taxa:
            nexboot.data.matrix[taxon] = np.array(nexorig.data.matrix[taxon])[inds].tolist()
        nexboot.write_to_file(outputfiles[b])
    return outputfiles



def main():
    args = parse_args()
    model = args.model
    ngen = args.ngen
    if ngen < 1:
        sys.exit('--ngen must be positive')
    if args.hours <= 0 or args.hours >= 24:
        sys.exit('-h must be at least 1 and at most 23')
    num_samples = 20000
    if args.B > 0:
        num_samples = max(100, num_samples/args.B)
    samplefreq = max(1, ngen // num_samples)
    printfreq = max(1, ngen // 500)
    nex_filename = os.path.basename(args.nexfile)
    basename = os.path.splitext(nex_filename)[0] + '-' + model
    if args.B > 0:
        if args.c <= 0:
            sys.exit('argument to -c must be positive')
        if args.a <= 0:
            sys.exit('argument to -a must be positive')
        basename += '-c-{}-a-{}'.format(args.c, args.a)
    baseoutfile = os.path.splitext(nex_filename)[0]

    outputdir = os.path.join(args.results, basename)
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(os.path.join(outputdir, 'logs'), exist_ok=True)
    shutil.copy(args.nexfile, outputdir)

    config_info = dict(baseoutfile=baseoutfile,
                       nexfile=nex_filename,
                       outgroup=args.outgroup,
                       model=model)
    if args.B > 0:
        config_info['B'] = args.B
        config_info['c'] = args.c
        config_info['a'] = args.a
    with open(os.path.join(outputdir, 'experiment.config'), 'w') as f:
        json.dump(config_info, f)

    def create_mb_file(nexfile, base, append):
        if append:
            mcmcargs = ' append=yes'
            mb_filename = base + '.append.mb'
        else:
            mcmcargs = ''
            mb_filename = base + '.mb'
        mb_file_content = MB_FILE_CONTENT_TEMPLATE.format(
            nexfile=nexfile, outgroup=args.outgroup,
            modelconfig=CONFIGS[model], ngen=ngen, samplefreq=samplefreq,
            printfreq=printfreq, baseoutfile=base, mcmcargs=mcmcargs)
        with open(os.path.join(outputdir, mb_filename), 'w') as f:
            f.write(mb_file_content)
        return mb_filename

    def create_scc_batch_file(mbfile, base):
        scc_batch_file_content = SCC_BATCH_FILE_TEMPLATE.format(
            jobname=base, h=args.hours, mbfile=mbfile+'.mb',
            mbappendfile=mbfile+'.append.mb')
        scc_batch_file = 'run_experiment.sh'
        with open(os.path.join(outputdir, scc_batch_file), 'w') as f:
            f.write(scc_batch_file_content)
        return scc_batch_file

    if args.B <= 0:
        create_mb_file(nex_filename, baseoutfile, False)
        create_mb_file(nex_filename, baseoutfile, True)
        scc_batch_filename = create_scc_batch_file(baseoutfile,
                                                   baseoutfile)
        batch_cmd = 'qsub {}'.format(scc_batch_filename)
    else:
        bootstrapfiles = generate_bootstrap_nexus_files(
            args.nexfile, os.path.join(outputdir, baseoutfile), args.B, args.c,
            args.a, args.by_codon_pos)

        for b, bootfile in enumerate(bootstrapfiles):
            bootfilename = os.path.basename(bootfile)
            mbbasefilename = baseoutfile+'.bootstrap{}'.format(b+1)
            create_mb_file(bootfilename, mbbasefilename, False)
            create_mb_file(bootfilename, mbbasefilename, True)
        mb_filename_generic = baseoutfile+'.bootstrap${SGE_TASK_ID}'
        scc_batch_filename = create_scc_batch_file(
            mb_filename_generic, baseoutfile+'.bootstrap')

        submit_missing_jobs_content = SUBMIT_MISSING_JOBS_TEMPLATE.format(
            B=args.B, jobname=baseoutfile, batchfilename=scc_batch_filename)
        submit_missing_jobs_file = 'submit_jobs_for_missing_results.sh'
        with open(os.path.join(outputdir, submit_missing_jobs_file), 'w') as f:
            f.write(submit_missing_jobs_content)
        batch_cmd = 'qsub -t 1-{} {}'.format(args.B, scc_batch_filename)
    if args.run:
        print('output directory:', outputdir)
        print(batch_cmd)
        res = subprocess.run(batch_cmd, cwd=outputdir, shell=True, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print('***batch command not run successfully***')
    else:
        print('cd', outputdir + ';', batch_cmd + ';', 'cd -')


if __name__ == '__main__':
    main()
