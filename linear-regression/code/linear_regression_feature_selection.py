import os
import sys
import argparse

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from itertools import combinations, chain
import numpy as np
from numpy.testing import assert_allclose
import scipy as sp
from scipy import stats
from scipy.special import digamma, polygamma
from sklearn.preprocessing import add_dummy_feature, scale
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import (load_boston, load_diabetes,
                              fetch_california_housing)
import pandas as pd

from util import (compute_model_mismatch,
                  generate_synthetic_data,
                  nonzero_sparse_inds,
                  plot_model_mismatch,
                  plot_model_mismatch_hist)


def load_building(return_X_y=True):
    XY = np.genfromtxt('data/residential-building.csv', delimiter=',', dtype=float)
    XY = scale(XY)
    X = XY[:,:-1]
    y = XY[:,-1]
    return X, y


DATASETS = { 'boston-housing' : load_boston,
             'diabetes' : load_diabetes,
             'california-housing' : fetch_california_housing,
             'res-building' : load_building,
             }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('-B', type=int, default=50,
                        help='minimum number of bootstrap samples')
    parser.add_argument('-r', type=int, default=10,
                        help='number of experiment repetitions')
    parser.add_argument('-N', type=int, default=100,
                        help='number of observations')
    parser.add_argument('-D', type=int, default=10,
                        help='data dimension')
    parser.add_argument('-s', type=int, default=0,
                        help='maximum number of nonzeros')
    parser.add_argument('-c', type=float, default=1.,
                        help='relative size of bootstrapped datasets')
    parser.add_argument('-a', type=float,  nargs='*', default=[1.],
                        help='scale bootstrap dataset size like N^a')
    parser.add_argument('--results', default='results',
                        help='results directory')
    parser.add_argument('--figures', default='figures',
                        help='figures directory')
    parser.add_argument('-k', type=int, default=1,
                        help='split real dataset in k pieces')
    parser.add_argument('--a0', type=float, default=1.)
    parser.add_argument('--b0', type=float, default=1.)
    parser.add_argument('--sigma0', type=float, default=1.)
    parser.add_argument('--include-full-results', action='store_true',
                        help='include full real dataset results')
    parser.add_argument('--other', action='store_true',
                        help='if sparse, merge all zero components in figures')
    parser.add_argument('-e', '--exclude', type=int, nargs='*', default=[],
                        help='exclude component when generating plots')
    return parser.parse_args()


def linreg_log_marginal_likelihood(X, y, w=None, mask=None, a0=1, b0=1, sigma0=1,
                                   compute_means_and_vars=False, verbose=False):
    # Prior: Normal-inverse-gamma(mu=0, lambda=1/sigma0^2, a0, b0)
    # X : N x D
    # y : N x 1
    # w : N x 1 (weights on each observation)
    # mask : D x 1 (binary inclusion variable for each component)
    if w is None:
        w = np.ones(X.shape[0])
    if mask is None:
        mask = np.full(X.shape[1], True)
    N = np.sum(w)
    D = np.sum(mask)
    Xw = np.sqrt(w[:,np.newaxis]) * X[:,mask]
    yw = np.sqrt(w) * y
    prec0 = 1/sigma0**2
    # posterior precision
    precN = Xw.T.dot(Xw)
    precN[np.diag_indices_from(precN)] += prec0
    Xy = Xw.T.dot(yw)
    muN = np.linalg.solve(precN, Xy)
    aN = a0 + 0.5*N
    bN = b0 + 0.5*(yw.dot(yw) - muN.dot(Xy))
    if verbose:
        print(aN, bN, bN/(aN-1))
    _, logdet_precN = np.linalg.slogdet(precN)
    lml = -0.5*N*np.log(2*np.pi) + 0.5*D*np.log(prec0) - 0.5*logdet_precN \
           + a0*np.log(b0) - aN*np.log(bN) + sp.special.loggamma(aN) - \
           sp.special.loggamma(a0)
    if not compute_means_and_vars:
        return lml
    covN = bN / (aN - 1) * np.linalg.inv(precN)
    # return the mean and variance of -log(prec)
    mean = np.concatenate([muN, [digamma(aN) - digamma(bN)]])
    var = np.concatenate([np.diag(covN), [polygamma(1, aN)]])
    return lml, mean, var


def compute_all_log_marginal_likelihoods(X, y, max_nonzeros=None,
                                         nested_models=False, **kwargs):
    dim = X.shape[1]
    if max_nonzeros is None:
        max_nonzeros = dim
    if nested_models:
        num_models = dim
    else:
        num_models = int(np.sum([sp.special.binom(dim, k) for k in range(max_nonzeros+1)]))
    if num_models > 2**22: # 2^22 is about 4.2M
        sys.exit('too many models! (# of models = {})'.format(num_models))
    lmls = np.full(num_models, np.nan)
    if nested_models:
        models_iter = [range(k) for k in range(dim)]
    else:
        models_iter = chain.from_iterable(combinations(range(dim), k) for k in range(max_nonzeros+1))
    models = []
    _, mean, var = linreg_log_marginal_likelihood(X, y, compute_means_and_vars=True,
                                                  verbose=False, **kwargs)
    for i, model in enumerate(models_iter):
        mask = np.full(dim, False)
        mask[list(model)] = True
        lmls[i] = linreg_log_marginal_likelihood(X, y, mask=mask, **kwargs)
        models.append(model)
    probs = np.exp(lmls - sp.special.logsumexp(lmls))
    return probs, lmls, np.array(models, dtype=object), mean, var


def compute_model_probs(lmls, models, D0, D, is_probs=False):
    if D0 <= 0 or D0 >= D:
        raise ValueError('expected number of nonzeros must be > 0 and < D')
    if D <= 0:
        raise ValueError('D must be positive')
    if is_probs:
        lmls = np.log(lmls)
    model_sizes = np.array([len(model) for model in models])
    empty_model = model_sizes == 0
    lmls[empty_model] = -np.inf
    prob_inclusion = D0/D
    lmls += model_sizes * np.log(prob_inclusion)
    lmls += (D - model_sizes) * np.log(1 - prob_inclusion)
    probs = np.exp(lmls - sp.special.logsumexp(lmls))
    return probs


def compute_pips(probs, models, D):
    pips = np.zeros(D)
    for p, m in zip(probs, models):
        pips[list(m)] += p
    return pips


def run_bootstraps(B, bbprobs, bbmeans, bbvars, bb_rep_pips, bbN, p,
                   X, y, s, a0, b0, sigma0, nested_models, models, D):
    Bcurr = bbprobs.shape[0]
    if Bcurr < B:
        Bnew = B - Bcurr
        bbweights = np.random.multinomial(bbN, p, Bnew)
        bbprobs = np.concatenate((bbprobs, np.zeros((Bnew, bbprobs.shape[1]))))
        bbmeans = np.concatenate((bbmeans, np.zeros((Bnew, bbmeans.shape[1]))))
        bbvars = np.concatenate((bbvars, np.zeros((Bnew, bbvars.shape[1]))))
        bb_rep_pips = np.concatenate((bb_rep_pips, np.zeros((Bnew, bb_rep_pips.shape[1]))))
        for b in range(Bcurr, B):
            bbprobs[b], _, _, bbmeans[b], bbvars[b] = \
                compute_all_log_marginal_likelihoods(X, y, s, w=bbweights[b - Bcurr],
                                                     a0=a0, b0=b0, sigma0=sigma0,
                                                     nested_models=nested_models)
            bb_rep_pips[b] = compute_pips(bbprobs[b], models, D)
    return bbprobs, bbmeans, bbvars, bb_rep_pips


def run_single_experiment(seed, X, y, c, a, B, s, a0, b0, sigma0, filepath,
                          nested_models, max_stderr=0.02):
    N, D = X.shape
    s = s if s > 0 and s <= D else D
    # standard Bayes
    bprobs, _, models, bmean, bvar = compute_all_log_marginal_likelihoods(
        X, y, s, a0=a0, b0=b0, sigma0=sigma0, nested_models=nested_models)
    # BayesBag
    p = np.ones(N)/N
    bbN = max(1, int(c*N**a)) #// 1.5 #// max(1,np.log10(N))
    # ensure that all observations are used with probability > .99
    B = int(max(B, (N - .5) * np.log(N/.01) // bbN))
    bbprobs = np.zeros((0, bprobs.size))
    bbmeans = np.zeros((0, D+1))
    bbvars = np.zeros((0, D+1))
    bb_rep_pips = np.zeros((0, D))
    err = np.inf
    while True:
        bbprobs, bbmeans, bbvars, bb_rep_pips = \
            run_bootstraps(B, bbprobs, bbmeans, bbvars, bb_rep_pips, bbN, p,
                X, y, s, a0, b0, sigma0, nested_models, models, D)
        bbpips_stderrs = np.std(bb_rep_pips, axis=0) / np.sqrt(B)
        err = np.max(bbpips_stderrs)
        if err > max_stderr:
            B = int(B * 1.1 + 1)
        else:
            break
    mean_bbprobs = np.mean(bbprobs, axis=0)
    bbmean = np.mean(bbmeans, axis=0)
    bbvar = np.mean(bbmeans**2 + bbvars, axis=0) - bbmean**2
    # compute posterior inclusion probabilities
    bpips = compute_pips(bprobs, models, D)
    bbpips = compute_pips(mean_bbprobs, models, D)
    
    # save results
    np.savez(filepath, seed=seed, X=X, y=y, models=models,
             bayesprobs=bprobs, bbprobs=mean_bbprobs,
             bpips=bpips, bbpips=bbpips,
             bmean=bmean, bvar=bvar,
             bbmean=bbmean, bbvar=bbvar,
             bbpips_stderrs=bbpips_stderrs)


def run_single_synthetic_experiment(mode, seed, rep, c, a, B, D, s, N,
                                    a0, b0, sigma0, base_outpath):
    filepath = base_outpath + '-rep-{}.npz'.format(rep)
    if not os.path.exists(filepath):
        print('rep', rep)
        X, y, beta_opt = generate_synthetic_data(mode, seed, D, N)
        run_single_experiment(seed, X, y, c, a, B, s, a0, b0, sigma0, filepath,
                              mode.startswith('synth-inliers'))
    return filepath

def bb_label(a):
    if a == 1:
        return r'BayesBag ($M = N$)'
    return r'BayesBag ($M = N^{{{}}}$)'.format(a)


def _plot_comparison(pip_dicts, bkey, bbkey, mode, D, pip0, true_inds, other_inds, exclude_comps=[],
                     kind='point', use_log=True, sig_inds=None, ax=None):
    if ax is not None:
        kind = 'point'
    if kind == 'point':
        kwargs = dict(join=False, legend=False, legend_out=False, dodge=.3,
                      ci='sd')
    elif kind == 'swarm':
        kwargs = dict(legend=False, legend_out=False, dodge=True)
    elif kind == 'strip':
        kwargs = dict(legend=False, legend_out=False, dodge=.3, jitter=.3)
    elif kind == 'box' or kind == 'boxen':
        kwargs = dict(legend=False, legend_out=False)
    elif kind == 'violin':
        kwargs = dict(legend=False, legend_out=False, inner=None)
    elif kind != 'density':
        raise ValueError('unsupported plot kind "{}"'.format(kind))
    bbpips_dict = {}
    for a, pip_dict in pip_dicts.items():
        bpips = pip_dict[bkey]
        bbpips_dict[a] = pip_dict[bbkey]
    if bpips.ndim > 1:
        num_samples = bpips.shape[0]
    else:
        num_samples = 1
        bpips = bpips[np.newaxis,:]
        bbpips_dict = { a : bbpips[np.newaxis,:] for a, bbpips in bbpips_dict.items() }
    if num_samples > 20 and kind == 'strip':
        kwargs['alpha'] = .35
    if mode.startswith('synth'):
        num_components = D
        if true_inds is None:
            true_inds = np.arange(1, num_components+1)
            other_inds = None
        if other_inds is not None and len(other_inds) > 0:
            cols = pd.MultiIndex.from_product([['pip'], list(true_inds) + ['other']],
                                                   names=['value', 'component'])
            num_components = len(true_inds) + 1
        else:
            sig_inds = np.full(num_components, True)
            if len(exclude_comps) > 0:
                sig_inds[np.array(exclude_comps)-1] = False
            components = np.arange(1, sig_inds.size+1)[sig_inds]
            bpips = bpips[:,sig_inds]
            bbpips_dict = { a : bbpips[:,sig_inds] for a, bbpips in bbpips_dict.items() }
            num_components = np.sum(sig_inds)
            cols = pd.MultiIndex.from_product([['pip'], components],
                                              names=['value', 'component'])
    else:
        # exclude constant offset
        bpips = bpips[:,1:]
        bbpips_dict = { a : bbpips[:,1:] for a, bbpips in bbpips_dict.items() } 
        cutoff = min(3*pip0, .25)
        if sig_inds is None:
            sig_inds = np.sum(bpips > cutoff, axis=0) > 0
            for a, bbpips in bbpips_dict.items():
                bb_sig_inds = np.sum(bbpips > cutoff, axis=0) > 0
                sig_inds = np.logical_or(sig_inds, bb_sig_inds)
        components = np.arange(1, sig_inds.size+1)[sig_inds]
        bpips = bpips[:,sig_inds]
        bbpips_dict = { a : bbpips[:,sig_inds] for a, bbpips in bbpips_dict.items() } 
        num_components = np.sum(sig_inds)
        cols = pd.MultiIndex.from_product([['pip'], components],
                                          names=['value', 'component'])
    aspect = max(1.5, (1 + len(pip_dicts))/2 * num_components / 6.5)
    bind = pd.MultiIndex.from_product([['Bayes'], np.arange(num_samples)],
                                      names=['type', 'rep'])
    bdf = pd.DataFrame(bpips, index=bind, columns=cols)
    df = bdf.stack().reset_index(level=['component', 'type'])
    for a, bbpips in bbpips_dict.items():
        bbind = pd.MultiIndex.from_product([[bb_label(a)], np.arange(num_samples)],
                                       names=['type', 'rep'])
        bbdf = pd.DataFrame(bbpips, index=bbind, columns=cols)
        stackedbbdf = bbdf.stack().reset_index(level=['component', 'type'])
        df = df.append(stackedbbdf)
    if use_log:
        raise ValueError("use_log not currently supported")
        if kind == 'point':
            print('Bayes:', np.mean(np.log(1-bpips),axis=0))
            print('BayesBag:', np.mean(np.log(1-bbpips),axis=0))
        if other_inds is not None:
            other = df.component == 'other'
            df.loc[other, 'pip'] = 1 - df.loc[other, 'pip']
        df['negative log loss'] = np.log(df.pip+1e-10)
        ylabel = 'negative log loss'
    else:
        ylabel = 'pip'
    if kind == 'density':
        if other_inds is not None:
            not_other = df.component != 'other'
            df.loc[not_other, 'pip'] = 1 - df[not_other].pip
        g = sns.FacetGrid(df, col='component', hue='type')
        g = g.map(sns.distplot, ylabel, kde=False, norm_hist=True,
                  hist_kws=dict(cumulative=True),
                  bins=np.linspace(0,1,21)).set(xlim=(0,1))
        return num_components, sig_inds, g
    elif ax is None:
        g = sns.catplot(x='component', y=ylabel, hue='type', data=df,
                        kind=kind, height=4, aspect=aspect, **kwargs)
        if ylabel == 'pip':
            g.set_ylabels('posterior inclusion\nprobability')
        return num_components, sig_inds, g
    else:
        nc = len(ax.collections)
        ax = sns.pointplot(x='component', y=ylabel, hue='type', data=df,
                           markers='*', ax=ax, join=False, dodge=.3, ci='sd')
                           #plot_kws=dict(alpha=.8))
        plt.setp(ax.collections[nc:], alpha=.6)
        if ylabel == 'pip':
            ax.set_ylabel('posterior inclusion\nprobability')



def plot_comparison(outfile, pip_dicts, mode, D0, D, kind, extra=False, 
                    mark_true=None, true_inds=None, other_inds=None, exclude_comps=[],
                    use_log=True):
    # bpips, bbpips
    pip0 = D0 / D
    outfile += '-' + kind
    num_components, sig_inds, g = _plot_comparison(pip_dicts, 'bpips', 'bbpips',
                                                   mode, D, pip0, 
                                                   true_inds, other_inds,
                                                   kind=kind, use_log=use_log,
                                                   exclude_comps=exclude_comps,)
    if extra:
        _plot_comparison(pip_dicts, 'bpips_extra', 'bbpips_extra', mode, D, 
                         pip0, true_inds, other_inds, exclude_comps=exclude_comps,
                         sig_inds=sig_inds, use_log=use_log, ax=plt.gca())
    if kind != 'density' and not use_log:
        xlim = plt.xlim()
        if np.any([np.any(pip_dict['bpips'] > .99) or np.any(pip_dict['bbpips'] > .99) for pip_dict in pip_dicts.values()]):
            plt.plot([-.5,num_components-.5], [1,1], ':', color='gray')
        plt.plot([-.5,num_components-.5], [pip0,pip0], ':', color='gray')
        ylim = plt.ylim()
        for i in range(num_components//2):
            rect = patches.Rectangle((2*i+.5,ylim[0]),1,ylim[1]-ylim[0],
                                     edgecolor='none',facecolor='gray', alpha=0.2)
            plt.gca().add_patch(rect)
        if mark_true is not None:
            mark_inds = np.searchsorted(np.flatnonzero(sig_inds), mark_true)
            for mark_ind in mark_inds:
                rect = patches.Rectangle((mark_ind-.4975,ylim[0]+.005),.98,(ylim[1]-ylim[0])-.01,
                                         edgecolor='red',facecolor='none', alpha=1)
                plt.gca().add_patch(rect)
        plt.gca().set_xlim([-.5, num_components-.5])
        plt.gca().set_ylim(ylim)
        sns.despine()
    g.add_legend(title='')
    plt.savefig(outfile + '.pdf', bbox_inches='tight')
    if kind != 'density':
        plt.gca().get_legend().remove()
        plt.savefig(outfile + '-no-legend.pdf', bbox_inches='tight')


def plot_diffs(outfile, pip_diffs):
    kind = 'strip'
    kwargs = dict()
    if pip_diffs.ndim > 1:
        num_samples = pip_diffs.shape[0]
    else:
        num_samples = 1
        pip_diffs = pip_diffs[np.newaxis,:]
    num_components = pip_diffs.shape[1]
    aspect = max(1.5, num_components / 6.5)
    cols = pd.MultiIndex.from_product([['pip difference'], np.arange(1, num_components+1)],
                                      names=['value', 'component'])
    ind = pd.MultiIndex.from_product([np.arange(num_samples)],
                                      names=['rep'])
    df = pd.DataFrame(pip_diffs, index=ind, columns=cols).stack().reset_index(level=['component'])
    sns.catplot(x='component', y='pip difference', data=df,
                kind=kind, height=4, aspect=aspect, **kwargs)
    plt.plot([-.5,num_components-.5], [0,0], ':', color='gray')
    sns.despine()
    plt.savefig(outfile, bbox_inches='tight')


def print_most_likely_models(probs, models, filename, cutoff=.01):
    sorted_inds = np.argsort(probs)[::-1]
    if cutoff >= 1:
        num_include = cutoff
    else:
        num_include = np.sum(probs > cutoff)
    with open(filename, 'w') as f:
        for i in sorted_inds[:num_include]:
            print('{:.3f} {}'.format(probs[i], models[i]), file=f)

def diff_test(diffs, name, alpha):
    mean_diff = np.mean(diffs)
    _, p_value = stats.wilcoxon(diffs)
    if p_value < alpha:
        larger = 'BayesBag' if mean_diff > 0 else 'Bayes'
        sig_str = '{} significantly larger by {:.3f}'.format(larger, np.abs(mean_diff))
    else:
        sig_str = 'neither significantly larger'
    print('{}: {} (p-value = {:.2g})'.format(name, sig_str, p_value))


def run_synthetic_experiment(args, base_outpath, base_figpath, base_a_outpath):
    D = args.D
    D0 = 3
    mark_true = None
    if args.mode.startswith('synth-inliers'):
        _, _, datatype = args.mode.split('-')
        D0 = D/2
        mark_true = None
        true_inds = []
    else:
        _, corrtype, sparsity, regtype, noisetype = args.mode.split('-')
        is_sparse = sparsity[1:] == 'sparse'
        if is_sparse:
            D0 = int(sparsity[0])
            mark_true = nonzero_sparse_inds(D0, args.D)
            print('nonzero component #s:', *(mark_true+1))
        if is_sparse and args.other:
            base_figpath += '-other'
            true_inds = mark_true
            mark_true = None
        else:
            true_inds = np.arange(D)
    other_inds = [i for i in np.arange(D) if i not in true_inds]
    n_reps = args.r
    seed_start = 1843
    files_dict = { a : list() for a in args.a }
    for a, files in files_dict.items():
        afmt = '-a-{}'.format(a) if a != 1 else ''
        for rep in range(n_reps):
            files.append(run_single_synthetic_experiment(
                args.mode, seed_start+rep, rep+1, args.c, a, args.B, args.D, args.s,
                args.N, args.a0, args.b0, args.sigma0, base_a_outpath.format(afmt)))
    names = [('bayesprobs', 'bpips'), ('bbprobs', 'bbpips')]
    pn_size = len(true_inds) + (len(other_inds)>0)
    pip_dicts = { a : { pn : np.zeros((n_reps, pn_size)) for _, pn in names } for a in args.a }
    pip_diffs =  np.zeros((n_reps, D))
    model_mismatch = np.zeros((n_reps, D+1))
    for a, files in files_dict.items():
        pip_dict = pip_dicts[a]
        for rep, filepath in enumerate(files):
            result = np.load(filepath, allow_pickle=True)
            bvar = result['bvar']
            bbvar = result['bbvar']
            model_mismatch[rep] = compute_model_mismatch(bvar, bbvar, args.N,
                                                         args.a0, args.b0,
                                                         args.sigma0)
            all_pips = {}
            for probsn, pipsn in names:
                probs = compute_model_probs(result[probsn], result['models'],
                                            D0=D0, D=D, is_probs=True)
                all_pips[pipsn] = compute_pips(probs, result['models'], D)
                pips = pip_dict[pipsn]
                pips[rep,:len(true_inds)] = all_pips[pipsn][true_inds]
                if len(other_inds) > 0:
                    pips[rep,-1] = np.max(all_pips[pipsn][other_inds])
            pip_diffs[rep] = all_pips['bbpips'] - all_pips['bpips']
        mean_pip_diffs = np.mean(pip_diffs, axis=0)
        for d in range(D):
            diff_test(pip_diffs[:,d], 'component {}'.format(d+1), .05/D)
        if len(other_inds) > 0:
            max_pip_diffs = pip_dict['bbpips'][:,-1] - pip_dict['bpips'][:,-1]
            diff_test(max_pip_diffs, 'max component', .05/D)

        for name, pips in pip_dict.items():
            max_comps = np.argmax(pips, axis=1)
            unique, counts = np.unique(max_comps, return_counts=True)
            print(name, list(zip(unique+1, counts)))


    figpath = base_figpath + '-reps-{}'.format(n_reps)
    if args.other:
        plt.plot()

    for kind in ['strip']: # ['swarm', 'strip']:
        plot_comparison(base_figpath, pip_dicts, D0=D0, D=D, mode=args.mode, 
                        kind=kind, use_log=False, mark_true=mark_true,
                        exclude_comps=args.exclude)
    if len(args.a) == 1:
        with sns.plotting_context('notebook', font_scale=2.75,
                                  rc={'lines.linewidth': 2}):
            plot_model_mismatch_hist(figpath, model_mismatch, ncols=2)
            plot_model_mismatch_hist(figpath, model_mismatch, 2, ncols=2)


def _format_array(a):
    return ', '.join('{:.4f}'.format(x) for x in a)


def _run_real_data_experiment(args, a, base_outpath, X, y, seed, D, D0):
    if args.k > 1:
        splits = KFold(n_splits=args.k, shuffle=True, random_state=seed).split(X)
    else:
        splits = [(None, np.arange(X.shape[0]))]
    files = []
    for rep, (_, indices) in enumerate(splits):
        filepath = base_outpath + '-rep-{}.npz'.format(rep+1)
        files.append(filepath)
        if not os.path.exists(filepath):
            np.random.seed(seed)
            run_single_experiment(seed+rep+1, X[indices], y[indices], args.c, a,
                                  args.B, args.s, args.a0, args.b0, args.sigma0,
                                  filepath, False)
    names = [('bayesprobs', 'bpips'), ('bbprobs', 'bbpips')]
    pip_dict = { pn : np.zeros((args.k, X.shape[1])) for _, pn in names }
    model_mismatch = np.zeros((args.k, D+1))
    for rep, filepath in enumerate(files):
        result = np.load(filepath, allow_pickle=True)
        bvar = result['bvar']
        bbvar = result['bbvar']
        model_mismatch[rep] = compute_model_mismatch(bvar, bbvar, args.N,
                                                     args.a0, args.b0,
                                                     args.sigma0)
        for probsn, pipsn in names:
            probs = compute_model_probs(result[probsn], result['models'],
                                        D0=D0, D=D, is_probs=True)
            pip_dict[pipsn][rep] = compute_pips(probs, result['models'], D)
    return pip_dict, model_mismatch


def run_real_data_experiment(args, base_outpath, base_figpath, base_a_outpath):
    if args.mode not in DATASETS:
        sys.exit('invalid mode')
    Xorig, yorig = DATASETS[args.mode](return_X_y=True)
    X = scale(Xorig, axis=0)
    y = scale(yorig)
    D = X.shape[1]
    D0 = 3
    seed = 2453
    pip_dicts = dict()
    for a in args.a:
        afmt = '-a-{}'.format(a) if a != 1 else ''
        pip_dicts[a], model_mismatch = _run_real_data_experiment(
            args, a, base_a_outpath.format(afmt), X, y, seed, D, D0)
        if args.include_full_results:
            k_orig = args.k
            args.k = 1
            base_outpath_extra = os.path.join(args.results, base_filename(args, a))
            pip_dict_extra, _ = _run_real_data_experiment(args, a, base_outpath_extra,
                                                          X, y, seed, D, D0)
            args.k = k_orig
            pip_dicts[a]['bpips_extra'] = pip_dict_extra['bpips']
            pip_dicts[a]['bbpips_extra'] = pip_dict_extra['bbpips']
    if args.k == 1:
        print('model mismatch = {:.3f}'.format(np.max(model_mismatch)))
    if len(args.a) == 1:
        plot_model_mismatch(base_figpath, model_mismatch, 'strip')
        plot_model_mismatch(base_figpath, model_mismatch, 'strip', True)
    for kind in ['swarm']: #, 'point']:
        plot_comparison(base_figpath, pip_dicts, D0=D0, D=D, mode=args.mode, 
                        kind=kind, use_log=False, extra=args.include_full_results)


def add_maybe(s, value, default, fmt):
    return s + (fmt.format(value) if value != default else '')


def base_filename(args, a):
    sizestr = add_maybe('', args.c, 1., '-c-{}')
    if a is not None:
        sizestr = add_maybe(sizestr, a, 1., '-a-{}')
    else:
        sizestr += '{}'
    hyperstr = add_maybe('', args.a0, 1., '-a0-{}')
    hyperstr = add_maybe(hyperstr, args.b0, 1., '-b0-{}')
    hyperstr = add_maybe(hyperstr, args.sigma0, 1., '-sigma0-{}')
    if args.mode.startswith('synth'):
        return '{}{}-B-{}-D-{}-s-{}-N-{}{}'.format(args.mode, sizestr, args.B,
                                                   args.D, args.s, args.N,
                                                   hyperstr)
    elif args.mode.startswith('nott-kohn'):
        return '{}{}-B-{}-s-{}-N-{}{}'.format(args.mode, sizestr, args.B, args.s,
                                              args.N, hyperstr)
    else:
        return '{}{}-B-{}-s-{}-k-{}{}'.format(args.mode, sizestr, args.B, args.s,
                                              args.k, hyperstr)


def main():
    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2})

    args = parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    as_str = '-'.join(['{}'.format(a) for a in args.a])
    base_outpath = os.path.join(args.results, base_filename(args, as_str))
    base_figpath = os.path.join(args.figures, base_filename(args, as_str))
    base_a_outpath = os.path.join(args.results, base_filename(args, None))
    if args.mode.startswith('synth'):
        run_synthetic_experiment(args, base_outpath, base_figpath, base_a_outpath)
    else:
        run_real_data_experiment(args, base_outpath, base_figpath, base_a_outpath)



if __name__ == '__main__':
    main()
