import os
import sys

import numpy as np
import scipy as sp
from scipy.special import polygamma
from scipy import stats
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import scale
import pandas as pd

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

    

def compute_model_mismatch(bvar, bbvar, N, a0=1, b0=1, sigma0=1):
    if a0 <= 1:
        beta_var0 = [np.inf]*(bvar.size-1)
    else:
        beta_var0 = [b0/(a0-1)*sigma0**2]*(bvar.size-1)
    var0 = np.array(beta_var0 + [polygamma(1, a0)])
    sig2_div_Nv0 = 1/(var0/bvar - 1)
    sig2_div_s2 = (1 - bvar/var0)/(bbvar/bvar - 1)
    tmp = .5*(1 + sig2_div_s2)
    copts = -np.ones_like(tmp)
    if bvar.shape != bbvar.shape or tmp.shape != bvar.shape or var0.shape != bvar.shape:
        raise ValueError(var0.shape, bvar.shape, bbvar.shape, tmp.shape)
    valid_inds = tmp**2 >= sig2_div_Nv0
    tmp_valid = tmp[valid_inds]
    sig2_div_Nv0_valid = sig2_div_Nv0[valid_inds]
    copts[valid_inds] = tmp_valid - sig2_div_Nv0_valid + np.sqrt(tmp_valid**2 - sig2_div_Nv0_valid)
    model_mismatch = 2/copts - 1
    model_mismatch[copts < 1] = np.nan
    model_mismatch[bvar > var0] = np.nan
    model_mismatch[bbvar > var0] = np.nan
    return model_mismatch


def nonzero_sparse_inds(k, D):
    inds = np.array([(i*(D+.5))//(k+1) for i in range(1,k+1)], dtype=int) - 1
    return inds


def get_beta(sparsity, D):
    if sparsity == 'dense':
        return 2**(2-np.arange(D)/2)
    if sparsity[-6:] != 'sparse':
        sys.exit('invalid sparsity type {}'.format(sparsity))
    k = int(sparsity[:-6])
    beta = np.zeros(D)
    inds = nonzero_sparse_inds(k, D)
    beta[inds] = 1
    return beta


def generate_y_for_X_beta(noisetype, X, beta):
    means = X.dot(beta)
    N = means.size
    err = np.random.randn(N)
    if noisetype == 'gaussian':
        y = means + err
    elif noisetype == 'heavy':
        df = 4
        y = means + err/np.sqrt(np.random.chisquare(df, N)/df)
    else:
        sys.exit('invalid noise type {}'.format(noisetype))
    return y


def generate_synthetic_data(mode, seed, D, N):
    rng = np.random.default_rng()
    mode_parts = mode.split('-')
    if len(mode_parts) == 3 and mode_parts[1] == 'inliers':
        datatype = mode_parts[2]
        is_zero = rng.integers(2, size=N)
        Xbase = 2*rng.random((N,1)) - 1
        Xbase[is_zero, :] = 0
        if datatype == 'poly':
            X = Xbase**np.arange(0,D)[np.newaxis,:]
        elif datatype == 'fourier':
            if D % 2 != 1:
                 sys.exit('D must be odd when using fourier')
            Xrescaled = Xbase*np.arange(1,D//2+1)[np.newaxis,:]
            X = np.concatenate([Xbase/np.sqrt(2), np.sin(Xrescaled), np.cos(Xrescaled)], axis=1) / np.pi
        y = np.sqrt(0.05)*rng.normal(size=N)
        beta_opt = np.zeros(D)
        y[is_zero] = 0
    else:
        if len(mode_parts) != 5:
            sys.exit('invalid mode')
        _, corrtype, sparsity, regtype, noisetype = mode_parts
        np.random.seed(seed)
        if corrtype.startswith('corrsimple'):
            if len(corrtype) == 10:
                stdev = 1
            else:
                try:
                    stdev = float(corrtype[10:])
                except ValueError:
                    sys.exit('invalid correlation type {}'.format(corrtype))
            scale = 8
            locs = np.linspace(0,D/scale,D,endpoint=False).reshape(-1,1)
            cov = stdev**2 * pairwise_kernels(locs, metric='rbf')
            X = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, N)
        elif corrtype.startswith('corr'):
            if len(corrtype) == 4:
                scale = 8
            else:
                try:
                    scale = float(corrtype[4:])
                except ValueError:
                    sys.exit('invalid correlation type {}'.format(corrtype))
            df = 10
            # ORIGINAL, which does not match paper description 
            # stds = np.ones((D,1))
            # stds[::2] = np.sqrt((df-2)/df)
            # locs = np.linspace(0,D/scale,D).reshape(-1,1)
            # K = pairwise_kernels(locs, metric='rbf')
            # cov = stds * K * stds.T
            # X = np.random.multivariate_normal(np.zeros(D), cov, N)
            # rescale = np.sqrt(np.random.chisquare(df, (N,1))/(df-2))
            # X[:,::2] = X[:,::2] / rescale
            # CORRECTED, which does match paper description 
            locs = np.linspace(0,D/scale,D,endpoint=False).reshape(-1,1)
            cov = pairwise_kernels(locs, metric='rbf')
            rescale = np.ones((N,D))
            rescale[:,::2] = 1/np.sqrt(rng.chisquare(df, (N,1))/(df-2))
            covs = np.einsum('ij,ki,kj->kij',cov,rescale,rescale)
            mvn = np.vectorize(lambda cov: rng.multivariate_normal(np.zeros(cov.shape[0]), cov, method='cholesky'), signature='(n,n)->(n)')
            X = mvn(covs)
        elif corrtype == 'uncorr':
            X = rng.normal(size=(N, D))
        else:
            sys.exit('invalid correlation type {}'.format(corrtype))
        beta0 = get_beta(sparsity, D)
        if regtype == 'nonlinear':
            Xgen = X**3
            beta_opt = 3*beta0
        elif regtype == 'linear':
            Xgen = X
            beta_opt = beta0
        else:
            sys.exit('invalid regression type {}'.format(regtype))
        y = generate_y_for_X_beta(noisetype, Xgen, beta0)
    return X, y, beta_opt



def plot_model_mismatch(figpath, model_mismatch, kind, limited=False):
    figpath += '-' + kind
    if limited:
        figpath += '-short'
    if kind == 'violin':
        kwargs = dict(scale='width')
    else:
        kwargs = dict(jitter=.3)
    Dp1 = model_mismatch.shape[1]
    cols = [r'$\beta_{{{}}}$'.format(i) for i in range(1, Dp1)] + [r'$\log(\sigma^2)$']
    df = pd.DataFrame(model_mismatch, columns=cols)
    df['overall'] = df.max(axis=1, skipna=False) # since we are looking at hat{m} we use max
    mm_str = r'$\hat m$'
    df = df.melt(value_vars=df.columns, var_name='variable', value_name=mm_str)
    df['bad_point'] = df[mm_str].isna()
    df.loc[df['bad_point'], mm_str] = 0
    if limited:
        col_order = ['overall'] + cols[:-1:3] + cols[-1:]
    else:
        col_order = ['overall'] + cols
    aspect = max(1, len(col_order) / 8 + .7)
    g = sns.catplot(kind=kind, data=df, x='variable', y=mm_str, hue='bad_point',
                order=col_order, aspect=aspect, hue_order=[False, True],
                palette=['black', 'red'], legend=False, **kwargs)
    g.set_axis_labels('')
    xlim = plt.xlim()
    ys = np.linspace(0,1,100)
    diff = ys[1] - ys[0]
    for y in ys:
        rect = patches.Rectangle((xlim[0],y),xlim[1]-xlim[0],diff,
                                 edgecolor='none',facecolor='black',
                                 alpha=.5*np.abs(y))
        plt.gca().add_patch(rect)
    min_x_start = len(cols) - .42
    # rect = patches.Rectangle((min_x_start,-.99),0.9,2.01,
    #                           edgecolor='gray', facecolor='none', linewidth=3)
    # plt.gca().add_patch(rect)
    plt.plot([.5, .5], [-1,1], ':', color='gray')
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim([-1,1.03])
    plt.ylabel(r'$\hat m$')
    sns.despine()
    plt.savefig(figpath+'-model-mismatch.pdf', bbox_inches='tight')



def plot_model_mismatch_hist(figpath, model_mismatch, limited=False, ncols=1):
    if limited is not False:
        figpath += '-short'
    Dp1 = model_mismatch.shape[1]
    cols = [r'$\beta_{{{}}}$'.format(i) for i in range(1, Dp1)] + [r'$\log(\sigma^2)$']
    df = pd.DataFrame(model_mismatch, columns=cols)
    df['overall'] = df.max(axis=1, skipna=False) # since we are looking at hat{m} we use max
    mm_str = r'$\hat m$'
    df = df.melt(value_vars=df.columns, var_name='variable', value_name=mm_str)
    df['bad_point'] = df[mm_str].isna()
    max_x = 1.4
    df.loc[df['bad_point'], mm_str] = max_x - .05
    if limited is not False:
        if isinstance(limited, int):
            incr = (Dp1 - 1)/limited
            beta_cols = [cols[int(incr*k)] for k in range(limited)]
        else:
            beta_cols = cols[:-1:8]
        row_order = ['overall'] + beta_cols + cols[-1:]
    else:
        row_order = ['overall'] + cols
    def plot_zero_line(*args, **kwargs):
        ax = plt.gca()
        ax.plot([0,0], ax.get_ylim() ,':k', linewidth=3)
    g = sns.FacetGrid(df, col='variable', col_order=row_order, col_wrap=ncols,
                      hue='bad_point', aspect=2.5, palette=sns.color_palette()[:4:3],
                      sharex=False, sharey=False)
    bins = np.linspace(-1,max_x,int(10*max_x+11))
    g.map(plt.hist, mm_str, bins=bins).set(xlim=(-1,max_x))
    g.map_dataframe(plot_zero_line)
    g.facet_axis(0,0).set_facecolor('xkcd:light grey')
    regular_ticks = np.linspace(-1,1,5)
    g.set(xticks=np.concatenate([regular_ticks, [max_x - .05]]))
    g.set_xticklabels(['{:.2f}'.format(v) for v in regular_ticks] + ['NA'])
    g.set_titles('{col_name}')
    # g.set_axis_labels(y_var=cols)
    g.set_axis_labels('mismatch index')
    sns.despine()
    plt.savefig(figpath+'-model-mismatch-hist.pdf') #, bbox_inches='tight')
