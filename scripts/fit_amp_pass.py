#! /usr/bin/env python
"""
A script for fitting the amplitudes and passbands of antennas given
starting parameters in "a.loc" and a list of sources (from "a.src").

Author: Aaron Parsons
Date: 3/07/08
Revisions:
"""

import aipy as a, numpy as n, sys, os, optparse

o = optparse.OptionParser()
o.set_usage('fit_amp_pass.py [options] *.uv')
o.set_description(__doc__)
o.add_option('-c', '--cat', dest='cat',
    help='A list of several sources (separated by commas) to use.')
o.add_option('-l', '--loc', dest='loc', 
    help='Use location-specific info for this location.')
o.add_option('-x', '--decimate', dest='decimate', default=1, type='int',
    help='Only use every Nth time step in fitting calculations.')
o.add_option('-p', '--prms', dest='prms', 
    help='Comma delimited list of paramters to fit independently for each antenna (amp,passband,bm_xwidth,bm_ywidth).')
o.add_option('-a', '--ants', dest='ants', default='*',
    help='Comma delimited list of antennas to fit. Default fits amp/passband for all antennas.')
o.add_option('-s', '--shared_prms', dest='shared_prms', 
    help='Comma delimited list of paramters to fit shared for all antennas (amp,passband,bm_xwidth,bm_ywidth).')
o.add_option('-o', '--other_prms', dest='other_prms', 
    help='Source=param pairs')
o.add_option('--nchan', dest='nchan', default=0, type='int',
    help='Number of fit points along a bandpass (w/ interpolation inbetween).')
o.add_option('--baselines', dest='baselines', default='all',
    help='Select which antennas/baselines to include in plot.  Options are: "all", "auto", "cross", "<ant1 #>_<ant2 #>" (a specific baseline), or "<ant1 #>,..." (a list of active antennas).')
o.add_option('--chan', dest='chan', default='all',
    help='Select which channels (taken after any delay/fringe transforms) to plot.  Options are: "all", "<chan1 #>,..." (a list of active channels), or "<chan1 #>_<chan2 #>" (a range of channels).  If "all" or a range are selected, a 2-d image will be plotted.  If a list of channels is selected, an xy plot will be generated.')

def convert_arg_range(arg):
    """Split apart command-line lists/ranges into a list of numbers."""
    arg = arg.split(',')
    return [map(float, option.split('_')) for option in arg]

def gen_chans(chanopt, uv):
    """Return an array of active channels and whether or not a range of
    channels is selected (as opposed to one or more individual channels)
    based on command-line arguments."""
    is_chan_range = True
    if chanopt == 'all': chans = n.arange(uv['nchan'])
    else:
        chanopt = convert_arg_range(chanopt)
        def conv(c): return c
        chanopt = [map(conv, c) for c in chanopt]
        if len(chanopt[0]) != 1:
            chanopt = [n.arange(x,y, dtype=n.int) for x,y in chanopt]
        chans = n.concatenate(chanopt)
    return chans.astype(n.int)

opts, args = o.parse_args(sys.argv[1:])

srcs = opts.cat.split(',')
cat = a.src.get_catalog(srcs, type='fit')

uv = a.miriad.UV(args[0])
chans = gen_chans(opts.chan, uv)
aa = a.loc.get_aa(opts.loc, uv['sdf'], uv['sfreq'], uv['nchan'])
aa.select_chans(chans)
(uvw,t,(i,j)),d = uv.read()
aa.set_jultime(t)
cat.compute(aa)
if opts.ants == '*': ants = range(uv['nants'])
else: ants = eval('['+opts.ants+']')
del(uv)

if opts.prms is None: prms = []
else: prms = opts.prms.split(',')
if opts.shared_prms is None: sprms = []
else: sprms = opts.shared_prms.split(',')
prm_dict = {}
for ant in ants: prm_dict[ant] = prms[:]
prm_dict[ants[0]].extend(sprms)
print prm_dict
start_prms = aa.get_params(prm_dict, nchan=opts.nchan)
if opts.other_prms != None:
    src_prms = [w.split('=') for w in opts.other_prms.split(',')]
    sprm_dict = {}
    for src,prm in src_prms:
        if sprm_dict.has_key(src): sprm_dict.append(prm)
        else: sprm_dict[src] = [prm]
    start_prms.update(cat.get_params(sprm_dict))

prm_list, key_list = a.fit.flatten_prms(start_prms)
first_fit = None    # Used to normalize fit values to the starting fit

baselines = None
if not opts.baselines is 'all':
    baselines = [map(int, w.split('_')) for w in opts.baselines.split(',')]

def fit_func(prms):
    global first_fit
    prms = a.fit.reconstruct_prms(prms, key_list)
    a.fit.print_params(prms)
    for ant in ants:
        for prm in sprms:
            prms[ant][prm] = prms[ants[0]][prm]
    aa.set_params(prms)
    cat.set_params(prms)
    s_eqs,fluxes,indices,mfreqs = cat.get_vecs()
    score,cnt,curtime = 0,0,None
    for uvfile in args:
        sys.stdout.write('.'), ; sys.stdout.flush()
        uv = a.miriad.UV(uvfile)
        if not baselines is None:
            for (i,j) in baselines: uv.select('antennae',i,j, include=True)
        else: uv.select('auto', 0, 0, include=False)
        for (uvw,t,(i,j)),d in uv.all():
            # Use only every Nth integration, if decimation is specified.
            if curtime != t:
                curtime = t
                cnt = (cnt + 1) % opts.decimate
                if cnt == 0:
                    aa.set_jultime(t)
                    cat.compute(aa)
            if cnt != 0: continue
            sim_d = aa.sim(i, j, s_eqs, fluxes, indices=indices, mfreqs=mfreqs,
                pol=a.miriad.pol2str[uv['pol']])
            d = d.take(chans)
            difsq = n.abs(d - sim_d)**2
            score += difsq.filled(0).sum()
    print
    if first_fit is None: first_fit = score
    print 'Score:', score / first_fit, '(Base: %f)' % first_fit
    print '-------------------------------------------------------------------'
    return score / first_fit

a.optimize.fmin(fit_func, prm_list)
'''
def anneal(func, x0, T_x, cooling=lambda i: 3*(n.cos(i/50.)+1), 
        maxiter=1000, verbose=True):
    score = func(x0)
    for i in range(maxiter):
        if verbose: print 'Step:', i, 'T fraction:', cooling(i)
        delta = n.random.normal(scale=1., size=x0.shape) * cooling(i) * T_x
        n_x = x0 + delta
        n_score = func(n_x)
        if n_score < score: x0, score = n_x, n_score
        if verbose:
            print 'Best score:', score
            print 'Best x:', x0
    return x0, score

prm_list = n.array(prm_list)
x0, score = anneal(fit_func, prm_list, .0001*prm_list)
print x0
print 'Score:', score
print 'Done!'
'''
