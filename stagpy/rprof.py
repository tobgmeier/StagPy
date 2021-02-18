"""Plot radial profiles."""
import matplotlib.pyplot as plt

from . import conf, misc
from .stagyydata import StagyyData


def plot_rprofs(rprofs, lovs):
    """Plot requested radial profiles.

    Args:
        rprofs (:class:`~stagpy._step._Rprofs`): a radial profile viewer
            such as :attr:`Step.rprofs` and :attr:`_StepsView.rprofs_averaged`.
        lovs (nested list of str): nested list of profile names such as the one
            produced by :func:`stagpy.misc.list_of_vars`.
    """
    try:
        stepstr = rprofs.steps.stepstr
    except AttributeError:
        stepstr = str(rprofs.step.istep)
    sdat = rprofs.step.sdat

    for vfig in lovs:
        fig, axes = plt.subplots(ncols=len(vfig), sharey=True,
                                 figsize=(4 * len(vfig), 6))
        axes = [axes] if len(vfig) == 1 else axes
        fname = 'rprof_'
        for iplt, vplt in enumerate(vfig):
            xlabel = None
            profs_on_plt = (rprofs[rvar] for rvar in vplt)
            fname += '_'.join(vplt) + '_'
            for ivar, (rprof, rad, meta) in enumerate(profs_on_plt):
                if conf.rprof.depth:
                    rad = sdat.scale(rprofs.bounds[1], 'm')[0] - rad
                axes[iplt].plot(rprof, rad,
                                conf.rprof.style,
                                label=meta.description)
                if xlabel is None:
                    xlabel = meta.kind
                elif xlabel != meta.kind:
                    xlabel = ''
            if ivar == 0:
                xlabel = meta.description
            if xlabel:
                _, unit = sdat.scale(1, meta.dim)
                xlabel += f' ({unit})' if unit else ''
                axes[iplt].set_xlabel(xlabel)
            if vplt[0][:3] == 'eta':  # list of log variables
                axes[iplt].set_xscale('log')
            axes[iplt].set_xlim(left=conf.plot.vmin, right=conf.plot.vmax)
            if ivar:
                axes[iplt].legend()
        if conf.rprof.depth:
            axes[0].invert_yaxis()
        ylabel = 'Depth' if conf.rprof.depth else 'Radius'
        _, unit = sdat.scale(1, 'm')
        ylabel += f' ({unit})' if unit else ''
        axes[0].set_ylabel(ylabel)
        misc.saveplot(fig, fname + stepstr)


def plot_grid(step):
    """Plot cell position and thickness.

    The figure is call grid_N.pdf where N is replace by the step index.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    """
    drad, rad, _ = step.rprofs['dr']
    _, unit = step.sdat.scale(1, 'm')
    if unit:
        unit = f' ({unit})'
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(rad, '-ko')
    ax1.set_ylabel('$r$' + unit)
    ax2.plot(drad, '-ko')
    ax2.set_ylabel('$dr$' + unit)
    ax2.set_xlim([-0.5, len(rad) - 0.5])
    ax2.set_xlabel('Cell number')
    misc.saveplot(fig, 'grid', step.istep)


def cmd():
    """Implementation of rprof subcommand.

    Other Parameters:
        conf.rprof
        conf.core
    """
    sdat = StagyyData()

    if conf.rprof.grid:
        for step in sdat.walk.filter(rprofs=True):
            plot_grid(step)

    lovs = misc.list_of_vars(conf.rprof.plot)
    if not lovs:
        return

    if conf.rprof.average:
        plot_rprofs(sdat.walk.rprofs_averaged, lovs)
    else:
        for step in sdat.walk.filter(rprofs=True):
            plot_rprofs(step.rprofs, lovs)
