"""Plot scalar and vector fields."""

from __future__ import annotations

import typing
from itertools import chain

import matplotlib as mpl
import matplotlib.patches as mpat

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import _helpers, conf, phyvars
from .error import NotAvailableError
from .stagyydata import StagyyData
import f90nml
from cmcrameri import cm


from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colorbar
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import ListedColormap, Normalize

import copy

my_path = os.path.abspath(os.path.dirname(__file__))
cm_data = np.loadtxt(my_path+"/batlow.txt")
vik_map = LinearSegmentedColormap.from_list('vik', cm_data)



if typing.TYPE_CHECKING:
    from typing import Any, Dict, Iterable, Optional, Tuple, Union

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from numpy import ndarray

    from ._step import Step
    from .datatypes import Varf


# The location is off for vertical velocities: they have an extra
# point in (x,y) instead of z in the output


def _threed_extract(
    step: Step, var: str, walls: bool = False
) -> Tuple[Tuple[ndarray, ndarray], ndarray]:
    """Return suitable slices and coords for 3D fields."""
    is_vector = not valid_field_var(var)
    hwalls = is_vector or walls
    i_x: Optional[Union[int, slice]] = conf.field.ix
    i_y: Optional[Union[int, slice]] = conf.field.iy
    i_z: Optional[Union[int, slice]] = conf.field.iz
    if i_x is not None or i_y is not None:
        i_z = None
    if i_x is not None or i_z is not None:
        i_y = None
    if i_x is None and i_y is None and i_z is None:
        i_x = 0
    if i_x is not None:
        xcoord = step.geom.y_walls if hwalls else step.geom.y_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        i_y = i_z = slice(None)
        varx, vary = var + "2", var + "3"
    elif i_y is not None:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        i_x = i_z = slice(None)
        varx, vary = var + "1", var + "3"
    else:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.y_walls if hwalls else step.geom.y_centers
        i_x = i_y = slice(None)
        varx, vary = var + "1", var + "2"
    data: Any
    if is_vector:
        data = (
            step.fields[varx].values[i_x, i_y, i_z, 0],
            step.fields[vary].values[i_x, i_y, i_z, 0],
        )
    else:
        data = step.fields[var].values[i_x, i_y, i_z, 0]
    return (xcoord, ycoord), data


def valid_field_var(var: str) -> bool:
    """Whether a field variable is defined.

    Args:
        var: the variable name to be checked.
    Returns:
        whether the var is defined in :data:`~stagpy.phyvars.FIELD` or
        :data:`~stagpy.phyvars.FIELD_EXTRA`.
    """
    return var in phyvars.FIELD or var in phyvars.FIELD_EXTRA


def get_meshes_fld(
    step: Step, var: str, walls: bool = False
) -> Tuple[ndarray, ndarray, ndarray, Varf]:
    """Return scalar field along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: scalar field name.
        walls: consider the walls as the relevant mesh.
    Returns:
        tuple (xmesh, ymesh, fld, meta).  2D arrays containing respectively the
        x position, y position, the values and the metadata of the requested
        field.
    """
    #t1 = time.time()
    fld = step.fields[var]
    hwalls = (
        walls
        or fld.values.shape[0] != step.geom.nxtot
        or fld.values.shape[1] != step.geom.nytot
    )
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), vals = _threed_extract(step, var, walls)
    elif step.geom.twod_xz:
        xcoord = step.geom.x_walls if hwalls else step.geom.x_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        vals = fld.values[:, 0, :, 0]
    else:  # twod_yz
        xcoord = step.geom.y_walls if hwalls else step.geom.y_centers
        ycoord = step.geom.z_walls if walls else step.geom.z_centers
        if step.geom.curvilinear:
            pmesh, rmesh = np.meshgrid(xcoord, ycoord, indexing="ij")
            xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
        vals = fld.values[0, :, :, 0]
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xmesh, ymesh, vals, fld.meta


def get_meshes_vec(step: Step, var: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Return vector field components along with coordinates meshes.

    Only works properly in 2D geometry and 3D cartesian.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: vector field name.
    Returns:
        tuple (xmesh, ymesh, fldx, fldy).  2D arrays containing respectively
        the x position, y position, x component and y component of the
        requested vector field.
    """
    if step.geom.threed and step.geom.cartesian:
        (xcoord, ycoord), (vec1, vec2) = _threed_extract(step, var)
    elif step.geom.twod_xz:
        xcoord, ycoord = step.geom.x_walls, step.geom.z_centers
        vec1 = step.fields[var + "1"].values[:, 0, :, 0]
        vec2 = step.fields[var + "3"].values[:, 0, :, 0]
    elif step.geom.cartesian and step.geom.twod_yz:
        xcoord, ycoord = step.geom.y_walls, step.geom.z_centers
        vec1 = step.fields[var + "2"].values[0, :, :, 0]
        vec2 = step.fields[var + "3"].values[0, :, :, 0]
    else:  # spherical yz
        pcoord = step.geom.p_walls
        pmesh = np.outer(pcoord, np.ones(step.geom.nrtot))
        vec_phi = step.fields[var + "2"].values[0, :, :, 0]
        vec_r = step.fields[var + "3"].values[0, :, :, 0]
        vec1 = vec_r * np.cos(pmesh) - vec_phi * np.sin(pmesh)
        vec2 = vec_phi * np.cos(pmesh) + vec_r * np.sin(pmesh)
        pcoord, rcoord = step.geom.p_walls, step.geom.r_centers
        pmesh, rmesh = np.meshgrid(pcoord, rcoord, indexing="ij")
        xmesh, ymesh = rmesh * np.cos(pmesh), rmesh * np.sin(pmesh)
    if step.geom.cartesian:
        xmesh, ymesh = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xmesh, ymesh, vec1, vec2

def get_meshes_vec_spherical(step, var):
    """Return vector field components along with coordinates meshes.

    Only works properly in 2D spherical

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
        var (str): vector field name.
    Returns:
        tuple of :class:`numpy.array`: rmesh, pmesh, fldr, fldp
            2D arrays containing respectively the x position, y position, x
            component and y component of the requested vector field.
    """

    xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]
    pmesh = step.geom.p_mesh[0, :, :]
    rmesh = step.geom.r_mesh[0,:,:]
    vec_phi = step.fields[var + '2'][0, :, :, 0]
    vec_r = step.fields[var + '3'][0, :, :, 0]
    vec1 = vec_r * np.cos(pmesh) - vec_phi * np.sin(pmesh)
    vec2 = vec_phi * np.cos(pmesh) + vec_r * np.sin(pmesh)
    return xmesh, ymesh, vec1, vec2


def set_of_vars(arg_plot):
    """Build set of needed field variables.

    Each var is a tuple, first component is a scalar field, second component is
    either:

    - a scalar field, isocontours are added to the plot.
    - a vector field (e.g. 'v' for the (v1,v2,v3) vector), arrows are added to
      the plot.

    Args:
        arg_plot (str): string with variable names separated with
            ``,`` (figures), and ``+`` (same plot).
    Returns:
        set of str: set of needed field variables.
    """
    sovs = set(tuple((var + '+').split('+')[:2])
               for var in arg_plot.split(','))
    sovs.discard(('', ''))
    return sovs


def plot_scalar(step: Step,
    var: str,
    field: Optional[ndarray] = None,
    axis: Optional[Axes] = None,
    print_time = None, 
    print_substellar = False,
    draw_circle = False, 
    text_size = 9 ,
    paper_label = None,
    cbar_remove = False, 
    cbar_invisible = False,
    invisible_alpha = 1.0,
    more_info=False,
    op_melt = False, 
    text_color = 'black', **extra:Any,)-> Tuple[Figure, Axes, QuadMesh, Colorbar]:

    """Plot scalar field.

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of step.fields[var].  This is
            useful to plot a masked or rescaled array.  Note that if
            conf.scaling.dimensional is True, this field will be scaled
            accordingly.
        axis: the :class:`matplotlib.axes.Axes` object where the field should
            be plotted.  If set to None, a new figure with one subplot is
            created.
        extra: options that will be passed on to
            :func:`matplotlib.axes.Axes.pcolormesh`.
    Returns:
        fig, axis, surf, cbar
            handles to various :mod:`matplotlib` objects, respectively the
            figure, the axis, the surface returned by
            :func:`~matplotlib.axes.Axes.pcolormesh`, and the colorbar returned
            by :func:`matplotlib.pyplot.colorbar`.
    """

    #conf.field.interpolate = False #TGM: add this to make sure not to interpolate

    if step.geom.threed and step.geom.spherical:
        raise NotAvailableError("plot_scalar not implemented for 3D spherical geometry")

    xmesh, ymesh, fld, meta = get_meshes_fld(
        step, var, walls= not conf.field.interpolate
    )
    # interpolate at cell centers, this should be abstracted by field objects
    # via an "at_cell_centers" method or similar


    if fld.shape[0] > max(step.geom.nxtot, step.geom.nytot):
        fld = (fld[:-1] + fld[1:]) / 2


    
    if field is not None:
        fld = field
    if conf.field.perturbation:
        fld = fld - np.mean(fld, axis=0)
    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)

    if conf.field.interpolate and step.geom.spherical and step.geom.twod_yz:
        # add one point to close spherical annulus
        xmesh = np.concatenate((xmesh, xmesh[:1]), axis=0)
        ymesh = np.concatenate((ymesh, ymesh[:1]), axis=0)
        newline = (fld[:1] + fld[-1:]) / 2
        fld = np.concatenate((fld, newline), axis=0)
    xmin, xmax = xmesh.min(), xmesh.max()
    ymin, ymax = ymesh.min(), ymesh.max()

    fld, unit = step.sdat.scale(fld, meta.dim)

    if op_melt:
        _, _, mf_fld, mf_meta = get_meshes_fld(
            step, 'meltfrac', walls= not conf.field.interpolate
        )
        if conf.field.interpolate and step.geom.spherical and step.geom.twod_yz:
            newline = (mf_fld[:1] + mf_fld[-1:]) / 2
            mf_fld = np.concatenate((mf_fld, newline), axis=0)
        mf_fld, mf_unit = step.sdat.scale(mf_fld, mf_meta.dim)



    if axis is None:
        fig, axis = plt.subplots(ncols=1)
    else:
        fig = axis.get_figure()

    if step.sdat.par["magma_oceans_in"]["evolving_magma_oceans"]:
        rcmb = step.sdat.par["geometry"]["r_cmb"]
        xmax = rcmb + 1
        ymax = xmax
        xmin = -xmax
        ymin = -ymax
        rsurf = xmax if step.timeinfo["thick_tmo"] > 0 else step.geom.r_walls[-3]
        cmb = mpat.Circle((0, 0), rcmb, color="dimgray", zorder=0)
        psurf = mpat.Circle((0, 0), rsurf, color="indianred", zorder=0)
        axis.add_patch(psurf)
        axis.add_patch(cmb)

    # Define the mapping of 'var' to 'shading' values
    shading_map = {
        'T': 'gouraud',
        'eta': 'gouraud',
        'bs': None,
        'hz': None,
        # Add more mappings as needed
        # 'another_var': 'another_shading',
    }
    # Get the shading value based on 'var'
    # Set shading based on the value of var
    shading_var = shading_map.get(var, 'gouraud')

    # Define the mapping of 'var' to 'cmap' values
    shading_map = {
        'T': cm.batlow,
        'eta': cm.batlow,
        'bs': (cm.bam).reversed(),
        'hz': (cm.bam).reversed(),
        'meltfrac' : cm.managua.reversed(),
        # Add more mappings as needed
        # 'another_var': 'another_shading',
    }
    # Get the shading value based on 'var'
    # Set shading based on the value of var
    cmap_var = shading_map.get(var, cm.batlow)

    extra_opts = dict(
        #cmap=conf.field.cmap.get(var),
        cmap = cmap_var, 
        vmin=conf.plot.vmin,
        vmax=conf.plot.vmax,
        norm=mpl.colors.LogNorm() if var == "eta" else None,
        rasterized=conf.plot.raster,
        shading=shading_var,
    )
    extra_opts.update(extra)


    surf = axis.pcolormesh(xmesh, ymesh, fld, **extra_opts)

    cbar_adjust = 1.05
    if op_melt == True:
        low_melt = 0.01
        # Create a mask where melt fraction is less than 0.1
        mask = mf_fld < low_melt

        fld_masked = np.ma.masked_where(mask, mf_fld)
        # Set alpha to 0.0 where mask is True
        cmap_var = cm.managua.reversed()
        cmap_with_alpha = cmap_var(np.arange(cmap_var.N))
        cmap_with_alpha[:, -1] = np.where(np.arange(cmap_var.N) < low_melt * cmap_var.N, 0, 1)  # Adjust alpha channel
        custom_cmap = ListedColormap(cmap_with_alpha)
        mf_extra_opts = dict(
            #cmap=conf.field.cmap.get(var),
            cmap = cmap_var, 
            vmin=0,
            vmax=1,
            norm=None,
            rasterized=conf.plot.raster,
            shading='gouraud',
        )
        print(np.max(fld_masked), np.min(fld_masked))
        surf2 = axis.pcolormesh(xmesh, ymesh, fld_masked,**mf_extra_opts)
        cbar_adjust = 1.05



    cbar = None

    conf.field.colorbar=False
    if conf.field.colorbar: #TGM: turned this off by default as it is plotted below (for now), edit July23: try merge this
        print('bool colorbar', conf.field.colorbar)
        cax = make_axes_locatable(axis).append_axes("right", size="3%", pad=0.15)
        cbar = plt.colorbar(surf, cax=cax)
        cbar.set_label(
            meta.description
            + (" pert." if conf.field.perturbation else "")
            + (f" ({unit})" if unit else "")
        )
    if step.geom.spherical or conf.plot.ratio is None:
        axis.set_aspect("equal")
        axis.set_axis_off()
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())

    axis.set_adjustable("box")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)

    divider = make_axes_locatable(axis)
    #cax = divider.append_axes("right", size="5%", pad=+0.05)
    #cbar = plt.colorbar(surf, shrink=conf.field.shrinkcb,orientation="vertical", cax=cax)
    

    cbar_ts = text_size
    cax = divider.append_axes("bottom", size="5%", pad=+0.05)    
    cbar = plt.colorbar(surf,orientation="horizontal",cax=cax)
    cbar.set_label(meta.description +
               (' pert.' if conf.field.perturbation else '') +
               (' ({})'.format(unit) if unit else '') +
               (' (' + meta.dim + ')' if meta.dim != '1' else ' ( )'),color=text_color, size = cbar_ts)
    cbar.ax.tick_params(labelsize=cbar_ts+1, color=text_color)
    cbar.outline.set_edgecolor(text_color)
    #cbar.ax.xaxis.set_tick_params(color=text_color,rotation=270)
    cbar.ax.xaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=text_color)

    if op_melt:
        # Add the second colorbar for the masked data
        cax3 = divider.append_axes("right", size="5%", pad=0.05)  # Increase the pad to place it below the first colorbar
        cbar3 = plt.colorbar(surf2, cax=cax3, orientation="vertical")
        cbar3.set_label('Melt Fraction', size=cbar_ts,color=text_color)
        cbar3.ax.tick_params(labelsize=text_size-2, color=text_color)
        #cbar3.ax.xaxis.set_tick_params(color=text_color)
        cbar3.ax.yaxis.set_tick_params(color=text_color)
        cbar3.outline.set_edgecolor(text_color)
        plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color=text_color)


    cax2 = divider.append_axes("top", size="6%", pad=+0.0)

    cax2.axis('off')

    nml = step.sdat.par

    eta0 = str(int(math.log10(nml['viscosity']['eta0'])))
    Tcmb = str(int(nml['boundaries']['botT_val']))
    Tday = str(int(nml['boundaries']['topT_locked_subsolar']))
    Tnight = str(int(nml['boundaries']['topT_locked_farside']))
    radius  = nml['geometry']['r_cmb']+nml['geometry']['d_dimensional']
    rdim  = nml['geometry']['d_dimensional']
    rcmb = nml['geometry']['r_cmb']
    g_dim = nml['refstate']['g_dimensional']
    rppv = rcmb+(rdim-2740e3*(9.81/g_dim))
    rda = 0.5*(1-rcmb/(rcmb+rdim)) + 0.005 # without 0.03 this is the rdimensional in the axis system.
    if draw_circle == True:
        circle1 = plt.Circle((0, 0), rppv, edgecolor='w', linestyle = '--', fill=False, linewidth=1.0)
        axis.add_artist(circle1)
    if print_substellar == True:
        cax2.axvline(x=0.5,ymin=0,ymax=1.0,linestyle='dashed',color=text_color)
        cax2.text(0.48, 0.5, 'Day', horizontalalignment='right', verticalalignment='center',color=text_color, size = text_size,transform=cax2.transAxes)
        cax2.text(0.52, 0.5, 'Night', horizontalalignment='left', verticalalignment='center',color=text_color, size = text_size, transform=cax2.transAxes)
        bbox_props = dict(boxstyle="rarrow", ec="black", lw=0.5,fc='gold', alpha=invisible_alpha)
        #axis.text(-radius-0.068*radius, 0.0, "STAR", ha="right", va="center",bbox=bbox_props,size = text_size, color='black',fontweight='bold')
        axis.text(-radius-0.105*radius, 0.0, "STAR", ha="right", va="center",bbox=bbox_props,size = text_size, color='black',fontweight='bold',alpha=invisible_alpha)
        axis.text(rda , 0.5, "0$\degree$", ha="left", va="center", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(1-rda+0.005, 0.5, "180$\degree$", ha="right", va="center", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(0.5 , 1-rda, "90$\degree$", ha="center", va="top", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(0.5, rda, "-90$\degree$", ha="center", va="bottom", color=text_color,size=text_size-4,transform=axis.transAxes)
    if print_time != None:
        if paper_label != None:
            cax2.text(1.0, 0.4, '{:.2f}'.format(print_time)+' Gyr',horizontalalignment='right',verticalalignment='center',color=text_color, size = text_size)
            cax2.text(0.0, 0.4, '('+paper_label+')',horizontalalignment='left',verticalalignment='center',color=text_color, size = text_size,fontweight='bold')
            if more_info == True:
                axis.text(0.5, rda+0.055, "(Longitude)", ha="center", va="bottom",color=text_color,size=text_size,transform=axis.transAxes)
                axis.text(0.0, 0.0, '(Fe Core)',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
        #axis.text(0,0,'{:.2e}'.format(print_time)+' Myrs',horizontalalignment='center')
        else:
            #cax2.text(0.5, 1.2, '{:.2f}'.format(print_time)+' Gyr',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
            #cax2.text(0.5, 1.55, '{:.2f}'.format(print_time)+'\u2009Gyr',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
            axis.text(0.5, 0.5, '{:.2f}'.format(print_time)+'$\,$Gyr',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
        #axis.text(0.5,0.5,'$\eta_0=$'+'$10^{%s}$ Pa s' %(eta0)+'\n $T_{CMB}=%s$K \n $T_{day}=%s$K \n $T_{night}=%s$K' %(Tcmb, Tday, Tnight),horizontalalignment='center',verticalalignment='center',size = text_size,transform = axis.transAxes)

    if cbar_remove == True:
        cbar.remove()
        cax.axis('off')
    if cbar_invisible == True:
        cbar.remove()
        cax4 = divider.append_axes("bottom", size="5%", pad=+0.05,frameon=False)
        cax4.set_xlabel('Temperature',alpha=0)
        cax4.set_xticks([])
        cax4.set_yticks([])
        #cax4.axis('off')

    fig.set_size_inches(fig.get_size_inches()[0]*cbar_adjust, fig.get_size_inches()[1]*cbar_adjust)  # Increase the height of the figure
    return fig, axis, surf, cbar


def plot_iso(
    axis: Axes, step: Step, var: str, field: Optional[ndarray] = None, **extra: Any
) -> None:
    """Plot isocontours of scalar field.

    Args:
        axis: the :class:`matplotlib.axes.Axes` of an existing matplotlib
            figure where the isocontours should be plotted.
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the scalar field name.
        field: if not None, it is plotted instead of step.fields[var].  This is
            useful to plot a masked or rescaled array.  Note that if
            conf.scaling.dimensional is True, this field will be scaled
            accordingly.
        extra: options that will be passed on to
            :func:`matplotlib.axes.Axes.contour`.
    """
    xmesh, ymesh, fld, _ = get_meshes_fld(step, var)

    if field is not None:
        fld = field

    if conf.field.shift:
        fld = np.roll(fld, conf.field.shift, axis=0)
    extra_opts: Dict[str, Any] = dict(linewidths=1)
    if "cmap" not in extra and conf.field.isocolors:
        extra_opts["colors"] = conf.field.isocolors
    elif "colors" not in extra:
        extra_opts["cmap"] = conf.field.cmap.get(var)
    if conf.plot.isolines:
        extra_opts["levels"] = sorted(conf.plot.isolines)
    extra_opts.update(extra)
    axis.contour(xmesh, ymesh, fld, levels = [0.0,0.5,0.99999],colors=('black','peru','red'),alpha=0.7,**extra_opts)
    #axis.clabel(CS, inline=1, fontsize=10)



def plot_vec(axis: Axes, step: Step, var: str,arrow_v=0.005, mask_highv = False, q_scale_factor = 0.05, use_cm = False,text_color='black') -> None:
    """Plot vector field.

    Args:
        axis: the :class:`matplotlib.axes.Axes` of an existing matplotlib
            figure where the vector field should be plotted.
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the vector field name.
    """
    xmesh, ymesh, vec1, vec2 = get_meshes_vec(step, var)


    #######TGM: Some options to make vectors smaller/larger because of melt phases##########
    #55cnc
    dipz_factor = 15 
    #LHS?
    dipz = step.geom.nztot // dipz_factor
    sp = 1.5
    if conf.field.shift:
        vec1 = np.roll(vec1, conf.field.shift, axis=0)
        vec2 = np.roll(vec2, conf.field.shift, axis=0)
    if step.geom.spherical or conf.plot.ratio is None:
        dipx = int(0.5*dipz)
    else:
        dipx = step.geom.nytot if step.geom.twod_yz else step.geom.nxtot
        dipx = int(dipx // 10 * conf.plot.ratio) + 1

    vec_norm = np.sqrt(vec1**2.0 + vec2**2.0)
    if mask_highv == False:
        vec1_lowv = vec1
        vec2_lowv = vec2 
    else: 
        highv_factor = 5.0
        vec1_lowv = np.ma.masked_where(vec_norm > 2*np.mean(vec_norm), vec1)
        vec2_lowv = np.ma.masked_where(vec_norm > 2*np.mean(vec_norm), vec2)
        vec1_highv = np.ma.masked_where(vec_norm < 2*np.mean(vec_norm), vec1)
        vec2_highv = np.ma.masked_where(vec_norm < 2*np.mean(vec_norm), vec2)

    scale_v = 1.0
    q_scale = q_scale_factor*1.9028881819080357e-06

    if (use_cm == False):
        cm_scale = 1.0
        v_units = r'${0} \,\frac{{m}}{{yr}}$'
    else:
        cm_scale = 0.01
        v_units =  r'${0} \,\frac{{cm}}{{yr}}$'  

    Q = axis.quiver(xmesh[::dipx, ::dipz], ymesh[::dipx, ::dipz],
                scale_v*vec1_lowv[::dipx, ::dipz], scale_v*vec2_lowv[::dipx, ::dipz], headwidth = 3/sp, headlength = 5/sp, headaxislength = 4.5/sp, width = 0.0025, scale = q_scale, color = 'black', scale_units ='inches')
    qk = axis.quiverkey(Q, 0.7, 0.85, arrow_v*3.171e-8, v_units.format(arrow_v/cm_scale), labelpos='E',
                   coordinates='figure',labelsep=0.01, color = 'black',labelcolor=text_color,fontproperties={'size':10}) 

    if mask_highv == True: 
        Q2 = axis.quiver(xmesh[::dipx, ::dipz], ymesh[::dipx, ::dipz],
                    scale_v*vec1_highv[::dipx, ::dipz], scale_v*vec2_highv[::dipx, ::dipz], headwidth = 3/sp, headlength = 5/sp, headaxislength = 4.5/sp, width = 0.0025, scale = q_scale*highv_factor, color = 'red', scale_units ='inches')
        qk = axis.quiverkey(Q2, 0.7, 0.8, arrow_v*highv_factor*3.171e-8, v_units.format(arrow_v*highv_factor/cm_scale), labelpos='E',
                       coordinates='figure',labelsep=0.01, color = 'red', labelcolor=text_color,fontproperties={'size':10}) 
    ##########################################################################################


def _findminmax(
    sdat: StagyyData, sovs: Iterable[str]
) -> Dict[str, Tuple[float, float]]:
    """Find min and max values of several fields."""
    minmax: Dict[str, Tuple[float, float]] = {}
    for step in sdat.walk.filter(snap=True):
        for var in sovs:
            if var in step.fields:
                field = step.fields[var]
                vals, _ = sdat.scale(field.values, field.meta.dim)
                if var in minmax:
                    minmax[var] = (
                        min(minmax[var][0], np.nanmin(vals)),
                        max(minmax[var][1], np.nanmax(vals)),
                    )
                else:
                    minmax[var] = np.nanmin(vals), np.nanmax(vals)
    return minmax


def cmd() -> None:
    """Implementation of field subcommand.

    Other Parameters:
        conf.field
        conf.core
    """
    sdat = StagyyData()
    # no more than two fields in a subplot
    lovs = [[slov[:2] for slov in plov] for plov in conf.field.plot]
    minmax = {}
    if conf.plot.cminmax:
        conf.plot.vmin = None
        conf.plot.vmax = None
        sovs = set(slov[0] for plov in lovs for slov in plov)
        minmax = _findminmax(sdat, sovs)
    for step in sdat.walk.filter(snap=True):
        for vfig in lovs:
            fig, axes = plt.subplots(
                ncols=len(vfig), squeeze=False, figsize=(6 * len(vfig), 6)
            )
            for axis, var in zip(axes[0], vfig):
                if var[0] not in step.fields:
                    print(f"{var[0]!r} field on snap {step.isnap} not found")
                    continue
                opts: Dict[str, Any] = {}
                if var[0] in minmax:
                    opts = dict(vmin=minmax[var[0]][0], vmax=minmax[var[0]][1])
                plot_scalar(step, var[0], axis=axis, **opts)
                if len(var) == 2:
                    if valid_field_var(var[1]):
                        plot_iso(axis, step, var[1])
                    elif valid_field_var(var[1] + "1"):
                        plot_vec(axis, step, var[1])
            if conf.field.timelabel:
                time, unit = sdat.scale(step.timeinfo["t"], "s")
                time = _helpers.scilabel(time)
                axes[0, 0].text(
                    0.02, 1.02, f"$t={time}$ {unit}", transform=axes[0, 0].transAxes
                )
            oname = "_".join(chain.from_iterable(vfig))
            plt.tight_layout(w_pad=3)
            _helpers.saveplot(fig, oname, step.isnap)



def get_sfield(step, var):
    sfld = step.sfields[var].values[0,:,0]
    return sfld

def get_sfield_pp(step):
    xmesh, ymesh, T_field, meta = get_meshes_fld(step,'T')
    t_s = T_field[:,-1] #surface (top cell)
    t_d = T_field[:,-2] #one below top cell
    _, _, tcond , _ = get_meshes_fld(step,'tcond') 
    tcond_s = tcond[:,-1] #top cell
    print('tcond shape', np.shape(tcond))
    sfld = tcond_s*(t_d-t_s)/38316.535949707031

    return sfld

def plot_scalar_tracers(step: Step,
    var: str,
    axis: Optional[Axes] = None,
    print_time = None, 
    print_substellar = False,
    draw_circle = False, 
    text_size = 9 ,
    paper_label = None,
    cbar_remove = False, 
    cbar_invisible = False,
    colorbar_label = None,
    invisible_alpha=1.0,
    more_info=False,
    text_color = 'black', **extra:Any,)-> Tuple[Figure, Axes, QuadMesh, Colorbar]:

    """Plot scalar field for tracers (technically a coloured scatter plot).

    Args:
        step: a :class:`~stagpy._step.Step` of a StagyyData instance.
        var: the scalar field name (only tracer fields work).
        axis: the :class:`matplotlib.axes.Axes` object where the field should
            be plotted.  If set to None, a new figure with one subplot is
            created.
        extra: options that will be passed on to
            :func:`matplotlib.axes.Axes.scatter`.
    Returns:
        fig, axis, surf, cbar
            handles to various :mod:`matplotlib` objects, respectively the
            figure, the axis, the surface returned by
            :func:`~matplotlib.axes.Axes.pcolormesh`, and the colorbar returned
            by :func:`matplotlib.pyplot.colorbar`.
    """
    if step.geom.threed and step.geom.spherical:
        raise NotAvailableError("plot_scalar not implemented for 3D spherical geometry")

    x_pos = step.tracers['x'][0][::] #TGM: could change ::2 to some variable depending on how many tracers we want to plot
    y_pos = step.tracers['y'][0][::]
    field_tracer = step.tracers[var][0][::]
    print('MINMAX TRACER', np.min(field_tracer), np.max(field_tracer))

    xmin, xmax = x_pos.min(), x_pos.max()
    ymin, ymax = y_pos.min(), y_pos.max()



    if axis is None:
        fig, axis = plt.subplots(ncols=1)
    else:
        fig = axis.get_figure()

    if step.sdat.par["magma_oceans_in"]["evolving_magma_oceans"]:
        rcmb = step.sdat.par["geometry"]["r_cmb"]
        xmax = rcmb + 1
        ymax = xmax
        xmin = -xmax
        ymin = -ymax
        rsurf = xmax if step.timeinfo["thick_tmo"] > 0 else step.geom.r_walls[-3]
        cmb = mpat.Circle((0, 0), rcmb, color="dimgray", zorder=0)
        psurf = mpat.Circle((0, 0), rsurf, color="indianred", zorder=0)
        axis.add_patch(psurf)
        axis.add_patch(cmb)

    extra_opts = dict( 
        vmin=conf.plot.vmin,
        vmax=conf.plot.vmax,
    )
    extra_opts.update(extra)
    if(var == "Water conc." or  var   == "Carbon conc."):
         surf = axis.scatter(x_pos, y_pos, c=field_tracer,s=0.4,linewidths=0 ,alpha=0.5,norm=matplotlib.colors.LogNorm(vmin=0.001, vmax=1000),cmap=cm.batlow,edgecolors=None, **extra_opts)
    else: 
         surf = axis.scatter(x_pos, y_pos, c=field_tracer,s=0.4,alpha=0.5,linewidths=0, cmap=cm.batlow,edgecolors=None, **extra_opts)

    
    if step.geom.spherical or conf.plot.ratio is None:
        axis.set_aspect("equal")
        axis.set_axis_off()
    else:
        axis.set_aspect(conf.plot.ratio / axis.get_data_ratio())

    axis.set_adjustable("box")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)

    divider = make_axes_locatable(axis)
    #cax = divider.append_axes("right", size="5%", pad=+0.05)
    #cbar = plt.colorbar(surf, shrink=conf.field.shrinkcb,orientation="vertical", cax=cax)
    
    cax = divider.append_axes("bottom", size="5%", pad=+0.05)    
    cbar = plt.colorbar(surf,orientation="horizontal",cax=cax)
    cbar.set_label(colorbar_label,color=text_color, size = text_size) #TGM: update this, so we can use the metadata?
    cbar.ax.tick_params(labelsize=text_size+1, color=text_color)
    cbar.outline.set_edgecolor(text_color)
    #cbar.ax.xaxis.set_tick_params(color=text_color,rotation=270)
    cbar.ax.xaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=text_color)


    cax2 = divider.append_axes("top", size="6%", pad=+0.0)

    cax2.axis('off')

    nml = step.sdat.par

    eta0 = str(int(math.log10(nml['viscosity']['eta0'])))
    Tcmb = str(int(nml['boundaries']['botT_val']))
    Tday = str(int(nml['boundaries']['topT_locked_subsolar']))
    Tnight = str(int(nml['boundaries']['topT_locked_farside']))
    radius  = nml['geometry']['r_cmb']+nml['geometry']['d_dimensional']
    rdim  = nml['geometry']['d_dimensional']
    rcmb = nml['geometry']['r_cmb']
    g_dim = nml['refstate']['g_dimensional']
    rppv = rcmb+(rdim-2740e3*(9.81/g_dim))
    rda = 0.5*(1-rcmb/(rcmb+rdim)) + 0.005 # without 0.03 this is the rdimensional in the axis system.
    if draw_circle == True:
        circle1 = plt.Circle((0, 0), rppv, edgecolor='w', linestyle = '--', fill=False, linewidth=1.0)
        axis.add_artist(circle1)
    if print_substellar == True:
        cax2.axvline(x=0.5,ymin=0,ymax=1.0,linestyle='dashed',color=text_color)
        cax2.text(0.48, 0.5, 'Day', horizontalalignment='right', verticalalignment='center',color=text_color, size = text_size,transform=cax2.transAxes)
        cax2.text(0.52, 0.5, 'Night', horizontalalignment='left', verticalalignment='center',color=text_color, size = text_size, transform=cax2.transAxes)
        bbox_props = dict(boxstyle="rarrow", ec="black", lw=0.5,fc='gold',alpha=invisible_alpha)
        axis.text(-radius-0.068*radius, 0.0, "STAR", ha="right", va="center",bbox=bbox_props,size = text_size, color='black',fontweight='bold',alpha=invisible_alpha)
        axis.text(rda , 0.5, "0$\degree$", ha="left", va="center", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(1-rda+0.005, 0.5, "180$\degree$", ha="right", va="center", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(0.5 , 1-rda, "90$\degree$", ha="center", va="top", color=text_color,size=text_size-4,transform=axis.transAxes)
        axis.text(0.5, rda, "-90$\degree$", ha="center", va="bottom", color=text_color,size=text_size-4,transform=axis.transAxes)
    if print_time != None:
        if paper_label != None:
            cax2.text(1.0, 0.4, '{:.2f}'.format(print_time)+' Gyr',horizontalalignment='right',verticalalignment='center',color=text_color, size = text_size)
            cax2.text(0.0, 0.4, '('+paper_label+')',horizontalalignment='left',verticalalignment='center',color=text_color, size = text_size,fontweight='bold')
            if more_info == True:
                axis.text(0.5, rda+0.055, "(Longitude)", ha="center", va="bottom",color=text_color,size=text_size,transform=axis.transAxes)
                axis.text(0.0, 0.0, '(Fe Core)',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
        #axis.text(0,0,'{:.2e}'.format(print_time)+' Myrs',horizontalalignment='center')
        else:
            print('PRINT TIME', print_time)
            #cax2.text(0.5, 1.2, '{:.2f}'.format(print_time)+' Gyr',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
            axis.text(0.5, 0.5, '{:.2f}'.format(print_time)+'$\,$Gyr',horizontalalignment='center',verticalalignment='center',color=text_color, size = text_size)
        #axis.text(0.5,0.5,'$\eta_0=$'+'$10^{%s}$ Pa s' %(eta0)+'\n $T_{CMB}=%s$K \n $T_{day}=%s$K \n $T_{night}=%s$K' %(Tcmb, Tday, Tnight),horizontalalignment='center',verticalalignment='center',size = text_size,transform = axis.transAxes)

    if cbar_remove == True:
        cbar.remove()
        cax.axis('off')
    if cbar_invisible == True:
        cbar.remove()
        cax4 = divider.append_axes("bottom", size="5%", pad=+0.05,frameon=False)
        cax4.set_xlabel('Temperature',alpha=0)
        cax4.set_xticks([])
        cax4.set_yticks([])
        #cax4.axis('off')

    return fig, axis, surf, cbar


