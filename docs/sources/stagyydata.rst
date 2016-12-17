The StagyyData class
====================

The StagyyData class is a generic lazy accessor to StagYY output data you can
use in your own Python scripts. This section assumes the StagyyData instance
is called ``sdat``. You can create this instance like this::

    from stagpy import stagyydata
    sdat = stagyydata.StagyyData('path/to/run/')

where ``path/to/run/`` is the path towards the directory containing your run
(where the ``par`` file is). This path can be absolute or relative to the
current working directory.

Snapshots and time steps
------------------------

A StagYY run is a succession of time steps with information such as the mean
temperature of the domain outputted at each time step. Now and then, radial
profiles and complete fields are saved, constituting a snapshot.

A StagyyData instance has two attributes to access time steps and snapshots
in a consistent way: ``sdat.steps`` and ``sdat.snaps``. Accessing the ``n``-th
time step or the ``m-th`` snapshot is done using square brackets::

    sdat.steps[n]
    sdat.snaps[m]

These two expressions each return a :class:`_Step` instance. Moreover, if the
``m``-th snapshot was done at the ``n``-th step, both expressions return the
same :class:`_Step` instance. In other words, if for example the 100th snapshot
was made at the 1000th step, ``sdat.steps[1000] is sdat.snaps[100]`` is true.
The correspondence between time steps and snapshots is deduced from available
binary files.

Negative indices are allowed, ``sdat.steps[-1]`` being the last time step
(inferred from temporal series information) and ``sdat.snaps[-1]`` being the
last snapshot (inferred from available binary files).

Finally, ``steps`` and ``snaps`` accessors accept slices. ``sdat.steps[a:b:c]``
returns the generator ``(sdat.steps[n] for n in range(a, b, c))`` (negative
indices are also properly taken care of). This is useful to iterate through
time steps or snapshots. For example, the following code process every even
snapshot::

    for step in sdat.snaps[::2]:
        do_something(step)

.. class:: _Step

   Representation of time step and snapshot used by StagyyData. This class
   shouldn't be instantiated by the user. A :class:`_Step` instance is returned
   by access to ``steps`` and ``snaps`` items (e.g. ``step = sdat.steps[10]``).

   .. attribute:: sdat

      Reference towards the StagyyData instance holding the step object. It
      means ``sdat.steps[n].sdat is sdat`` evaluates to ``True`` for any valid
      step index ``n``. This is useful when you write a function to process
      a time step that need other information available in the StagyyData
      instance. You only need to feed a :class:`_Step` instance to such a
      function since you can access the parent StagyyData instance through
      this attribute.

   .. attribute:: istep

      Time step index. The relation ``sdat.steps[n].istep == n`` is always true.

   .. attribute:: isnap

      Snapshot index. The relation ``sdat.snaps[n].isnap == n`` is always true.

   .. attribute:: irsnap

      Radial profile index. ``sdat.rprof[step.irsnap]`` and ``step.rprof`` are
      two views of the same data. Equal to ``None`` if no radial profile
      exists for this time step.

   .. attribute:: itsnap

      Time series index. ``sdat.tseries[step.itsnap]`` and ``step.timeinfo``
      are two views of the same data. Equal to ``None`` if no temporal
      information exists for this time step.

   .. attribute:: rprof

      Radial profile data of the time step. Equal to ``None`` if no radial
      profile exists for this time step.

   .. attribute:: timeinfo

      Temporal data of the time step. Equal to ``None`` if no temporal data
      exists for this time step.

   .. attribute:: geom

      Geometry information as read from a binary file holding field
      information. Equal to ``None`` if no binary file exists for this time
      step.

   .. attribute:: fields

      Scalar and vector fields available at this time step.


Parameters file
---------------

Parameters set in the ``par`` file are accessible through the ``par`` attribute
of a StagyyData instance. ``sdat.par`` is organized as a dictionary of
dictionaries.  For example, to access the Rayleigh number from the ``refstate``
section of the par file, one can use ``sdat.par['refstate']['ra0']``. Parameters
that are not set in the par file are given a default value according to the par
file ``~/.config/stagpy/par``.

Radial profiles
---------------

Radial profile data are contained in the ``rprof`` attribute of a StagyyData
instance. This attribute is a three dimensional array, with indices in the
following order: temporal snapshot, variable (such as temperature or grid
position), radial index.

The radial profile of a given time step can be accessed from
:attr:`_Step.rprof` (e.g. ``sdat.steps[1000].rprof``).

Time series
-----------

Temporal data are contained in the ``tseries`` attribute of a StagyyData
instance. This attribute is a two dimensional array, with indices in the
following order: temporal snapshot, variable (such as mean temperature or
advective time).

The temporal data of a given time step can be accessed from
:attr:`_Step.timeinfo` (e.g. ``sdat.steps[1000].timeinfo``).

Geometry
--------

Geometry information are read from fields files. :attr:`_Step.geom` has
various attributes defining the geometry of the problem.

``cartesian``, ``curvilinear``, ``cylindrical``, ``spherical`` and ``yinyang``
booleans define the shape of the domain (``curvilinear`` being the opposite of
``cartesian``, ``True`` if ``cylindrical`` or ``spherical`` is ``True``).

``nxtot``, ``nytot``, ``nztot``, ``nbtot``, ``nttot``, ``nptot`` and ``nrtot``
are the total number of points in the various spatial directions. Note that
``nttot``, ``nptot`` and ``nrtot`` are the same as ``nxtot``, ``nytot`` and
``nztot`` regardless of whether the geometry is cartesian or curvilinear.

``x_coord``, ``y_coord`` and ``z_coord`` as well as ``t_coord``, ``p_coord``
and ``r_coord`` are the coordinates of cell centers in the threee directions.
As for the total number of points, they are the same regardless of the actual
geometry.

``x_mesh``, ``y_mesh`` and ``z_mesh`` are three dimensional meshes containing
the **cartesian** coordinates of cell centers (even if the geometry is
curvilinear).

``t_mesh``, ``p_mesh`` and ``r_mesh`` are three dimensional meshes containing
the **spherical** coordinates of cell centers (these are set as ``None`` if the
geometry is cartesian).

Scalar and vector fields
------------------------

Vector and scalar fields are accessible through the ``fields`` attribute of a
StagyyData instance, using their name as key. For example, the temperature
field of the 100th snapshot is obtained with ``sdat.snaps[100].fields['t']``.
Valid names of fields can be obtained by running ``% stagpy var``. Fields are
four dimensional arrays, with indices in the order x, y, z and block.
