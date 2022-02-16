.. _installation:

============
Installation
============

Most code is written in python, however timecritical parts such as descriptor
and prior-mean evaluations are implemented using cython (and cymem), and need
be compiled for your particular setup.

Requirements
------------

* Python (only tested with 3.6.3)
* ASE (tested with 3.17 and newer)
* Cython (tested with 0.28 and newer)
* cymem (tested with 1.31.2 and newer)
* mpi4py (tested with 3.0 and newer)
* GPAW

Install from source
-------------------

The code is avaliable as a tar-file :download:`gofee.tar.gz`.

After downloading the tar-file, unpack it using::

    tar -zxvf gofee.tar.gz

Then run the build_code file inside the gofee-folder, to compile descriptor
and prior-function, both used in the surrogate model. Do this using::

    ./build_code

This will compile the mentioned files for the python setup used
at the time of compiling.

Finally when using the code, you need to have the gofee-folder in
the PYTHONPATH. This is achieved using::

    export PYTHONPATH=<path-to-folder>/gofee:$PYTHONPATH

When this is done, and assuming you have a working GPAW installation, you
can run python scripts calling GOFFE using::

    mpiexec --mca mpi_warn_on_fork 0 gpaw-python script_calling_GOFEE.py

