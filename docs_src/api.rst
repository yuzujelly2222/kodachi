API reference
=============

.. automodule:: kodachi
   :no-members:

Index
-----

.. autoclass:: kodachi.Index
   :members: build_index, search, save, load
   :special-members: __init__, __len__, __repr__

Constants
---------

.. py:data:: kodachi.FORMAT_VERSION
   :type: int
   :value: 1

   Version of the ``.kodachi`` file format written by :meth:`Index.save`.
   :meth:`Index.load` raises :class:`ValueError` for files written in any
   other version.
