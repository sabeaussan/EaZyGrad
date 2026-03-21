eazygrad
================

Top-level API
-------------

.. automodule:: eazygrad
   :no-index:

Tensor Object
-------------

.. toctree::
   :maxdepth: 1

   eazygrad.tensor_object

Tensor Creation
---------------

.. currentmodule:: eazygrad

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tensor
   from_numpy
   randn
   uniform
   ones
   zeros

Optimization
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SGD
   Adam
   AdamW

Autograd Control
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   no_grad
   no_grad_ctx

.. toctree::
   :hidden:

   eazygrad.tensor_object
   generated/eazygrad.tensor
   generated/eazygrad.from_numpy
   generated/eazygrad.randn
   generated/eazygrad.uniform
   generated/eazygrad.ones
   generated/eazygrad.zeros
   generated/eazygrad.SGD
   generated/eazygrad.Adam
   generated/eazygrad.AdamW
   generated/eazygrad.no_grad
   generated/eazygrad.no_grad_ctx
