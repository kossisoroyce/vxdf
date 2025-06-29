.. VXDF documentation master file, created by Kossiso.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VXDF's documentation!
================================

Vector Xchange Data Format (VXDF) makes it easy to store, compress, and
retrieve large collections of vector-indexed text.

.. image:: _static/vxdf_logo.svg
   :width: 280px
   :alt: VXDF logo
   :align: center

Installation
------------

.. code-block:: bash

   pip install vxdf

Quick-start
-----------

Convert a PDF to VXDF via CLI:

.. code-block:: bash

   python -m vxdf convert my.pdf my.vxdf

Or from Python:

.. code-block:: python

   from vxdf.ingest import convert
   convert("my.pdf", "my.vxdf")

.. note::
   If this is your first time using an OpenAI model, the CLI will prompt you
   for an API key and store it in ``~/.vxdf/config.toml``.

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   reference/modules

Guides & Examples
-----------------

.. toctree::
   :maxdepth: 1

   guides/quick_start
   guides/langchain_integration

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
