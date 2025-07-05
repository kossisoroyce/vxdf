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

From Python:

.. code-block:: python

    from vxdf import VXDFWriter, VXDFReader

    # Create a small file
    data = [
        {"id": "1", "text": "hello", "vector": [0.1, 0.2]},
        {"id": "2", "text": "world", "vector": [0.3, 0.4]},
    ]
    with VXDFWriter("demo.vxdf", embedding_dim=2) as w:
        for chunk in data:
            w.add_chunk(chunk)

    # Read it back
    reader = VXDFReader("demo.vxdf")
    print(reader.get_chunk("2"))


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
   guides/cli_usage
   guides/langchain_integration

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
