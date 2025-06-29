Quick-start Guide
=================

This short guide walks you through converting a document to VXDF, inspecting
it, and querying it from Python.

1. Install VXDF
---------------

.. code-block:: bash

   pip install vxdf

2. Convert a PDF via the CLI
----------------------------

.. code-block:: bash

   python -m vxdf convert sample.pdf sample.vxdf

If this is the first time you use an OpenAI embedding model, the CLI will prompt
for an API key and store it in ``~/.vxdf/config.toml``.

3. Inspect the file
-------------------

.. code-block:: bash

   python -m vxdf info sample.vxdf

4. Query from Python
--------------------

.. code-block:: python

   from vxdf.reader import VXDFReader

   with VXDFReader("sample.vxdf") as reader:
       print(reader.header)
       chunk = reader.get_chunk("doc_0")
       print(chunk["text"])

That’s it! See the :doc:`../reference/modules` for the full API.
