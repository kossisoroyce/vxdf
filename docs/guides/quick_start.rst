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

   python -m vxdf convert sample.pdf sample.vxdf --provenance "sample.pdf"

If this is the first time you use an OpenAI embedding model, the CLI will prompt
for an API key and store it in ``~/.vxdf/config.toml``.

Creating a VXDF file
---------------------

To create a VXDF file, you use the `VXDFWriter`. You need to specify the output path and the dimension of your vector embeddings.

.. code-block:: python

    from vxdf import VXDFWriter

    # Define some data to write
    data = [
        {"id": "doc_1", "text": "This is the first document.", "vector": [0.1, 0.2, 0.3]},
        {"id": "doc_2", "text": "This is the second document.", "vector": [0.4, 0.5, 0.6]},
    ]

    # Create the VXDF file and write the data
    with VXDFWriter("my_first_vxdf.vxdf", embedding_dim=3) as writer:
        for chunk in data:
            writer.add_chunk(chunk)

    print("Successfully created my_first_vxdf.vxdf")

Reading from a VXDF file
-------------------------

To read data, you use the `VXDFReader`. You can retrieve individual chunks by their ID.

.. code-block:: python

    from vxdf import VXDFReader

    # Open the VXDF file for reading
    reader = VXDFReader("my_first_vxdf.vxdf")

    # Get a specific chunk by its ID
    chunk = reader.get_chunk("doc_2")

    print(chunk)
    # Expected output:
    # {'id': 'doc_2', 'text': 'This is the second document.', 'vector': [0.4, 0.5, 0.6]}

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
       # New metadata fields available since VXDF 0.2:
       print("Summary:", chunk.get("summary"))
       print("Provenance:", chunk.get("provenance"))

That’s it! See the :doc:`../reference/modules` for the full API.
