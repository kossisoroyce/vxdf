.. _cli_usage:

Command-Line Interface (CLI)
============================

VXDF comes with a powerful command-line interface (CLI) for creating, inspecting, and managing VXDF files. You can invoke it using `vxdf` or `python -m vxdf`.


Core Commands
-------------

convert
~~~~~~~

The `convert` command is the most common entry point. It converts various file formats (like PDF, DOCX, CSV, and plain text) into a VXDF file, automatically generating vector embeddings.

.. code-block:: bash

   # Convert a PDF to VXDF using an OpenAI model
   vxdf convert my_document.pdf my_document.vxdf --model openai

   # Convert a directory of text files recursively
   vxdf convert path/to/text_files/ my_docs.vxdf -r

Piping with `stdin` and `stdout`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use `-` as the input or output path to pipe data from `stdin` or to `stdout`. This is useful for integrating VXDF into shell pipelines.

.. code-block:: bash

   # Create a VXDF file from a string and write it to stdout
   echo "This is my text" | vxdf convert - - > my.vxdf


pack
~~~~~

The `pack` command is for more advanced use cases where you have pre-computed embeddings. It packs newline-delimited JSON (JSONL) into a VXDF file. Each JSON object must contain `id`, `text`, and `vector` keys.

.. code-block:: bash

   vxdf pack my_embeddings.jsonl my_data.vxdf --embedding-dim 384


info
~~~~~

The `info` command displays the header information and metadata for a VXDF file.

.. code-block:: bash

   vxdf info my_data.vxdf


list
~~~~~

The `list` command shows the document IDs of all the chunks stored in the file.

.. code-block:: bash

   vxdf list my_data.vxdf


get
~~~

The `get` command retrieves and prints a specific chunk by its document ID.

.. code-block:: bash

   vxdf get my_data.vxdf my-document-id-123


Shell Completion
----------------

For a better user experience, you can enable shell completion for `bash`, `zsh`, or `fish`.

.. code-block:: bash

   pip install vxdf[completion]
   activate-global-python-argcomplete --user

After restarting your shell, you can use the TAB key to auto-complete commands and options.
