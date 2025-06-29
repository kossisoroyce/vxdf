LangChain Integration Guide
===========================

This guide shows how to load a VXDF file in LangChain so you can query it with
LLM-powered chains (e.g. Retrieval-QA).

Prerequisites
-------------

.. code-block:: bash

   pip install vxdf langchain openai

You should also have an **OpenAI API key** available via the ``OPENAI_API_KEY``
environment variable *or* saved in ``~/.vxdf/config.toml`` (the CLI prompt we
saw earlier).

Example: build a Retrieval-QA chain
-----------------------------------

.. code-block:: python

   from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
   from langchain.vectorstores import FAISS
   from langchain.chains import RetrievalQA
   from langchain.chat_models import ChatOpenAI

   from vxdf.reader import VXDFReader

   # 1. Read the VXDF file and extract documents & embeddings
   docs, vectors = [], []
   with VXDFReader("my.vxdf") as r:
       for doc_id, meta in r.header["docs"].items():
           chunk = r.get_chunk(doc_id)
           docs.append(chunk["text"])
           vectors.append(chunk["vector"])

   # 2. Turn the stored embeddings into a FAISS index
   index = FAISS.from_embeddings(vectors, docs)

   # 3. Build the retrieval chain
   retriever = index.as_retriever(search_kwargs={"k": 5})
   llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
   chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

   # 4. Ask a question
   result = chain.run("What is the billing address on invoice FB8D355A-0009?")
   print(result)

Using a local embedding model
-----------------------------

If your VXDF was created with the default local model (``all-MiniLM-L6-v2``),
replace ``OpenAIEmbeddings`` with ``HuggingFaceEmbeddings``:

.. code-block:: python

   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

That’s it—LangChain sees VXDF as just another vector store.

Further reading
---------------

* :doc:`quick_start` – basic CLI & Python usage.
* :ref:`modindex` – complete API reference.
