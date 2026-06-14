=========================================
API reference
=========================================

The ``dscat`` Python package. The command-line interface is described in the
:doc:`CLI guide <../guides/cli>`; this page documents the importable API that backs
it. Each entry links to a page generated from the source docstrings.

.. currentmodule:: dscat

Data model
=========================================

The dataclasses that flow from the adapters into the index, and the configuration
that drives them.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   model.FeatureRow
   model.TableRow
   config.DatasetConfig
   config.Version

Catalogue index
=========================================

The SQLite catalogue and its typed read and write helpers. The tables it builds are
described in :doc:`schema`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   index.Catalogue

Ingestion
=========================================

Discovering versions, parsing dictionaries, and building the index.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ingest.run_ingest
   ingest.IngestSummary
   adapters.parse
   config.discover_versions
   config.load_configs
   config.version_sort_key

Queries
=========================================

Read queries over the catalogue, scoped to the latest version of each dataset by
default.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   queries.list_tables
   queries.describe
   queries.search
   queries.expand_query
   queries.find_feature
   queries.feature_sources
   queries.list_documents
   queries.find_documents
   queries.latest_version_map

Diffing
=========================================

Comparing a dataset's dictionary across versions.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff.diff_versions
   diff.DiffResult

Documentation files
=========================================

Discovering and converting a version's non-dictionary documents.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   docs.discover_docs
   docs.Engine
   docs.resolve_engine
   docs.convert_doc
   docs.extract_sections
   docs.cache_path

Dictionary input
=========================================

Reading Excel data dictionaries.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dictionary.sheet_names
   dictionary.read_sheet

Output rendering
=========================================

Rendering command results as a table, CSV, TSV, JSON, or Markdown.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   output.Format
   output.render

Paths and synonyms
=========================================

Locating the repository and its catalogue, and loading query synonyms.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   paths.find_repo_root
   paths.data_root
   paths.catalogue_dir
   paths.index_path
   paths.docs_cache_dir
   synonyms.load_synonyms
