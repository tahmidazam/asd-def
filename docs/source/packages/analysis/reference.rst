=========================================
API reference
=========================================

The importable ``analysis`` API. The package is built stage by stage, so this reference
grows as each pipeline stage is added. It currently covers the command-line interface, the
run and caching infrastructure, the cohort abstraction, feature typing, the mixture-model
wrapper, the enrichment and alignment used to recover and name the reference classes, the
model selection, stability, and cross-cohort replication that test the recovered solution,
and the stratification axes, binning policies, and acceptance requirements that the
stratified analysis is judged against before its bins are frozen.

Command-line interface
=========================================

.. automodule:: analysis.cli
   :members:

Configuration and paths
=========================================

.. automodule:: analysis.config
   :members:

.. automodule:: analysis.paths
   :members:

Runs and caching
=========================================

.. automodule:: analysis.cache
   :members:

.. automodule:: analysis.run
   :members:

.. automodule:: analysis.checkpoint
   :members:

.. automodule:: analysis.progress
   :members:

.. automodule:: analysis.profiling
   :members:

Cohort abstraction
=========================================

.. automodule:: analysis.cohort
   :members:

.. automodule:: analysis.cohort.schema
   :members:

.. automodule:: analysis.cohort.spark
   :members:

.. automodule:: analysis.cohort.ssc
   :members:

Feature typing and the model
=========================================

.. automodule:: analysis.features
   :members:

.. automodule:: analysis.model
   :members:

Enrichment and alignment
=========================================

.. automodule:: analysis.enrich
   :members:

.. automodule:: analysis.align
   :members:

.. automodule:: analysis.reference
   :members:

Selection, stability, and replication
=========================================

.. automodule:: analysis.selection
   :members:

.. automodule:: analysis.stability
   :members:

.. automodule:: analysis.replicate
   :members:

Stratification and pre-registration
=========================================

.. automodule:: analysis.strata
   :members:

.. automodule:: analysis.strata_data
   :members:

.. automodule:: analysis.requirements
   :members:
