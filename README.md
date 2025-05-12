# MSNger

**MSNger** is a modular pipeline for preprocessing MRIs and constructing Morphometric Similarity Networks (MSNs). It supports flexible configurations including preprocessing options, feature selection, similarity measures, and the choice of atlas or template.

This tool integrates three separate Docker-based tools into a single interface. While it's not a fully idealized Python package, version `0.1.0` prioritizes modularity and usability. Each stage of the workflow relies on its respective containerized tool, culminating in MSN construction via the `MSNForge` module in this repository.

> ⚠️ **Note**: As MSNger depends on external tools maintained separately, be sure to monitor their updates when upgrading this pipeline. Future work includes wrapping all components into a unified Python library.

Quick usage examples will be provided below.

For questions, contact `jor115@pitt.edu` or anyone at the Pediatric Imaging Research Center, Children's Hospital of Pittsburgh.
