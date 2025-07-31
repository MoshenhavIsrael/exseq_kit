# ExSeq Kit

A modular toolkit for spatial transcriptomics analysis using ExSeq (Expansion Sequencing) data.

This package provides core Python classes and functions to handle ExSeq-based spatial transcriptomics samples, including object representation, spatial analysis, and visualization.

## Structure

- **objects/**: Core classes such as `SampleObject` and `ProjectObject` to handle sample data and project-level organization.
- **analysis/**: Spatial analysis functions such as Moran’s I, cell type enrichment, and radial distribution.
- **visualization/**: Tools to visualize spatial gene expression, cell types, and spatial metrics.
- **io/**: Utilities to load, convert, and export data from various formats (CSV, JSON, etc.).

## Getting Started

```python
from exseq_kit.objects.sample_object import SampleObject
from exseq_kit.objects.project_object import ProjectObject
```

This toolkit assumes ExSeq data in tabular formats (e.g., `RNA_with_cells.csv`, `gene_cell_matrix.csv`, etc.) structured by sample folders.

## Requirements

- Python 3.7+
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- PySAL (for spatial statistics)

## License

This project is for internal research use.
