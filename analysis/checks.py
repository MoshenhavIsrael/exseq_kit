import os
import sys
import pandas as pd
import numpy as np
from __init__ import ProjectObject

# -----------------------
# Parameters
# -----------------------
NUM_PERMUTATIONS = 9999
WEIGHT_FUNC = "fixed_k"
POWER = 2
K = 8
SIGMA = 10.0
DROP_LABELS = {"No_type", "type_nan", "0"}

# -----------------------
# Project init
# -----------------------
folder = r"C:\Users\Moshe\OneDrive\coding\SpatialGenomicsLab\SegmentedData\Organoids\filtered_organoids"
rna_file_format = "RNA_with_cells_fixed.csv"
xy_scaling = 1 / 3.3
z_scaling = 1 / 3.3

project = ProjectObject(
    main_folder=folder,
    puncta_file_format=rna_file_format,
    xy_scaling=xy_scaling,
    z_scaling=z_scaling
)

project.compute_sample_info()
print(project.sample_info)

# -----------------------
# Step 1: Compute Moran’s I + Join Counts for cell types
# -----------------------
all_results = []

for samp_name, samp in project.samples.items():
    print(f"Calculating Moran’s I and Join Counts for {samp_name}...")
    df = samp.calculate_spatial_stats_for_celltypes(
        weight_func=WEIGHT_FUNC,
        power=POWER,
        k=K,
        sigma=SIGMA,
        include_join_counts=True,
        permutations=NUM_PERMUTATIONS,
        save_results=False,         # ← don't save to file
        drop_labels=DROP_LABELS
    )
    df["sample"] = samp_name
    all_results.append(df)

# merge all samples
celltype_spatial_stats = pd.concat(all_results, ignore_index=True)
# celltype_spatial_stats.to_csv(os.path.join(folder, "celltype_spatial_stats_all_samples.csv"), index=False)

# create comparison (wide) tables
comparison_MoranI = celltype_spatial_stats.pivot_table(index="cell_type", columns="sample", values="Moran_I")
comparison_MoranP = celltype_spatial_stats.pivot_table(index="cell_type", columns="sample", values="Moran_p")
comparison_JCp = celltype_spatial_stats.pivot_table(index="cell_type", columns="sample", values="JC_BB_p")
comparison_JCpsim_chi2 = celltype_spatial_stats.pivot_table(index="cell_type", columns="sample", values="JC_p_sim_chi2")
comparison_JCp_chi2 = celltype_spatial_stats.pivot_table(index="cell_type", columns="sample", values="JC_chi2_p")

# print summary
print("\n=== Combined results (long table) ===")
print(celltype_spatial_stats.head(20))

print("\n=== Moran's I comparison ===")
print(comparison_MoranI)

print("\n=== Join Counts p-value comparison ===")
print(comparison_JCp)
