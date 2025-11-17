import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from exseq_kit import SampleObject  # Assuming SampleObject is defined in sample_object.py in the same package

custom_palette = {
    'sick': 'red',
    'control': 'green'
}

class ProjectObject:
    def __init__(self, main_folder, puncta_file_format='RNA_with_cells.csv', auto_calc=False, condition_dict=None, sick_keywords='STXBP1', control_keywords='CTR', xy_scaling=0.17, z_scaling=0.4):
        self.path = main_folder
        self.samples = {}  # sample_name: SampleObject
        self.auto_calc = auto_calc
        self.groups = condition_dict or {}
        self.sick_keywords = sick_keywords
        self.control_keywords = control_keywords
        self.conditions = set()
        self.xy_scaling = xy_scaling
        self.z_scaling = z_scaling
        self._load_samples(main_folder, puncta_file_format)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def _infer_condition(self, folder_name):
        name = folder_name.upper()
        if self.sick_keywords in name:
            return 'sick'
        elif self.control_keywords in name:
            return 'control'
        return 'unknown'

    def compute_sample_info(self):
        records = []
        for name, sample in self.samples.items():
            num_cells = len(sample.cells)
            num_punctas = len(sample.RNA_loc_df)
            x_range = sample.cells['X_centroid'].max() - sample.cells['X_centroid'].min()
            y_range = sample.cells['Y_centroid'].max() - sample.cells['Y_centroid'].min()
            z_range = sample.cells['Z_centroid'].max() - sample.cells['Z_centroid'].min()

            num_genes = sample.RNA_loc_df['gene'].nunique() if 'gene' in sample.RNA_loc_df.columns else 0
            types = sample.cells['cell_type'].unique() if 'cell_type' in sample.cells.columns else []

            records.append({
                'sample': name,
                'condition': self.groups.get(name, 'unknown'),
                'num_cells': num_cells,
                'num_punctas': num_punctas,
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range,
                'num_genes': num_genes,
                'cell_types': types
            })
        self.sample_info = pd.DataFrame.from_records(records)
    def _load_samples(self, main_folder, puncta_file_format):
        for item in os.listdir(main_folder):
            item_path = os.path.join(main_folder, item)
            if os.path.isdir(item_path) and puncta_file_format in os.listdir(item_path):
                sample = SampleObject.from_files(item_path, puncta_file_format, auto_calc=self.auto_calc, xy_scaling=self.xy_scaling, z_scaling=self.z_scaling)
                if sample.sample_name not in self.groups:
                    self.groups[sample.sample_name] = self._infer_condition(item)
                self.conditions.add(self.groups[sample.sample_name])
                self.samples[sample.sample_name] = sample

    # def _load_shapes_from_tif(self, tif_file_format='FOV1FOV_1_round001_ch00.tif.cells.tif', tif_xy_scaling=None, tif_z_scaling=None):
    #     """
    #     Load shapes from TIF files in the project directory.
    #     """
    #     for sample_name, sample in self.samples.items():
    #         tif_path = os.path.join(sample.sample_folder, tif_file_format)
    #         if os.path.exists(tif_path):
    #             sample.load_shapes_from_tif(tif_path, tif_xy_scaling=tif_xy_scaling, tif_z_scaling=tif_z_scaling)
    #             sample.update_cells_features()
    def collect_all_cells(self):
        """
        Return a combined DataFrame of all cells in the project with sample and condition labels.
        """
        all_cells = []
        for sample_name, sample in self.samples.items():
            if hasattr(sample, 'cells'):
                df = sample.cells.copy()
                df['sample'] = sample_name
                df['condition'] = self.groups.get(sample_name, 'unknown')
                all_cells.append(df)
        if not all_cells:
            return pd.DataFrame()
        return pd.concat(all_cells, ignore_index=True)

    def analyze_cell_type_population(self, mode='relative', n_permutations=1000, x_label_size=10, y_label_size=12, figsize=(12, 4), save_path=None):
        all_types = sorted({ct for s in self.samples.values() for ct in s.cells['cell_type'].unique()})

        data = []
        sample_names, group_labels = [], []

        for name, sample in self.samples.items():
            counts = sample.cells['cell_type'].value_counts()
            total = len(sample.cells)
            row = [(counts.get(ct, 0) / total if mode == 'relative' else counts.get(ct, 0)) for ct in all_types]
            data.append(row)
            sample_names.append(name)
            group_labels.append(self.groups[name])

        df = pd.DataFrame(data, columns=all_types)
        df['sample'] = sample_names
        df['group'] = group_labels

        results = []
        for ct in all_types:
            group1 = df[df['group'] == 'control'][ct]
            group2 = df[df['group'] == 'sick'][ct]
            mw_stat, mw_p = mannwhitneyu(group1, group2, alternative='two-sided')

            # Permutation test
            observed_diff = abs(group1.mean() - group2.mean())
            combined = np.array(df[ct])
            labels = np.array(df['group'])
            greater_count = 0
            for _ in range(n_permutations):
                np.random.shuffle(labels)
                g1 = combined[labels == 'control']
                g2 = combined[labels == 'sick']
                if abs(g1.mean() - g2.mean()) >= observed_diff:
                    greater_count += 1
            perm_p = greater_count / n_permutations

            results.append({
                'cell_type': ct,
                'mannwhitney_p': mw_p,
                'perm_p': perm_p,
                'control_mean': group1.mean(),
                'sick_mean': group2.mean()
            })

        result_df = pd.DataFrame(results)

        # Plot 1: Boxplot
        from matplotlib.colors import to_rgba

        df_melt = df.melt(id_vars=['sample', 'group'], var_name='cell_type', value_name='count')
        ylabel = 'Proportion' if mode == 'relative' else 'Count'
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.boxplot(data=df_melt, x='cell_type', y='count', hue='group',
                    palette=custom_palette, hue_order=['control', 'sick'], ax=ax1)
        for patch in ax1.patches:
            color = to_rgba(patch.get_facecolor(), 0.6)
            patch.set_facecolor(color)
        ax1.set_ylabel(ylabel, fontsize=y_label_size)
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', labelsize=x_label_size)
        fig1.tight_layout()
        handles, _ = ax1.get_legend_handles_labels()
        ax1.legend(handles, ['Healthy', 'STXBP1'], loc='upper left')

        # Plot 3: Stacked Bar
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        df_bar = df.drop(columns=['sample', 'group'])
        control_samples = df[df['group'] == 'control']['sample'].tolist()
        sick_samples = df[df['group'] == 'sick']['sample'].tolist()
        sample_order = control_samples + sick_samples
        sample_labels = {name: f"{self.control_keywords}_{i+1}" for i, name in enumerate(control_samples)}
        sample_labels.update({name: f"{self.sick_keywords}_{i+1}" for i, name in enumerate(sick_samples)})
        df_bar.index = [sample_labels[name] for name in df['sample']]
        df_bar = df_bar.loc[[sample_labels[name] for name in sample_order]]
        # df_bar[df_bar.columns] = df_bar[df_bar.columns].div(df_bar.sum(axis=1), axis=0)
        df_bar.plot(kind='bar', stacked=True, colormap='tab20', ax=ax3)
        # ax3.set_title('Stacked Barplot of Cell Type Composition per Sample')
        ax3.set_ylabel(ylabel, fontsize=y_label_size)
        # ax3.tick_params(axis='x', rotation=45, ha='right')
        ax3.set_xticks(range(len(sample_order)), df_bar.index, rotation=45, ha='right')
        fig3.tight_layout()

        if save_path is not None:
            fig1.savefig(save_path + '/Celltype_'+ylabel+'_population_by_conditions', dpi=300, bbox_inches="tight")
            fig3.savefig(save_path + '/Celltype_'+ylabel+'_population_by_samples', dpi=300, bbox_inches="tight")

        return df, result_df, (fig1, fig3)

    def analyze_cell_distribution_along_axis(self, axis='Z', bin_size=5, num_bins=50, normalize_counts=False, normalize_axis=False, window_size=5, bandwidth=1.0, show_histogram=True, show_line=True, n_points=100, radial=None):
        """
        Analyze the distribution of cells along a given axis for each sample.

        Returns:
            control_data (dict): sample_name -> DataFrame of smoothed signals (one column per cell type)
            sick_data (dict): sample_name -> DataFrame of smoothed signals (one column per cell type)
        """
        from scipy.ndimage import gaussian_filter1d

        assert axis in ['X', 'Y', 'Z'], "Axis must be one of 'X', 'Y', or 'Z'"
        control_data = {}
        sick_data = {}
        for sample_name, sample in sorted(self.samples.items(), key=lambda x: (0 if self.groups.get(x[0]) == 'control' else 1, x[0])):
            merged = sample.cells.copy()
            if radial in ['spherical', 'cylindrical']:
                x = merged['X_centroid']
                y = merged['Y_centroid']
                z = merged['Z_centroid']
                x0, y0, z0 = x.mean(), y.mean(), z.mean()
                if radial == 'spherical':
                    axis_values = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
                else:  # cylindrical
                    axis_values = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                axis_label = 'Radial Distance'
                axis_column = 'radial_distance'
                merged[axis_column] = axis_values

            elif radial in ['spherical_volume', 'cylindrical_area']:
                x = merged['X_centroid']
                y = merged['Y_centroid']
                z = merged['Z_centroid']
                x0, y0, z0 = x.mean(), y.mean(), z.mean()
                if radial == 'spherical_volume':
                    axis_values = (4/3) * np.pi * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) ** (3/2)
                    axis_column = 'radial_value'
                else: # cylindrical_area
                    axis_values = np.pi * ((x - x0) ** 2 + (y - y0) ** 2)
                    axis_column = 'radial_area'
                axis_label = 'Radial Volume' if radial == 'spherical_volume' else 'Radial Area'
                merged[axis_column] = axis_values
            else:
                axis_column = f"{axis}_centroid"
                axis_values = merged[axis_column]
                axis_label = f"{axis} axis"

            if normalize_axis:
                min_val, max_val = axis_values.min(), axis_values.max()
                norm_axis_values = (axis_values - min_val) / (max_val - min_val)
            else:
                norm_axis_values = axis_values
                min_val, max_val = merged[axis_column].min(), merged[axis_column].max()
                merged[axis_column] = (merged[axis_column] - min_val) / (max_val - min_val)

            cell_types = merged['cell_type'].unique()
            if normalize_axis:
                bins = num_bins
            else:
                bins = np.arange(merged[axis_column].min(), merged[axis_column].max() + bin_size, bin_size)

            if show_histogram:
                fig_hist, axs = plt.subplots(len(cell_types), 1, figsize=(10, 3 * len(cell_types)), sharex=True)
                if len(cell_types) == 1:
                    axs = [axs]
                for i, ct in enumerate(sorted(cell_types)):
                    subset = merged[merged['cell_type'] == ct][axis_column]
                    counts, bins = np.histogram(subset, bins=bins)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])

                    if normalize_counts:
                        counts = counts / counts.sum() if counts.sum() > 0 else counts
                    smoothed = pd.Series(counts).rolling(window=window_size, center=True, min_periods=1).mean()
                    axs[i].bar(bin_centers, smoothed, width=bin_size * 0.8)
                    axs[i].set_title(f"{sample_name} - {ct}")
                axs[-1].set_xlabel(axis_label)
                fig_hist.tight_layout()
                fig_hist.show()

            # calculate counts signals and smooth them just for vizualization
            sample_signals = {}
            smoothed = {}
            for ct in sorted(cell_types):
                subset = merged[merged['cell_type'] == ct][axis_column]
                counts, bins = np.histogram(subset, bins=bins)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                if normalize_counts:
                    counts = counts / counts.sum() if counts.sum() > 0 else counts
                sample_signals[ct] = counts
                smoothed[ct] = gaussian_filter1d(counts, sigma=bandwidth, mode='nearest')
                if normalize_axis:
                    bin_min, bin_max = bin_centers.min(), bin_centers.max()
                    new_axis = np.linspace(0, 1, n_points)
                    norm_bin_centers = (bin_centers - bin_min) / (bin_max - bin_min)
                    smoothed[ct] = np.interp(new_axis, norm_bin_centers, smoothed[ct])
                else:
                    new_axis = bin_centers

            # store the original data (not smoothed) for further analysis
            if self.groups.get(sample_name) == 'control':
                index = norm_bin_centers if normalize_axis else bin_centers
                control_data[sample_name] = pd.DataFrame(sample_signals, index=index)
            elif self.groups.get(sample_name) == 'sick':
                index = norm_bin_centers if normalize_axis else bin_centers
                sick_data[sample_name] = pd.DataFrame(sample_signals, index=index)

            if show_line:
                fig_line = plt.figure(figsize=(10, 5))
                for ct in sorted(cell_types):
                    if ct not in smoothed:
                        continue
                    else:
                        plt.plot(new_axis, smoothed[ct], label=ct)
                plt.xlabel(axis_label)
                plt.ylabel("Proportion" if normalize_counts else "Count")
                plt.title(f"{sample_name} - Cell Type Distribution along {axis_label}")
                plt.legend()
                plt.tight_layout()
                fig_line.show()

        return control_data, sick_data



