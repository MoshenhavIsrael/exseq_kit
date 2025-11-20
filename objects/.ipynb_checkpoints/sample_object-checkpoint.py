import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from libpysal.weights import full2W
from esda.moran import Moran
from esda.join_counts import Join_Counts
from sklearn.neighbors import NearestNeighbors

class SampleObject:


    def __init__(self, sample_folder=None, RNA_loc_df=None, sample_name=None, cell_centroids=None, cell_polygons=None, distance_matrix=None, gene_cell_matrix=None, auto_calc = False, xy_scaling = 350 / 2048, z_scaling = 0.4):
        """
        Initializes the SampleObject.

        :param sample_folder: path to the folder in which all sample data are stored
        :param RNA_loc_df: DataFrame with all RNA punctas in the sample, showing the location, class, and cell regards for each puncta.
        :param cell_centroids: DataFrame with all cells in the sample, showing their centroids.
        :param cell_polygons: Dictionary containing polygon regards to each cell.
        :param sample_folder: Path to the folder where the data of sample stored.
        :param auto_calc: if True - automatic calculate the distance matrix.
        :param xy_scaling / z_scaling: factor change units from pixels to micro meters.
        """

        # Cache for libpysal W objects (keyed by: N_cells, method, parameters)
        self._W_cache = {}

        # scale coordinates
        RNA_loc_df.rename(columns={'X': 'X_pixels', 'Y': 'Y_pixels', 'Z': 'Z_pixels'}, inplace=True)
        RNA_loc_df['X'] = RNA_loc_df['X_pixels'] * xy_scaling
        RNA_loc_df['Y'] = RNA_loc_df['Y_pixels'] * xy_scaling
        RNA_loc_df['Z'] = RNA_loc_df['Z_pixels'] * z_scaling
        RNA_loc_df.drop(columns=['X_pixels', 'Y_pixels', 'Z_pixels'], inplace=True)

        # Rewrite column names of RNA_loc_df for uniformity
        if 'cellId' in RNA_loc_df.columns:
            RNA_loc_df.rename(columns={'cellId': 'cell_id'}, inplace=True)
        if 'cellType' in RNA_loc_df.columns:
            RNA_loc_df.rename(columns={'cellType': 'cell_type'}, inplace=True)
        if 'cell' in RNA_loc_df.columns and 'cell_id' not in RNA_loc_df.columns:
            RNA_loc_df.rename(columns={'cell': 'cell_id'}, inplace=True)

        # Change type of cell_id and cell_type to string if exist
        if 'cell_id' in RNA_loc_df.columns:
            RNA_loc_df['cell_id'] = RNA_loc_df['cell_id'].astype(str)
        if 'cell_type' in RNA_loc_df.columns:
            RNA_loc_df['cell_type'] = RNA_loc_df['cell_type'].astype(str)

        # Store object parameters
        self.RNA_loc_df = RNA_loc_df
        self.sample_folder = sample_folder
        self.gene_list = RNA_loc_df['gene'].unique()
        if sample_name is None:
            sample_name = sample_folder.split("\\")[-1]
            if sample_name == '':
                sample_name = sample_folder.split("/")[-1]

            # Replace the backslashes with '_' in the sample name
            sample_name = sample_name.replace("\\", "_")
            sample_name = sample_name.replace("/", "_")
            self.sample_name = sample_name
        else:
            self.sample_name = sample_name
        self.cell_polygons = cell_polygons
        self.sample_folder = sample_folder
        self.xy_scaling = xy_scaling
        self.z_scaling = z_scaling
        cells_file_format = "FOV1FOV_1_round001_ch00.tif.cells.tif"
        self.load_cell_shapes_from_tif(cells_file_format, downsample=1, tif_xy_scaling=0.171*xy_scaling, tif_z_scaling=0.4*z_scaling)
        self.update_cells_features(xy_scaling=8*0.171*xy_scaling, z_scaling=8*0.4*z_scaling)
        # # get cell centroids from parameter or calculate it from data
        # if len(np.shape(cell_centroids))>1:
        #     self.cell_centroids = cell_centroids
        # else:
        #     self.cell_centroids = self._create_cell_centroids()
        #
        # # Get cell details if exist
        # if 'cell_id' in RNA_loc_df.columns:
        #     cell_ids = RNA_loc_df['cell_id'].unique()
        #     cell_ids.sort()
        #     cell_ids = [str(cell_id) for cell_id in cell_ids]
        #     if '0' in cell_ids:
        #         cell_ids.remove('0')
        #     if 0 in cell_ids:
        #         cell_ids.remove(0)
        #     if 'NaN' in cell_ids:
        #         cell_ids.remove('NaN')
        #     self.cell_ids = cell_ids
        #
        #     # Add cell types and centroid data and save it with cell_ids in 'cells' dataframe
        #     if 'cell_type' in RNA_loc_df.columns:
        #         cells = RNA_loc_df[RNA_loc_df['cell_id'].isin(cell_ids)][['cell_id', 'cell_type']]
        #     else:
        #         cells = pd.DataFrame(RNA_loc_df[RNA_loc_df['cell_id'].isin(cell_ids)]['cell_id'])
        #     cells = cells.drop_duplicates().reset_index(drop=True)
        #     cells['cell_id'] = cells['cell_id'].astype(str)
        #     # sort 'cells' by 'cell_id'
        #     cells.sort_values(by='cell_id', inplace=True)
        #     if hasattr(self, 'cell_centroids'):
        #         cells = cells.merge(self.cell_centroids, on='cell_id', how='left')
        #     self.cells = cells
        #
        # # get cell types if exist
        # if 'cell_type' in RNA_loc_df.columns:
        #     cell_types = RNA_loc_df['cell_type'].unique().astype(str)
        #     cell_types.sort()
        #     cell_types = [str(cell_type) for cell_type in cell_types]
        #     if 'No_type' in cell_types:
        #         cell_types.remove('No_type')
        #     if 'NaN' in cell_types:
        #         cell_types.remove('NaN')
        #     if '0' in cell_types:
        #         cell_types.remove('0')
        #     if 0 in cell_types:
        #         cell_types.remove(0)
        #     self.cell_types = cell_types

        # calculate gene cell matrix from data
        if auto_calc:
            # calculate and save to file
            self.gene_cell_matrix = self._create_gene_cell_matrix()
        else:
            self.gene_cell_matrix = self._create_gene_cell_matrix(output_file=None)

        # create data frame to store basic info about genes
        if gene_cell_matrix is not None:
            self.gene_info = pd.DataFrame({
                'gene': self.gene_cell_matrix.index,
                'num_cells': self.gene_cell_matrix.astype(bool).sum(axis=1),
                'expression_in_cells': self.gene_cell_matrix.sum(axis=1)
            })
            self.gene_info['gene'] = self.gene_info['gene'].astype(str)
            self.gene_info.set_index('gene', inplace=True)
            # calculate total counts of each gene from the RNA_loc_df. then store it in gene_info aligned by gene
            total_counts = self.RNA_loc_df.groupby('gene')['X'].count()
            self.gene_info = self.gene_info.join(total_counts.rename('total_counts'))
            self.gene_info['external_counts'] = self.gene_info['total_counts'] - self.gene_info['expression_in_cells']

        # get distance matrix from parameter or calculate it from data
        if len(np.shape(distance_matrix)) > 1:
            self.distance_matrix = distance_matrix
            distance_matrix.index = distance_matrix.index.astype(str)
            distance_matrix.columns = distance_matrix.columns.astype(str)

            # --- Clean cell IDs: keep only the numeric part before the first underscore
            distance_matrix.index = [cid.split('_')[0] for cid in distance_matrix.index]
            distance_matrix.columns = [cid.split('_')[0] for cid in distance_matrix.columns]

        elif auto_calc:
            self.calculate_distance_matrix(print_stages=True)

    @classmethod
    def from_files(cls, sample_folder=None, RNA_file=None, centroids_file=None, polygons_file=None, distance_matrix_file=None, gene_cell_matrix_file=None, auto_calc=False, xy_scaling = 350 / 2048, z_scaling = 0.4):
        """
        Creates a SampleObject instance from file inputs, allowing parts to be skipped.

        :param RNA_file: Path to the file containing RNA location data. Can be skipped if None.
        :param centroids_file: Path to the file containing cell centroid data. Can be skipped if None.
        :param polygons_file: Path to the file containing cell polygon data. Can be skipped if None.
        :param sample_folder: Path to the folder where the data of sample stored.

        :return: An instance of SampleObject.
        """
        import pandas as pd
        import json
        import os

        # Load files from sample_folder if provided
        if sample_folder is not None:
            csv_files = [f for f in os.listdir(sample_folder) if f.endswith('.csv')]
            # Assume if sample folder has a single csv file - it is the RNA file
            if len(csv_files) == 1:
                RNA_file = os.path.join(sample_folder, csv_files[0])
                RNA_loc_df = pd.read_csv(RNA_file)
                new_RNA_file_name = os.path.join(sample_folder, "RNA_with_cells.csv")
                # os.rename(RNA_file, new_RNA_file_name)
                return cls(sample_folder, RNA_loc_df, None, None, auto_calc=auto_calc, xy_scaling=xy_scaling, z_scaling=z_scaling)

            elif RNA_file is None or distance_matrix_file is None or gene_cell_matrix_file is None:
                if RNA_file is None and "RNA_with_cells.csv" in csv_files:
                    RNA_file = os.path.join(sample_folder, "RNA_with_cells.csv")
                if gene_cell_matrix_file is None and "gene_cell_matrix.csv" in csv_files:
                    gene_cell_matrix_file = os.path.join(sample_folder, "gene_cell_matrix.csv")
                if distance_matrix_file is None and "distance_matrix.csv" in csv_files:
                    distance_matrix_file = os.path.join(sample_folder, "distance_matrix.csv")

        # If RNA_file, distance_matrix_file, or gene_cell_matrix_file are not provided yet, get them from user input
        if RNA_file is None or distance_matrix_file is None or gene_cell_matrix_file is None:

            if RNA_file is None:
                RNA_file = input("Enter the path for the RNA file (or press Enter to skip): ") or None
            # if centroids_file is None:
            #     centroids_file = input("Enter the path for the centroids file (or press Enter to skip): ") or None
            # if polygons_file is None:
            #     polygons_file = input("Enter the path for the polygons file (or press Enter to skip): ") or None
            if sample_folder is None:
                sample_folder = input("Enter the path for the sample folder: ") or None

        # Load files and create the SampleObject instance
        if not os.path.dirname(RNA_file):
            RNA_file = os.path.join(sample_folder, RNA_file)
        RNA_loc_df = pd.read_csv(RNA_file) if RNA_file else None
        cell_centroids = pd.read_csv(centroids_file) if centroids_file else None
        cell_polygons = json.load(open(polygons_file)) if polygons_file else None
        distance_matrix = pd.read_csv(distance_matrix_file, header=0, index_col=0) if distance_matrix_file else None
        gene_cell_matrix = pd.read_csv(gene_cell_matrix_file, header=0, index_col=0) if gene_cell_matrix_file else None

        return cls(sample_folder=sample_folder, RNA_loc_df=RNA_loc_df, cell_centroids=cell_centroids, cell_polygons=cell_polygons, distance_matrix=distance_matrix, gene_cell_matrix=gene_cell_matrix, auto_calc=auto_calc, xy_scaling=xy_scaling, z_scaling=z_scaling)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def update_cells_features(self, xy_scaling=None, z_scaling=None):
        """
        Constructs or updates self.cells table using TIF (preferred) or RNA data.
        Combines voxel geometry, RNA puncta, centroids, convex hulls, and derived features.
        """
        assert hasattr(self, 'RNA_loc_df'), "RNA_loc_df is required."

        import pandas as pd
        from scipy.spatial import ConvexHull

        # Use provided scaling or default to object attributes
        xy_scaling = xy_scaling if xy_scaling is not None else self.xy_scaling
        z_scaling = z_scaling if z_scaling is not None else self.z_scaling

        # --- Determine base list of cell IDs ---
        if hasattr(self, 'cell_voxels'):
            base_cell_ids = self.cell_voxels['cell_id'].astype(str).unique()
        else:
            base_cell_ids = self.RNA_loc_df['cell_id'].astype(str).unique()

        # --- Initialize cells table ---
        cells = pd.DataFrame({'cell_id': base_cell_ids})

        # --- Add number of punctas ---
        puncta_counts = self.RNA_loc_df['cell_id'].value_counts().rename('num_punctas').astype(int)
        puncta_counts.index = puncta_counts.index.astype(str)
        cells = cells.merge(puncta_counts, left_on='cell_id', right_index=True, how='left')
        cells['num_punctas'] = cells['num_punctas'].fillna(0).astype(int)

        # --- Add puncta-based centroids ---
        centroids = self.RNA_loc_df.groupby('cell_id')[['X', 'Y', 'Z']].mean().reset_index()
        centroids.rename(columns={'X': 'X_centroid', 'Y': 'Y_centroid', 'Z': 'Z_centroid'}, inplace=True)
        centroids['cell_id'] = centroids['cell_id'].astype(str)
        cells = cells.merge(centroids, on='cell_id', how='left')

        # --- Add voxel-based features if available ---
        if hasattr(self, 'cell_voxels'):
            voxel_features = self.cell_voxels.groupby('cell_id')[['X', 'Y', 'Z']].agg(
                cell_volume_voxels=('X', 'count'), # Count of voxels per cell,
                voxel_centroid_x=('X', 'mean'),
                voxel_centroid_y=('Y', 'mean'),
                voxel_centroid_z=('Z', 'mean'),
                Z_min=('Z', 'min'),
                Z_max=('Z', 'max')
            ).reset_index()
            voxel_features['cell_id'] = voxel_features['cell_id'].astype(str)
            voxel_features['Z_span'] = voxel_features['Z_max'] - voxel_features['Z_min']
            # Fix volume calculation
            voxel_features['cell_volume_voxels'] = voxel_features['cell_volume_voxels'] * (xy_scaling ** 2) * z_scaling
            cells = cells.merge(voxel_features.drop(columns=['Z_min', 'Z_max']), on='cell_id', how='left')

        # --- Add convex hull features if available ---
        if hasattr(self, 'cell_hulls') and isinstance(self.cell_hulls, dict):
            convex_data = []
            for cid, hull in self.cell_hulls.items():
                if hull is not None:
                    points = hull[['X', 'Y', 'Z']].values
                    if len(points) >= 4:
                        ch = ConvexHull(points)
                        convex_data.append({
                            'cell_id': cid,
                            'convex_volume': ch.volume,
                            'convex_area': ch.area
                        })
            if convex_data:
                df_convex = pd.DataFrame(convex_data)
                df_convex['cell_id'] = df_convex['cell_id'].astype(str)
                cells = cells.merge(df_convex, on='cell_id', how='left')

        # --- Derived features ---
        if 'cell_volume_voxels' in cells.columns:
            cells['density'] = cells['num_punctas'] / (cells['cell_volume_voxels'] + 1e-3)
        if 'convex_volume' in cells.columns and 'cell_volume_voxels' in cells.columns:
            cells['compactness'] = cells['cell_volume_voxels'] / (cells['convex_volume'] + 1e-3)

        # --- Remove background cell with ID '0' ---
        cells = cells[cells['cell_id'] != '0']

        # --- Add cell_type if exists in RNA_loc_df ---
        if 'cell_type' in self.RNA_loc_df.columns:
            type_counts = self.RNA_loc_df.groupby('cell_id')['cell_type'].nunique()
            ambiguous = type_counts[type_counts > 1].index.astype(str).tolist()
            if ambiguous:
                print(f"Warning: inconsistent cell_type assignments for cell_ids: {ambiguous}")
            cell_types = self.RNA_loc_df[['cell_id', 'cell_type']].drop_duplicates()
            cell_types['cell_id'] = cell_types['cell_id'].astype(str)
            cell_types['cell_type'] = cell_types['cell_type'].astype(str)
            cells = cells.merge(cell_types, on='cell_id', how='left')
            # cells['cell_type'].fillna("type_nan", inplace=True)

        # Store the updated cells DataFrame
        self.cells = cells
        self.cells.cell_type.fillna("type_nan", inplace=True)

    def load_cell_shapes_from_tif(self, file_path, downsample=1, tif_xy_scaling=None, tif_z_scaling=None):
        """
        Load a 3D labeled TIF of cells and extract geometric features per cell.

        Parameters:
        - file_path: path to the TIF file (3D numpy array with cell labels).
        - downsample: factor to reduce spatial resolution for speed.
        - tif_xy_scaling, tif_z_scaling: optional overrides for TIF resolution (in µm per voxel).

        This updates:
        - self.cell_voxels: DataFrame with X,Y,Z positions and cell IDs
        - self.cell_hulls: dictionary with convex hull points per cell
        - self.cells: adds 'cell_volume_voxels', 'centroid_x/y/z', 'convex_area', 'convex_volume'
        """
        import os
        import numpy as np
        import pandas as pd
        from skimage import io
        from scipy.spatial import ConvexHull

        factor = 8 * downsample
        xy_scale = tif_xy_scaling if tif_xy_scaling is not None else self.xy_scaling
        z_scale = tif_z_scaling if tif_z_scaling is not None else self.z_scaling

        if not os.path.dirname(file_path):
            file_path = os.path.join(self.sample_folder, file_path)
        cells = io.imread(file_path)
        if downsample > 1:
            cells = cells[::downsample, ::downsample, ::downsample]

        coords = pd.DataFrame(np.argwhere(cells), columns=['z_pixel', 'x_pixel', 'y_pixel'])
        # Rename columns to match RNA data format
        coords.rename(columns={'x_pixel': 'y_pixel', 'y_pixel': 'x_pixel'}, inplace=True)
        coords['cell_id'] = cells[cells != 0]
        coords = coords[coords['cell_id'] != 0]

        coords['X'] = coords['x_pixel'] * xy_scale * factor
        coords['Y'] = coords['y_pixel'] * xy_scale * factor
        coords['Z'] = coords['z_pixel'] * z_scale * factor

        self.cell_voxels = coords
        self.cell_voxels['cell_id'] = self.cell_voxels['cell_id'].astype(str)
        self.cell_hulls = {}

        cell_features = []
        for cid, group in coords.groupby('cell_id'):
            voxel_volume = len(group)
            centroid = group[['X', 'Y', 'Z']].mean().values
            try:
                hull = ConvexHull(group[['X', 'Y', 'Z']])
                convex_area = hull.area
                convex_volume = hull.volume
                self.cell_hulls[str(cid)] = group.iloc[np.unique(hull.simplices.reshape(-1))]
            except:
                convex_area = np.nan
                convex_volume = np.nan
                self.cell_hulls[str(cid)] = None
        #
        #     cell_features.append({
        #         'cell_id': str(cid),
        #         'cell_volume_voxels': voxel_volume,
        #         'centroid_x': centroid[0],
        #         'centroid_y': centroid[1],
        #         'centroid_z': centroid[2],
        #         'convex_area': convex_area,
        #         'convex_volume': convex_volume
        #     })
        #
        # df_features = pd.DataFrame(cell_features)
        #
        # if hasattr(self, 'cells') and 'cell_id' in self.cells.columns:
        #     self.cells['cell_id'] = self.cells['cell_id'].astype(str)
        #     df_features['cell_id'] = df_features['cell_id'].astype(str)
        #     self.cells = self.cells.merge(df_features, on='cell_id', how='left')
        # else:
        #     self.cells = df_features

    def _create_gene_cell_matrix(self, output_file="gene_cell_matrix.csv"):
        """
        Creates a DataFrame where rows are genes, columns are cells, and each element
        represents the number of copies of a gene in a cell.

        This implementation is optimized for large DataFrames.

        :param output_file: Path to save the resulting DataFrame as a CSV file.
        :return: DataFrame of gene counts per cell.
        """

        if self.RNA_loc_df is None:
            raise ValueError("RNA_loc_df is required to create the gene-cell matrix.")
        if 'cell_id' not in self.RNA_loc_df.columns:
            return None

        # Use pivot_table for efficiency
        gene_cell_matrix = self.RNA_loc_df.pivot_table(
            index='gene',
            columns='cell_id',
            aggfunc='size',
            fill_value=0
        )
        # Ensure the index is a string and remove any column with the name '0' (means no cell related)
        gene_cell_matrix.index = gene_cell_matrix.index.astype(str)
        if '0' in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns='0')
        if 0 in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns=0)

        # Save the matrix to a CSV file
        if output_file is not None:
            gene_cell_matrix.to_csv(self.sample_folder+"//"+output_file)

        self.gene_cell_matrix = gene_cell_matrix
        return gene_cell_matrix

    def calculate_distance_matrix(self, output_file="distance_matrix.csv", print_stages=False, fast_mode=True, first_order_threshold=150):
        """
        Calculates the distance matrix using hybrid approach: for distant cells, calculate distances
        between centroids; for nearby cells use the shortest distance between any pair of punctas

        :param output_file: Path to save the distance matrix as a CSV file.
        """
        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import cdist

        # Extract unique cell IDs from self
        cell_ids = self.cell_ids

        distance_matrix = pd.DataFrame(index=cell_ids, columns=cell_ids, dtype=float)
        if print_stages:
            import time
            start_time = time.time()
            older_time = time.time()
            print(f"Calculating distance matrix of {len(cell_ids)}X{len(cell_ids)} cells...")
        cell_ids_2 = cell_ids.copy()
        # Iterate over each cell and calculate distances
        if fast_mode:
            # For fast mode take the distances of cell centroids
            if self.cell_centroids is None:
                cell_centroids = self._create_cell_centroids()
            else:
                cell_centroids = self.cell_centroids

            # Get the first order distances between the centroids of the cells
            centroids = cell_centroids[['X_centroid', 'Y_centroid', 'Z_centroid']].values
            distance_matrix = cdist(centroids, centroids)

            if print_stages:
                first_order_time = time.time()
                print(f"first order distances calculation took {first_order_time - start_time:.3f} seconds")

            # get the second order of distance for the cells under threshold
            second_order_indices = np.where(distance_matrix < first_order_threshold)
            print(f"calculating second order distance for {len(second_order_indices[0])} pairs of cells")
            for i in range(len(second_order_indices[0])):
                cell1_idx = second_order_indices[0][i]
                cell2_idx = second_order_indices[1][i]
                cell1 = cell_ids[cell1_idx]
                cell2 = cell_ids[cell2_idx]
                if cell1 != cell2:
                    rna_cell1 = self.RNA_loc_df[self.RNA_loc_df['cell_id'] == cell1][['X', 'Y', 'Z']].values
                    rna_cell2 = self.RNA_loc_df[self.RNA_loc_df['cell_id'] == cell2][['X', 'Y', 'Z']].values
                    min_distance = np.min(cdist(rna_cell1, rna_cell2))
                    distance_matrix[cell1_idx, cell2_idx] = min_distance
                    distance_matrix[cell2_idx, cell1_idx] = min_distance
                if print_stages:
                    if i % 1000 == 0 and i > 0:
                        iter_time = time.time()
                        print(f"iterations {i-999}-{i} took {iter_time - older_time:.1f} seconds")
                        older_time = iter_time

            # Fill the diagonal with infinity
            np.fill_diagonal(distance_matrix, np.inf)
            # Convert the distance matrix to a DataFrame
            distance_matrix = pd.DataFrame(distance_matrix, index=cell_ids, columns=cell_ids)

        # For the "not fast mode", all distances are shortest distance between any pair of punctas
        else:
            for cell1 in cell_ids:
                rna_cell1 = self.RNA_loc_df[self.RNA_loc_df['cell_id'] == cell1][['X', 'Y', 'Z']].values
                for cell2 in cell_ids_2:
                    if cell1 == cell2:
                        distance_matrix.loc[cell1, cell2] = np.inf
                    else:
                        rna_cell2 = self.RNA_loc_df[self.RNA_loc_df['cell_id'] == cell2][['X', 'Y', 'Z']].values
                        min_distance = np.min(cdist(rna_cell1, rna_cell2))
                        distance_matrix.loc[cell1, cell2] = min_distance
                        distance_matrix.loc[cell2, cell1] = min_distance
                cell_ids_2.remove(cell1)
                if print_stages:
                    iter_time = time.time()
                    print(f"iteration over {cell1} row took {iter_time - older_time} seconds")
                    older_time = iter_time
        # Save the distance matrix as a CSV file
        distance_matrix.to_csv(self.sample_folder+"//"+output_file)

        if print_stages:
            print(f"calculation of all matrix took {(time.time() - start_time)/60:.2f} minutes")
        # Assign to the class instance
        self.distance_matrix = distance_matrix

    def _create_cell_centroids(self):
        """
        Creates a DataFrame with the centroids of each cell.

        :return: DataFrame with cell centroids.
        """
        if self.RNA_loc_df is None:
            raise ValueError("RNA_loc_df is required to create the cell centroids.")

        if 'cell_id' not in self.RNA_loc_df.columns:
            return None

        # Calculate puncta-based centroids
        cell_centroids = self.RNA_loc_df.groupby('cell_id')[['X', 'Y', 'Z']].mean().reset_index()
        cell_centroids.rename(columns={'X': 'X_centroid', 'Y': 'Y_centroid', 'Z': 'Z_centroid'}, inplace=True)
        # sort the cell centroids by cell_id
        cell_centroids.sort_values(by='cell_id', inplace=True)
        # Ensure cell_id is a string and remove any cells with id '0'
        cell_centroids['cell_id'] = cell_centroids['cell_id'].astype(str)
        if '0' in cell_centroids['cell_id'].values:
            cell_centroids = cell_centroids[cell_centroids['cell_id'] != '0']

        # # Save the centroids to a CSV file
        # cell_centroids.to_csv(self.sample_folder+"//"+"cell_centroids.csv", index=False)

        return cell_centroids

    def _build_weights_matrix(self, dist_matrix, weight_func="inverse", power=2, k=5, sigma=10):
        """
        Build a dense weights matrix from a dense distance matrix, without converting to a libpysal W.

        Parameters
        ----------
        dist_matrix : (N, N) np.ndarray
            Pairwise distances aligned to cell_ids order.
        weight_func : {'inverse', 'fixed_k', 'gaussian'}
            Weighting scheme.
        power : int or float
            Exponent for inverse distance (used when weight_func == 'inverse').
        k : int
            Number of neighbors (used when weight_func == 'fixed_k').
        sigma : float
            Gaussian kernel sigma (used when weight_func == 'gaussian').

        Returns
        -------
        weights : (N, N) np.ndarray
            Dense weight matrix.
        """
        if weight_func == "inverse":
            # w_ij = 1 / d_ij^power (0 on the diagonal and for zero distances)
            weights = np.where(dist_matrix > 0, np.power(dist_matrix, -power), 0.0)

        elif weight_func == "fixed_k":
            # Build KNN graph using a precomputed distance matrix
            dm = dist_matrix.copy()
            np.fill_diagonal(dm, 0.0)
            nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
            nn.fit(dm)
            _, idxs = nn.kneighbors(dm)
            weights = np.zeros_like(dist_matrix, dtype=float)
            for i, neigh in enumerate(idxs):
                weights[i, neigh] = 1.0

        elif weight_func == "gaussian":
            # w_ij = exp( - d_ij^2 / (2 * sigma^2) )
            weights = np.exp(-(dist_matrix ** 2) / (2.0 * (sigma ** 2)))
            np.fill_diagonal(weights, 0.0)

        else:
            raise ValueError("Unsupported weight_func. Choose 'inverse', 'fixed_k', or 'gaussian'.")

        np.fill_diagonal(weights, 0.0)
        return weights

    def _get_or_make_W(self, cell_ids, weight_func="inverse", power=2, k=5, sigma=10):
        """
        Create (or fetch from cache) a libpysal W object consistent with the given cell_ids order and parameters.

        Notes
        -----
        - Ensures self.distance_matrix is aligned to the given cell_ids order.
        - Caches by (N_cells, weight_func, power, k, sigma) to avoid recomputing.
        """
        # Align distance matrix to cell_ids (as strings)
        self.distance_matrix.index = self.distance_matrix.index.astype(str)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(str)
        dist_mat = self.distance_matrix.loc[cell_ids, cell_ids].values

        key = (len(cell_ids), weight_func, float(power), int(k), float(sigma))
        if key in self._W_cache:
            return self._W_cache[key]

        weights = self._build_weights_matrix(dist_mat, weight_func=weight_func, power=power, k=k, sigma=sigma)
        W = full2W(weights, ids=list(cell_ids))

        self._W_cache[key] = W
        return W

    @staticmethod
    def _is_binary_vector(y):
        """
        Return True if y is strictly binary {0,1} (or constant, which is a degenerate binary).
        """
        y = np.asarray(y, dtype=float)
        uv = np.unique(y[~np.isnan(y)])
        return np.all(np.isin(uv, [0.0, 1.0])) or (uv.size == 1)

    @staticmethod
    def _safe_moran(y, W, permutations=999):
        """
        Robust Moran's I wrapper:
        - If y has zero variance (constant vector), return (I=NaN, z=0, p=1) without warnings.
        - Otherwise, compute standard esda.Moran with permutations.
        """
        y = np.asarray(y, dtype=float)
        if y.size == 0 or np.allclose(y, y[0]):
            return dict(I=np.nan, z=0.0, p=1.0)
        mor = Moran(y, W, permutations=permutations)
        return dict(I=mor.I, z=mor.z_sim, p=mor.p_z_sim)

    def _run_spatial_stats(
        self,
        signals: dict,              # mapping: name -> 1D np.ndarray aligned to cell_ids
        cell_ids,
        weight_func="inverse",
        power=2,
        k=5,
        sigma=10,
        do_moran=True,
        do_join_counts=True,
        permutations=999
    ):
        """
        Core engine: given multiple signals on the same cell lattice, compute Moran's I for all,
        and Join Counts for binary signals.

        Parameters
        ----------
        signals : dict[str, np.ndarray]
            Each vector must be aligned to the same cell_ids order.
        cell_ids : list[str]
            Cell IDs order used to align W and signals.
        weight_func, power, k, sigma : see _build_weights_matrix
        do_moran : bool
            If True, compute Moran's I for all signals.
        do_join_counts : bool
            If True, compute Join Counts for binary signals only.
        permutations : int
            Number of permutations for both Moran and Join Counts.

        Returns
        -------
        pd.DataFrame
            One row per signal with Moran/JC stats and metadata.
        """
        W = self._get_or_make_W(cell_ids, weight_func=weight_func, power=power, k=k, sigma=sigma)
        rows = []
        for name, y in signals.items():
            y = np.asarray(y, dtype=float)

            # Moran's I (works for continuous and binary)
            I = z = p = np.nan
            if do_moran:
                m = self._safe_moran(y, W, permutations=permutations)
                I, z, p = m["I"], m["z"], m["p"]

            # Join Counts (only meaningful for strictly binary, non-constant vectors)
            jc_bb = jc_bw = jc_ww = np.nan
            jc_bb_p = np.nan
            if do_join_counts and self._is_binary_vector(y):
                uv = np.unique(y[~np.isnan(y)])
                if uv.size == 2:  # not constant, true 0/1
                    jc = Join_Counts(y, W, permutations=permutations)
                    jc_bb, jc_bw, jc_ww = jc.bb, jc.bw, jc.ww
                    # permutation p-value for BB from esda API
                    jc_bb_p = jc.p_sim_bb
                    # direction of spatial pattern
                    jc_bb_dir = "clustered" if jc_bb > jc.mean_bb else "dispersed"

                    # Chi Squared statistic and p-value
                    jc_chi2 = jc.chi2
                    jc_chi2_p = jc.chi2_p  # analytic p-value
                    jc_p_sim_chi2 = jc.p_sim_chi2  # permutation p-value for chi2

            rows.append({
                "name": name,
                "Moran_I": I, "Moran_z": z, "Moran_p": p,
                "JC_BB": jc_bb, "JC_BW": jc_bw, "JC_WW": jc_ww,
                "JC_BB_p": jc_bb_p, "JC_BB_dir": jc_bb_dir, "JC_chi2": jc_chi2,
                "JC_chi2_p": jc_chi2_p, "JC_p_sim_chi2": jc_p_sim_chi2,
                "weight_method": weight_func,
                "weight_param": dict(inverse=power, fixed_k=k, gaussian=sigma).get(weight_func),
                "n_cells": len(cell_ids),
                "n_ones": int(np.nansum(y)) if self._is_binary_vector(y) else np.nan
            })

        return pd.DataFrame(rows).sort_values(["Moran_p", "Moran_I"], na_position="last")

    # -----------------------------
    # Genes: unified API (continuous)
    # -----------------------------
    def calculate_spatial_stats_for_genes(
        self,
        weight_func="inverse", power=2, k=5, sigma=10,
        normalize_counts=False,
        permutations=999,
        save_results=True
    ):
        """
        Compute Moran's I for each gene (continuous signal per cell).
        Join Counts is disabled for genes.

        Returns
        -------
        pd.DataFrame
        """
        if getattr(self, "distance_matrix", None) is None:
            raise ValueError("Distance matrix is missing. Calculate it first.")
        if getattr(self, "gene_cell_matrix", None) is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")

        gcm = self.gene_cell_matrix.copy()
        gcm.columns = gcm.columns.astype(str)

        # Drop problematic columns named 0
        for bad in (0, "0"):
            if bad in gcm.columns:
                gcm = gcm.drop(columns=bad)

        # --- Keep only cells that exist in distance matrix ---
        dm = self.distance_matrix.copy()
        dm.index = dm.index.astype(str);
        dm.columns = dm.columns.astype(str)
        valid_cols = [cid for cid in gcm.columns if cid in dm.index]
        gcm = gcm[valid_cols]

        cell_ids = gcm.columns

        # Optional per-cell normalization (column-wise)
        if normalize_counts:
            gcm = gcm.div(gcm.sum(axis=0), axis=1)

        # Build signals: each gene -> vector over cells
        signals = {gene: gcm.loc[gene].values for gene in gcm.index}

        df = self._run_spatial_stats(
            signals, cell_ids,
            weight_func=weight_func, power=power, k=k, sigma=sigma,
            do_moran=True, do_join_counts=False,
            permutations=permutations
        ).rename(columns={"name": "Gene"})

        # Optional enrichment from self.gene_info
        if hasattr(self, "gene_info"):
            extra_cols = ["total_counts", "expression_in_cells", "num_cells"]
            extra = self.gene_info[[c for c in extra_cols if c in self.gene_info.columns]].copy()
            df = df.merge(extra, left_on="Gene", right_index=True, how="left")
            if normalize_counts:
                # Optional display-only normalization of counts
                if "total_counts" in df:
                    s1 = df["total_counts"].sum(skipna=True)
                    if s1 and s1 > 0:
                        df["normalized_counts (total)"] = df["total_counts"] / s1
                if "expression_in_cells" in df:
                    s2 = df["expression_in_cells"].sum(skipna=True)
                    if s2 and s2 > 0:
                        df["normalized_counts (in cells)"] = df["expression_in_cells"] / s2

        if save_results:
            norm_tag = "_normalized" if normalize_counts else ""
            param = dict(inverse=power, fixed_k=k, gaussian=sigma)[weight_func]
            out = f"{self.sample_folder}/genes_spatial_stats_{weight_func}_{param}{norm_tag}.csv"
            df.to_csv(out, index=False)

        return df

    # --------------------------------
    # Cell types: unified API (binary)
    # --------------------------------
    def calculate_spatial_stats_for_celltypes(
        self,
        weight_func="inverse", power=2, k=5, sigma=10,
        include_join_counts=True,
        permutations=999,
        save_results=True,
        drop_labels=("No_type", "NaN", "0")
    ):
        """
        Compute Moran's I (and optionally Join Counts) for each cell type using one-vs-rest binary signals.

        Returns
        -------
        pd.DataFrame
        """
        if getattr(self, "distance_matrix", None) is None:
            raise ValueError("Distance matrix is missing. Calculate it first.")
        if not hasattr(self, "cells") or "cell_id" not in self.cells or "cell_type" not in self.cells:
            raise ValueError("cells DataFrame with ['cell_id','cell_type'] is required.")

        cells = self.cells.copy()
        cells["cell_id"] = cells["cell_id"].astype(str)
        cells["cell_type"] = cells["cell_type"].astype(str)
        cells = cells.set_index("cell_id")

        # --- Filter cells that have molecular content (num_punctas > 0) and exist in distance matrix ---
        dm = self.distance_matrix.copy()
        dm.index = dm.index.astype(str);
        dm.columns = dm.columns.astype(str)
        content_ids = set(cells.index[cells["num_punctas"].fillna(0) > 0].astype(str))
        ids_in_dm = set(dm.index)
        cell_ids = sorted(content_ids & ids_in_dm)
        cells = cells.loc[cell_ids]

        cell_ids = cells.index.astype(str).tolist()

        # Collect distinct cell types (drop specified labels)
        all_types = [t for t in cells["cell_type"].unique() if t not in set(drop_labels)]
        all_types = sorted(all_types)

        # Build binary signals: each type -> 0/1 vector over cells
        signals = {}
        ct_series = cells["cell_type"]
        for ct in all_types:
            y = (ct_series == ct).reindex(cell_ids).astype(int).values
            signals[ct] = y

        df = self._run_spatial_stats(
            signals, cell_ids,
            weight_func=weight_func, power=power, k=k, sigma=sigma,
            do_moran=True, do_join_counts=include_join_counts,
            permutations=permutations
        ).rename(columns={"name": "cell_type"})

        if save_results:
            param = dict(inverse=power, fixed_k=k, gaussian=sigma)[weight_func]
            out = f"{self.sample_folder}/celltypes_spatial_stats_{weight_func}_{param}.csv"
            df.to_csv(out, index=False)

        return df

    # -----------------------------
    # Backward-compatible wrapper
    # -----------------------------
    def calculate_spatial_morans_I(
        self,
        weight_func="inverse", power=2, k=5, sigma=10,
        normalize_counts=False, save_results=True, permutations=999
    ):
        """
        Backward-compatible method: returns Moran's I for genes only.
        This calls the new unified genes method under the hood.
        """
        return self.calculate_spatial_stats_for_genes(
            weight_func=weight_func, power=power, k=k, sigma=sigma,
            normalize_counts=normalize_counts,
            permutations=permutations,
            save_results=save_results
        )

    def calculate_spatial_morans_I_old(self, weight_func="inverse", power=2, k=5, sigma=10, normalize_counts=False, save_results=True):
        """
        Calculates spatial Moran's I for each gene in the sample using the esda package.

        :param weight_func: The function to determine weights. Options: 'inverse', 'fixed_k', 'gaussian'.
            - 'inverse': Uses inverse distance raised to a power.
            - 'fixed_k': Uses k-nearest neighbors. Currently in editing.
            - 'gaussian': Uses Gaussian kernel weights.
        :param power: Power for inverse distance weighting. Ignored if weight_func is not 'inverse'.
        :param k: Number of nearest neighbors for k-nearest neighbors. Ignored if weight_func is not 'fixed_k'.
        :param sigma: Standard deviation for Gaussian kernel. Ignored if weight_func is not 'gaussian'.
        :param normalize_counts: Whether to scale gene counts before calculating Moran's I.
        :param save_results: Whether to save the results to a CSV file in the sample folder.
        :return: DataFrame with Moran's I values for each gene.
        """
        import numpy as np
        import pandas as pd
        from libpysal.weights import full2W
        from esda.moran import Moran

        # Ensure necessary attributes exist
        if not hasattr(self, "distance_matrix") or self.distance_matrix is None:
            raise ValueError("Distance matrix is missing. Calculate it first.")
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")
        if not hasattr(self, "sample_folder") or not self.sample_folder:
            if save_results:
                output_folder = input("Sample folder path is missing in the object. Please enter the path to save results: ")
                self.sample_folder = output_folder

        # Extract data
        gene_cell_matrix = self.gene_cell_matrix
        gene_cell_matrix.columns = gene_cell_matrix.columns.astype(str)
        if 0 in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns=0)
        if '0' in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns='0')
        cell_ids = gene_cell_matrix.columns

        # Normalize gene counts if requested
        gene_cell_matrix_normalized = gene_cell_matrix.div(gene_cell_matrix.sum(axis=0), axis=1)
        normalize_string = "_normalized" if normalize_counts else ""

        # Construct weight matrix
        self.distance_matrix.index = self.distance_matrix.index.astype(str)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(str)
        dist_matrix = self.distance_matrix.loc[cell_ids, cell_ids].values  # Ensure matching order

        if weight_func == "inverse":
            weighting_parameter = power
            weights = np.where(dist_matrix > 0, np.power(dist_matrix, -power), 0)  # Avoid division by zero

        elif weight_func == "fixed_k":
            from sklearn.neighbors import NearestNeighbors
            weighting_parameter = k
            knn = NearestNeighbors(n_neighbors=k)
            dist_matrix_for_knn = dist_matrix
            np.fill_diagonal(dist_matrix_for_knn, 0)
            # np.fill_diagonal(dist_matrix_for_knn, max(dist_matrix_for_knn.flatten()))
            knn.fit(dist_matrix_for_knn)
            distances, indices = knn.kneighbors()
            weights = np.zeros_like(dist_matrix)
            for i, neighbors in enumerate(indices):
                for neighbor in neighbors:
                    weights[i, neighbor] = 1.0

        elif weight_func == "gaussian":
            weighting_parameter = sigma
            weights = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unsupported weight_func: {weight_func}. Choose 'inverse', 'fixed_k', or 'gaussian'.")

        # Weights of self loop is 0
        np.fill_diagonal(weights, 0)

        # Convert the dense weight matrix into a PySAL weights object with cell IDs
        w = full2W(weights, list(cell_ids))

        # Calculate Moran's I for each gene
        morans_results = []
        for gene in gene_cell_matrix.index:
            gene_expression = gene_cell_matrix_normalized.loc[gene].values if normalize_counts else gene_cell_matrix.loc[gene].values
            moran = Moran(gene_expression, w)
            morans_results.append({
                "Gene": gene,
                "Moran_I": moran.I,
                "p_value": moran.p_z_sim,
                "z_score": moran.z_sim
            })

            # Optional - Save morans partial results to file - to avoid loss of results in case of interruption
            save_partial_results = False # Set to True if you want to save partial results
            if save_partial_results:
                morans_df = pd.DataFrame(morans_results)
                output_file = f"{self.sample_folder}/morans_I_with_weight_method_of_{weight_func}_{weighting_parameter:.2f}{normalize_string}.csv"
                morans_df.to_csv(output_file, index=False)
                print(f"Moran's I is calculated for gene {gene} and saved to file")


        # Convert results to DataFrame
        morans_df = pd.DataFrame(morans_results)

        # Add counts of each gene in the sample using self.gene_info
        if hasattr(self, 'gene_info'):
            morans_df = morans_df.merge(self.gene_info[['total_counts', 'expression_in_cells', 'num_cells']],
                                        left_on='Gene', right_index=True, how='left')

            if normalize_counts:
                morans_df["total_counts"] = morans_df["total_counts"] / morans_df["total_counts"].sum()
                morans_df.rename(columns={"total_counts": "normalized_counts (total)"}, inplace=True)
                morans_df["expression_in_cells"] = morans_df["expression_in_cells"] / morans_df["expression_in_cells"].sum()
                morans_df.rename(columns={"expression_in_cells": "normalized_counts (in cells)"}, inplace=True)


        # Save results if requested
        if save_results:
            output_file = f"{self.sample_folder}/morans_I_with_weight_method_of_{weight_func}_{weighting_parameter:.2f}{normalize_string}.csv"
            morans_df.to_csv(output_file, index=False)

        return morans_df


    def variance_analisis(self, with_shuffle=False):
        """
        Placeholder for a method to analyze the variance of gene expression in the sample.

        :return:
        """
        import numpy as np
        import pandas as pd

        # Ensure necessary attributes exist
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")
        if not hasattr(self, "sample_folder") or not self.sample_folder:
            raise ValueError("Sample folder path is missing in the object.")

        # Extract data
        gene_cell_matrix = self.gene_cell_matrix
        gene_cell_matrix.columns = gene_cell_matrix.columns.astype(str)
        if 0 in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns=0)
        if '0' in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns='0')

        # Calculate mean and variance for each gene
        gene_means = gene_cell_matrix.mean(axis=1)
        gene_vars = gene_cell_matrix.var(axis=1)

        # Shuffle raw data and calculate the gene cell matrix
        if with_shuffle:
            RNA_loc_df_shuffled_by_gene = self.RNA_loc_df.copy()
            RNA_loc_df_shuffled_by_gene['gene'] = np.random.permutation(RNA_loc_df_shuffled_by_gene['gene'])
            gene_cell_matrix_shuffled_1 = RNA_loc_df_shuffled_by_gene.pivot_table(
                index='gene',
                columns='cell_id',
                aggfunc='size',
                fill_value=0
            )
            gene_cell_matrix_shuffled_1.columns = gene_cell_matrix_shuffled_1.columns.astype(str)
            if 0 in gene_cell_matrix_shuffled_1.columns:
                gene_cell_matrix_shuffled_1 = gene_cell_matrix_shuffled_1.drop(columns=0)
            if '0' in gene_cell_matrix_shuffled_1.columns:
                gene_cell_matrix_shuffled_1 = gene_cell_matrix_shuffled_1.drop(columns='0')
            gene_means_shuffled = gene_cell_matrix_shuffled_1.mean(axis=1)
            gene_vars_shuffled = gene_cell_matrix_shuffled_1.var(axis=1)

        # plot variance vs mean
        if with_shuffle:
            # plot variance vs mean for real and shuffled data
            import matplotlib.pyplot as plt
            plt.subplots(1, 2, figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(gene_means, gene_vars)
            plt.xlabel('Mean')
            plt.ylabel('Variance')
            plt.title(f'Variance analisis for {self.sample_name}')
            plt.subplot(1, 2, 2)
            plt.scatter(gene_means_shuffled, gene_vars_shuffled)
            plt.xlabel('Mean')
            plt.ylabel('Variance')
            plt.title(f'Variance analisis of shuffled data for {self.sample_name}')
            plt.show()
        else:
            # plot variance vs mean for real data
            import matplotlib.pyplot as plt
            plt.scatter(gene_means, gene_vars)
            plt.xlabel('Mean')
            plt.ylabel('Variance')
            plt.title(f'Variance analisis for {self.sample_name}')
            plt.show()

        # Create and return a DataFrame with the results
        results = pd.DataFrame({
            "Gene": gene_means.index,
            "Mean": gene_means.values,
            "Variance": gene_vars.values
        })
        results = results.sort_values(by="Mean", ascending=False)
        return results


    def distribution_over_cells(self, gene, filter_zeros=True, logged=True):
        """
        Placeholder for a method to analyze the distribution of gene expression over cells in the sample.

        :return:
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import scipy.stats as stats

        def chi_square_test(data, dist, params):
            """
            Perform Chi-Square goodness-of-fit test for Poisson or Negative Binomial.

            Parameters:
            data (array-like): Observed gene expression counts (nonzero if filtering).
            dist (str): 'poisson' or 'nbinom'.
            params (tuple): Parameters for the chosen distribution.

            Returns:
            float: p-value (low p-value means poor fit).
            """
            # Bin observed data
            observed_counts, bin_edges = np.histogram(data, bins=np.arange(0, max(data) + 2) - 0.5)

            # Compute expected frequencies based on the theoretical distribution
            if dist == 'poisson':
                expected_probs = stats.poisson.pmf(bin_edges[:-1], mu=params[0])
            elif dist == 'nbinom':
                expected_probs = stats.nbinom.pmf(bin_edges[:-1], *params)
            else:
                raise ValueError("Choose 'poisson' or 'nbinom'.")

            # Scale expected counts to match total observed counts
            expected_counts = expected_probs * np.sum(observed_counts)
            expected_counts = np.maximum(expected_counts, 1e-5)  # Avoid zero values

            # Ensure total sums match (fixes the error)
            expected_counts *= np.sum(observed_counts) / np.sum(expected_counts)

            # Perform Chi-square test
            chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

            return p_value

        # Ensure necessary attributes exist
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")
        if not hasattr(self, "sample_folder") or not self.sample_folder:
            raise ValueError("Sample folder path is missing in the object.")

        # Extract data
        gene_cell_matrix = self.gene_cell_matrix
        if gene not in gene_cell_matrix.index:
            raise ValueError(f"Gene '{gene}' not found in DataFrame index.")

        gene_cell_matrix.columns = gene_cell_matrix.columns.astype(str)
        if 0 in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns=0)
        if '0' in gene_cell_matrix.columns:
            gene_cell_matrix = gene_cell_matrix.drop(columns='0')
        cell_ids = gene_cell_matrix.columns

        data = gene_cell_matrix.loc[gene].values  # Extract the values for the specific gene

        if filter_zeros:
            data = data[data > 0]  # Remove zero values

        if len(data) == 0:
            raise ValueError(f"All values for gene '{gene}' are zero. Nothing to plot.")

        # Fit Poisson and Negative Binomial distributions
        filtered_data = data #[data > 0]  # Remove zeros if needed
        poisson_pval = chi_square_test(filtered_data, 'poisson', (np.mean(filtered_data),))
        p = np.mean(filtered_data)/np.var(filtered_data)
        r = np.mean(filtered_data)**2/(np.var(filtered_data)-np.mean(filtered_data)+1e-10)
        nbinom_pval = chi_square_test(filtered_data, 'nbinom', (r, p))  # Example NB params
        print(f"Poisson Fit p-value: {poisson_pval}")
        print(f"NB Fit p-value: {nbinom_pval}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if logged:
            if filter_zeros:
                data = np.log(data)
            else:
                data = np.log1p(data)

        # Boxplot
        logged_str = " (logged)" if logged else ""
        sns.boxplot(y=data, ax=axes[0], color="lightblue")
        axes[0].set_title(f'Boxplot of {gene} Expression {logged_str}')
        axes[0].set_ylabel('Expression Level')

        # Histogram
        sns.histplot(data, kde=True, ax=axes[1], color="lightblue")
        axes[1].set_title(f'Histogram of {gene} Expression {logged_str}')
        axes[1].set_xlabel('Expression Level')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
        # return plt

    def plot_gene_expression_by_cell_type(self, gene, cell_type, filter_zeros=True, logged=True, return_ax=False, ax=None):
        """
        Placeholder for a method to plot gene expression by cell type.
        :param gene: The gene to plot.
        :param cell_type: The cell type to filter results.
        :param filter_zeros: Whether to filter out zero values.
        :param logged: Whether to log-transform the data.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        # Ensure necessary attributes exist
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")
        if not hasattr(self, "RNA_loc_df") or self.RNA_loc_df is None:
            raise ValueError("RNA_loc_df is missing. Calculate it first.")
        required_cols = ["cell_type", "gene", "X", "Y", "Z", "cell_id"]
        if not all(col in self.RNA_loc_df.columns for col in required_cols):
            raise ValueError("RNA_loc_df must contain 'cell_type', 'gene', 'X', 'Y', 'Z' and 'cell_id' columns.")

        # Extract data
        RNA_loc_df = self.RNA_loc_df
        gene_cell_matrix = self.gene_cell_matrix
        if gene not in RNA_loc_df['gene'].values:
            raise ValueError(f"Gene '{gene}' not found in DataFrame.")
        if cell_type not in RNA_loc_df['cell_type'].values:
            raise ValueError(f"Cell type '{cell_type}' not found in DataFrame.")
        cell_centroids = self.cell_centroids
        if cell_centroids is None:
            cell_centroids = self._create_cell_centroids()

        # Filter data
        filtered_data = RNA_loc_df[(RNA_loc_df['gene'] == gene) & (RNA_loc_df['cell_type'] == cell_type)]
        if filter_zeros:
            cell_ids = filtered_data['cell_id'].unique().astype(str)
        else:
            cell_ids = RNA_loc_df[RNA_loc_df['cell_type'] == cell_type]['cell_id'].unique().astype(str)

        # Filter the gene-cell matrix and cell centriods to include only the relevant cells and the relevant gene
        if gene not in gene_cell_matrix.index:
            raise ValueError(f"Gene '{gene}' not found in gene-cell matrix.")
        gene_cell_matrix = gene_cell_matrix.loc[gene]

        gene_cell_matrix = gene_cell_matrix[gene_cell_matrix.index.intersection(cell_ids)]
        cell_centroids = cell_centroids[cell_centroids['cell_id'].astype(str).isin(cell_ids)]

        # Plot spatial gene expression map of the cells with the same cell type
        if not return_ax:
            fig, ax = plt.subplots(figsize=(7, 5))
        if logged:
            gene_cell_matrix = np.log1p(gene_cell_matrix)

        # Plot the centroids of the cells with small black points and the gene expression as color map
        scatter = ax.scatter(cell_centroids['X_centroid'], cell_centroids['Y_centroid'], c=gene_cell_matrix.values, cmap='viridis_r', s=8)
        tissue_id = self.sample_name
        ax.set_title(f'Gene Expression of {gene} over {cell_type} Cells (in {tissue_id})')
        ax.set_xlabel(r'X [$\mu$]')
        ax.set_ylabel(r'Y [$\mu$]')
        plt.colorbar(scatter, label=f'{gene} Expression Level')
        if return_ax:
            return ax
        else:
            plt.show()
            return fig

    def calc_cell_correlation_matrix(self, filter_genes=None, filter_cells=None, method='morans_i_term', save_results=False):
        """
        Placeholder for a method to calculate the correlation matrix of cells based on gene expression.

        :param filter_genes: List of genes to filter. If None, all genes are used.
        :param method: Method for calculating correlation. Options: 'pearson', 'spearman', 'morans_i_term'.
        :param save_results: Whether to save the results to a CSV file.
        """
        import pandas as pd
        import numpy as np

        # Ensure necessary attributes exist
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")
        if not hasattr(self, "sample_folder") or not self.sample_folder:
            raise ValueError("Sample folder path is missing in the object.")

        # Extract data
        gene_cell_matrix = self.gene_cell_matrix
        gene_cell_matrix.columns = gene_cell_matrix.columns.astype(str)
        cell_ids = self.cell_ids

        # Filter genes and cells if specified
        if filter_cells is not None:
            gene_cell_matrix = gene_cell_matrix[gene_cell_matrix.columns.intersection(filter_cells)]
        if filter_genes is not None:
            gene_cell_matrix = gene_cell_matrix.loc[filter_genes]
            cell_ids = filter_cells

        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = gene_cell_matrix.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = gene_cell_matrix.corr(method='spearman')
        elif method == 'morans_i_term':
            # Enable this approach just for single gene
            if len(gene_cell_matrix.shape) > 1:
                raise ValueError("For 'morans_i_term', please provide a single gene.")
            # Calculate the mean and variance of the gene expression across cells
            gene_expression = gene_cell_matrix
            mean_expression = np.mean(gene_expression)
            variance_expression = np.var(gene_expression)
            if variance_expression == 0:
                raise ValueError("Variance is zero. Cannot calculate Moran's I term.")
            # Calculate the Moran's I term for each pair of cells
            # The formula is: (xi - mean) * (xj - mean) / variance
            # Initialize the correlation matrix
            corr_matrix = pd.DataFrame(index=cell_ids, columns=cell_ids, dtype=float)
            for i, cell1 in enumerate(cell_ids):
                xi = gene_cell_matrix.loc[cell1] - mean_expression
                for j, cell2 in enumerate(cell_ids):
                    if i == j:
                        corr_matrix.loc[cell1, cell2] = 1.0
                    elif i<j:
                        xj = gene_cell_matrix.loc[cell2] - mean_expression
                        corr_matrix.loc[cell1, cell2] = (xi * xj) / variance_expression
                        corr_matrix.loc[cell2, cell1] = corr_matrix.loc[cell1, cell2]

        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'pearson', 'spearman', or 'morans_i_term'.")

        # Save results if requested
        if save_results:
            output_file = f"{self.sample_folder}/cell_correlation_{method}.csv"
            corr_matrix.to_csv(output_file)

        return corr_matrix


    def plot_correlation_vs_distance(self, threshold=100, filter_genes=None, filter_types=None, method='morans_i_term', save_results=False):
        """
        Placeholder for a method to plot the correlation of cells vs distance between cells.
        :param gene:
        :param filter_genes:
        :param method:
        :param save_results:
        :return:
        """

        # First load distance matrix and calculate correlation matrix
        if not hasattr(self, "distance_matrix") or self.distance_matrix is None:
            raise ValueError("Distance matrix is missing. Calculate it first.")
        if not hasattr(self, "gene_cell_matrix") or self.gene_cell_matrix is None:
            raise ValueError("Gene-cell matrix is missing. Calculate it first.")

        # Filter cell types if needed
        if filter_types is not None:
            if not hasattr(self, "cells") or self.cells is None:
                raise ValueError("Cell types are missing. Please load them first.")
            if type(filter_types) == str:
                filter_types = [filter_types]
            cell_ids = self.cells[self.cells['cell_type'].isin(filter_types)]['cell_id'].astype(str).unique()
            distance_matrix = self.distance_matrix.loc[cell_ids, cell_ids]
            correlation_matrix = self.calc_cell_correlation_matrix(filter_genes=filter_genes, filter_cells=cell_ids, method=method,
                                                                   save_results=save_results)
        else:
            distance_matrix = self.distance_matrix
            correlation_matrix = self.calc_cell_correlation_matrix(filter_genes=filter_genes, method=method,
                                                                   save_results=save_results)

        # Extract the distance values and correlation values without the diagonal and without duplications
        distance_values = []
        correlation_values = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if distance_matrix.iloc[i, j] > threshold:
                    continue
                distance_values.append(distance_matrix.iloc[i, j])
                correlation_values.append(correlation_matrix.iloc[i, j])

        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 5))
        # Plot scatter plot for all correlations VS distances
        ax.scatter(distance_values, correlation_values, alpha=0.5)
        ax.set_xlabel("Distance (um)")
        method_str = method if method != 'morans_i_term' else 'Moran\'s I term'
        ax.set_ylabel(f"Correlation ({method_str})")
        ax.set_title(f"Correlation vs Distance")
        if method == 'morans_i_term':
            ax.set_title(f"Correlation vs Distance for gene {filter_genes}")
            if filter_types:
                ax.set_title(f"Correlation vs Distance for gene {filter_genes} in {filter_types[0]}")
        # Calculate and plot quantiles as function of distance (estimated correlation function)
        bins = np.linspace(0, max(distance_values), 20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        corr_dict = {}
        for i in range(len(bins) - 1):
            mask = (distance_values >= bins[i]) & (distance_values < bins[i + 1])
            if np.sum(mask) > 0:
                corr_dict[i] = [correlation_values[j] for j in range(len(correlation_values)) if mask[j]]
            else:
                corr_dict[i] = [0]
        q1 = []
        q2 = []
        q3 = []
        means = []
        for key in corr_dict.keys():
            q1.append(np.quantile(corr_dict[key], 0.95))
            q2.append(np.quantile(corr_dict[key], 0.5))
            q3.append(np.quantile(corr_dict[key], 0.75))
            means.append(np.mean(corr_dict[key]))
        ax.plot(bin_centers, q1, 'r--', label='95%')
        ax.plot(bin_centers, q2, 'g--', label='50%')
        ax.plot(bin_centers, q3, 'b--', label='75%')
        ax.plot(bin_centers, means, 'k-', label='Mean')
        # Print the slope of the mean line
        slope = np.polyfit(bin_centers, means, 1)[0]
        print(f"Slope of the mean line: {slope}")
        ax.legend()
        plt.show()

        # Plot 2D histogram of distances and correlations
        log_color_scale = True
        from matplotlib.colors import LogNorm

        # 2D histogram
        hist, xedges, yedges = np.histogram2d(distance_values, correlation_values, bins=30)

        fig, ax = plt.subplots(figsize=(7, 5))
        # c = ax.imshow(hist.T, origin='lower',
        #               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        #               aspect='auto', cmap='viridis_r')

        # Colorbar
        if log_color_scale:
            # To avoid log(0), set zeros to a small positive number or mask them
            hist = np.where(hist == 0, np.nan, hist)
            norm = LogNorm(vmin=np.nanmin(hist), vmax=np.nanmax(hist))
            c = ax.pcolormesh(xedges, yedges, hist.T, cmap='viridis_r', shading='auto', norm=norm)
        else:
            c = ax.pcolormesh(xedges, yedges, hist.T, cmap='viridis_r', shading='auto')

        # Colorbar
        fig.colorbar(c, ax=ax, label='Frequency (log scale)' if log_color_scale else 'Frequency')
        # Labels
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel(f"Correlation ({method_str})")

        # Dynamic title
        title = "2D Histogram of Correlation vs Distance"
        if filter_genes and len(filter_genes) < 3:
            title += f" for gene {filter_genes}"
        if filter_types and len(filter_types) == 1:
            title += f" in {filter_types[0]} cells"
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        # region commented - Plot histogram of distances related to the top quantile of correlations
        # # Filter the top quantile from all correlations, and show histogram of distances related to these correlations
        # # Get the top quantile of correlations from correlation matrix and remove the diagonal and duplications
        #
        # upper_matrix_indices = np.triu_indices_from(correlation_matrix.values, k=1)
        # upper_matrix_correlations = correlation_matrix.values[upper_matrix_indices]
        # upper_matrix_distances = distance_matrix.values[upper_matrix_indices]
        #
        # # Get the top quantile of correlations
        # quantile = np.quantile(upper_matrix_correlations, 0.95)
        # # Get the distances related to these correlations
        # top_corr_distances = upper_matrix_distances[upper_matrix_correlations >= quantile]
        #
        # # Calculate histogram of distances related to the top quantile of correlations and normalize it by the histogram of distances
        # # Get the histogram of all distances
        # hist_all, bin_edges = np.histogram(upper_matrix_distances, bins=30)
        # # Get the histogram of top correlated pairs distances (keep the same bins)
        # hist_top, _ = np.histogram(top_corr_distances, bins=bin_edges)
        # # Normalize by the histogram of all distances (each bin separately)
        # hist_top_normalized = hist_top / hist_all
        #
        # # Plot normalized histogram
        # fig, ax = plt.subplots(figsize=(7, 5))
        # ax.bar(bin_edges[:-1], hist_top_normalized, width=np.diff(bin_edges), align='edge', edgecolor='black', alpha=0.7)
        # ax.set_xlabel("Distance (um)")
        # ax.set_ylabel("Relation (top 5%/ all)")
        # ax.set_title(f"Histogram of Distances for Top 5% Correlations")
        # if filter_genes and len(filter_genes)<3:
        #     ax.set_title(f"Histogram of Distances for Top 5% Correlations for gene {filter_genes}")
        # if filter_types and len(filter_types) == 1:
        #     ax.set_title(f"Histogram of Distances for Top 5% Correlations for {filter_types[0]} cells")
        # if filter_genes and len(filter_genes)<3 and filter_types and len(filter_types) == 1:
        #     ax.set_title(f"Histogram of Distances for Top 5% Correlations for gene {filter_genes} in {filter_types[0]} cells")
        # plt.show()
        #
        # # Plot hist_all
        # fig, ax = plt.subplots(figsize=(7, 5))
        # ax.bar(bin_edges[:-1], hist_all, width=np.diff(bin_edges), align='edge', edgecolor='black', alpha=0.7)
        # ax.set_xlabel("Distance (um)")
        # ax.set_ylabel("Frequency")
        # ax.set_title(f"Histogram of Distances")
        # if filter_genes and len(filter_genes)<3:
        #     ax.set_title(f"Histogram of Distances for gene {filter_genes}")
        # if filter_types and len(filter_types) == 1:
        #     ax.set_title(f"Histogram of Distances for {filter_types[0]} cells")
        # if filter_genes and len(filter_genes)<3 and filter_types and len(filter_types) == 1:
        #     ax.set_title(f"Histogram of Distances for gene {filter_genes} in {filter_types[0]} cells")
        # plt.show()
        # endregion

    def plot_cell_with_puncta(self, cell_ids, gene_filter=None, opacity=0.2, use_voxels=False):
        """
        Display 3D convex hulls or voxel clouds of selected cells along with their RNA puncta.

        Parameters:
        - cell_ids: str or list of str, cell ID(s) to plot
        - gene_filter: optional str or list of str to filter puncta by gene name
        - opacity: float between 0 and 1 for the hull transparency
        - use_voxels: if True, plot cell using voxel cloud instead of convex hull
        """
        import plotly.graph_objects as go
        import plotly.express as px

        if isinstance(cell_ids, str):
            cell_ids = [cell_ids]
        if gene_filter is not None and isinstance(gene_filter, str):
            gene_filter = [gene_filter]

        fig = go.Figure()
        colors = px.colors.qualitative.Set2  # default color cycle

        for idx, cid in enumerate(cell_ids):
            if use_voxels and hasattr(self, 'cell_voxels'):
                points = self.cell_voxels[self.cell_voxels['cell_id'] == str(cid)]
                fig.add_trace(go.Scatter3d(
                    x=points['X'], y=points['Y'], z=points['Z'],
                    mode='markers',
                    marker=dict(size=1.5, color=colors[idx % len(colors)], opacity=opacity),
                    name=f"Cell {cid} (voxels)"
                ))
            else:
                hull_points = self.cell_hulls.get(str(cid))
                if hull_points is None:
                    continue

                # Create mesh from convex hull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(hull_points[['X', 'Y', 'Z']])
                simplices = hull.simplices

                fig.add_trace(go.Mesh3d(
                    x=hull_points['X'],
                    y=hull_points['Y'],
                    z=hull_points['Z'],
                    i=simplices[:, 0],
                    j=simplices[:, 1],
                    k=simplices[:, 2],
                    color=colors[idx % len(colors)],
                    opacity=opacity,
                    name=f"Cell {cid}"
                ))

        # Plot puncta
        df = self.RNA_loc_df.copy()
        df = df[df['cell_id'].isin(cell_ids)]
        if gene_filter is not None:
            df = df[df['gene'].isin(gene_filter)]

        edit_mode = True
        if edit_mode:
            for i, gene in enumerate(df['gene'].unique()):
                for cell in cell_ids:
                    sub = df[(df['gene'] == gene) & (df['cell_id'] == cell)]
                    fig.add_trace(go.Scatter3d(
                        x=sub['X'], y=sub['Y'], z=sub['Z'],
                        mode='markers',
                        marker=dict(size=3, color=colors[i % len(colors)]),
                        name=f"{gene} (c={cell})"
                    ))
        else:
            for i, gene in enumerate(df['gene'].unique()):
                sub = df[df['gene'] == gene]
                fig.add_trace(go.Scatter3d(
                    x=sub['X'], y=sub['Y'], z=sub['Z'],
                    mode='markers',
                    marker=dict(size=3, color=colors[i % len(colors)]),
                    name=gene
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X (µm)',
                yaxis_title='Y (µm)',
                zaxis_title='Z (µm)'
            ),
            title=f"Cell(s) {', '.join(cell_ids)} with RNA Puncta",
            width=800, height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        fig.show()
