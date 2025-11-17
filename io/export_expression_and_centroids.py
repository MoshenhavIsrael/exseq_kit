# דרישות: הקובץ sample_object.py שלך באותו נתיב, ויש לך אובייקט SampleObject טעון בשם 'samp'

import exseq_kit
import pandas as pd
import numpy as np
import os

def export_spagft_expression_and_coords(
    samp: SampleObject,
    out_folder: str,
    expr_filename: str = "expression_matrix.csv",
    coord_filename: str = "coordinates.csv",
    include_z: bool = True
):
    """
    שומר שני קבצים בפורמט ש-SpaGFT מצפה לו פחות או יותר:
    1) expression_matrix.csv: שורות = גנים, עמודות = תאים/ספוטים (IDs), ערכים = ספירות.
    2) coordinates.csv: columns = [spot_id, X, Y(, Z)] עם אותם מזהים בדיוק כפי שבמטריצת הביטוי.
    """

    os.makedirs(out_folder, exist_ok=True)

    # --- 1) מטריצת הביטוי ---
    # קיימת כבר ב- SampleObject.gene_cell_matrix (שורות=גנים, עמודות=cell_id)
    expr = samp.gene_cell_matrix.copy()
    # וידוא שאין עמודת '0' (רקע/ללא-תא) ושכל המזהים הם מחרוזות עקביות:
    expr.columns = expr.columns.astype(str)
    for bad in ['0', 0]:
        if bad in expr.columns:
            expr = expr.drop(columns=str(bad))

    # נשמור כ-CSV: index = gene symbols, columns = cell/spot IDs
    expr_out = os.path.join(out_folder, expr_filename)
    expr.to_csv(expr_out, index=True)

    # --- 2) קואורדינטות לכל מזהה עמודה ---
    # נשתמש בצנטרואידים לפני-כן מחושבים (או נחושב מידית מה-RNA_loc_df):
    centroids = getattr(samp, 'cell_centroids', None)
    if centroids is None or centroids is False:
        centroids = samp._create_cell_centroids()

    # יישור הזמנים: נשאיר רק את אותם מזהי תאים/ספוטים שמופיעים במטריצה (כדי להבטיח התאמה 1:1)
    centroids = centroids.copy()
    centroids['cell_id'] = centroids['cell_id'].astype(str)
    centroids = centroids[centroids['cell_id'].isin(expr.columns)]

    # סדר עמודות: spot_id, X, Y, (Z אופציונלי)
    cols = ['cell_id', 'X_centroid', 'Y_centroid']
    if include_z and 'Z_centroid' in centroids.columns:
        cols.append('Z_centroid')

    coords_df = centroids[cols].rename(columns={
        'cell_id': 'spot_id',
        'X_centroid': 'X',
        'Y_centroid': 'Y',
        'Z_centroid': 'Z'
    })

    # כדי להבטיח סדר זהה בין קבצים (לא חובה, אבל נעים)
    coords_df = coords_df.set_index('spot_id').loc[expr.columns].reset_index()

    coords_out = os.path.join(out_folder, coord_filename)
    coords_df.to_csv(coords_out, index=False)

    return expr_out, coords_out

# דוגמה לשימוש (התאם נתיבים/שמות קבצים):
samp_folder = r"C:\Users\Moshe\OneDrive\coding\SpatialGenomicsLab\SegmentedData\breast_cancer\880"
samp = SampleObject.from_files(samp_folder)
output_folder = r"C:\Users\Moshe\OneDrive\coding\SpatialGenomicsLab\SegmentedData\breast_cancer\880\exported_spagft"
expr_path, coord_path = export_spagft_expression_and_coords(
    samp,
    out_folder=output_folder,
    expr_filename="expression_matrix.csv",
    coord_filename="coordinates.csv",
    include_z=False

)
print(expr_path, coord_path)
