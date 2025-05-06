import os
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import time
import argparse

def group_files(files):
    """
    Categorizes a list of file names into different uncertainty groups and extracts unique cell names.

    Parameters:
    -----------
    files : list of str
        A list of file names formatted as "{cell_name}_{uncertainty_level}.parquet.gzip".

    Returns:
    --------
    unique_cells : list of str
        A sorted list of unique cell names extracted from file names.
    
    file_groups : OrderedDict
        A dictionary where keys are uncertainty categories (`highCert`, `cert`, `uncert`, `highUncert`)
        and values are lists containing file names belonging to each category.
    """
    
    # Initialize an OrderedDict with uncertainty categories
    file_groups = OrderedDict([
        ("highCert", []),
        ("cert", []),
        ("uncert", []),
        ("highUncert", [])
    ])   
    
    unique_cells = set()

    # Split each file name at the last underscore
    for file in files:
        parts = file.rsplit('_', 1)  # Splitting only at the 2nd to the last underscore
        if len(parts) != 2:
            continue  # Skip files that do not match expected pattern

        cell, uncertainty_with_ext = parts
        # Remove extension from the uncertainty part
        uncertainty = uncertainty_with_ext.split('.')[0]
        unique_cells.add(cell)

        # Categorize based on the uncertainty level
        if "highCert" in uncertainty:
            file_groups["highCert"].append(file)
        elif "highUncert" in uncertainty:
            file_groups["highUncert"].append(file)
        elif "cert" in uncertainty:
            file_groups["cert"].append(file)
        elif "uncert" in uncertainty:
            file_groups["uncert"].append(file)

    return sorted(unique_cells), file_groups


def read_uncert_data(input_dir: str, cell: str):
    """
    Reads parquet data files for a given cell line and categorizes them based on uncertainty levels.

    Parameters:
    -----------
    input_dir : str
        Directory containing the parquet files (should end with a slash).
    cell : str
        Cell line name (used as a prefix to locate files).

    Returns:
    --------
    df_cert : pd.DataFrame
        DataFrame containing high-certainty variant predictions ("highCert" & "cert").
    df_uncert : pd.DataFrame
        DataFrame containing uncertain predictions ("uncert" & "highUncert").
    """
    
    uncertainty_levels = ['_highCert', '_cert', '_uncert', '_highUncert']
    dataframes = []

    for level in uncertainty_levels:
        file_path = f"{input_dir}{cell}{level}.parquet.gzip"
        # Uncomment the next line if you need additional columns:
        # data = pd.read_parquet(file_path, columns=['chrom', 'pos', 'name', 'gve', 'pval'])
        data = pd.read_parquet(file_path, columns=['chrom', 'pos'])
        data['chrom'] = data['chrom'].astype(np.int8)
        data['pos'] = data['pos'].astype(np.uint32)
        dataframes.append(data)

    # Concatenate dataframes for high-certainty and uncertain groups
    df_cert = pd.concat(dataframes[:2], axis=0)    # "highCert" & "cert"
    df_uncert = pd.concat(dataframes[2:], axis=0)    # "uncert" & "highUncert"

    # Sample the larger DataFrame so that both groups have the same number of rows
    if df_cert.shape[0] > df_uncert.shape[0]:
        df_cert = df_cert.sample(n=df_uncert.shape[0], ignore_index=True)
    elif df_uncert.shape[0] > df_cert.shape[0]:
        df_uncert = df_uncert.sample(n=df_cert.shape[0], ignore_index=True)

    return df_cert, df_uncert


def get_clinvar_enrich(cells, pred1_path, inPath, level, row_labels, cln_var_df, thr):
    """
    Computes enrichment by merging prediction data with a ClinVar variants DataFrame.

    Parameters:
    -----------
    cells : list of str
        List of cell names to process.
    pred1_path : str
        Directory path for pred1 files.
    inPath : str
        Directory path for uncertain prediction files.
    level : str
        Uncertainty level suffix (e.g., '_highCert') to select the proper file.
    row_labels : pd.DataFrame
        DataFrame containing row labels with 'chrom' and 'pos' columns.
    cln_var_df : pd.DataFrame
        DataFrame containing ClinVar variant information with columns "chrom" and "pos".
    thr : float
        Threshold for filtering prediction values.

    Returns:
    --------
    enrich_df : pd.DataFrame
        DataFrame with enrichment counts from two sets of predictions.
    """
    results = []

    for cell in cells:
        try:
            start_time = time.time()
            file1 = cell + '_gve.parquet.gzip'
            # Read the pred1 file for the current cell
            data1 = pd.read_parquet(os.path.join(pred1_path, file1))
            
            # Insert row labels for merging
            data1.insert(0, "chrom", row_labels["chrom"])
            data1.insert(1, "pos", row_labels["pos"])
            
            # Filter rows where the absolute value in the column named after the cell exceeds the threshold
            data1 = data1[data1[cell].abs() > thr].sort_values(by=cell, key=lambda x: x.abs(), ascending=False)
            
            # Read the uncertain prediction file for the current cell and uncertainty level
            file_path = os.path.join(inPath, f"{cell}{level}.parquet.gzip")
            data2 = pd.read_parquet(file_path, columns=['chrom', 'pos', 'gve', 'pval'])
            data2['chrom'] = data2['chrom'].astype(np.int8)
            data2['pos'] = data2['pos'].astype(np.uint32)
            
            # Adjust data1 to have the same number of rows as data2
            data1 = data1.iloc[:data2.shape[0], :]
            data1.reset_index(drop=True, inplace=True)
            
            # Merge with ClinVar data on 'chrom' and 'pos'
            merged_df1 = pd.merge(cln_var_df, data1, on=["chrom", "pos"], how="inner")
            merged_df2 = pd.merge(cln_var_df, data2, on=["chrom", "pos"], how="inner")
            enrich1 = merged_df1.shape[0]
            enrich2 = merged_df2.shape[0]
            
            results.append([enrich1, enrich2])
            print(f"{cell} -> pred1 enrichment: {enrich1}, pred150 enrichment: {enrich2} (time: {(time.time()-start_time)/60:.2f} min)")
        except Exception as e:
            print(f"Error processing cell {cell}: {e}")
    
    results = np.array(results)
    enrich_df = pd.DataFrame(results, columns=['pred1', 'pred150'])
    enrich_df.index = cells
    return enrich_df


def main(group, thr, limit):
    """
    Main function to run ClinVar enrichment analysis.

    Parameters:
    -----------
    group : int
        Group number to determine file paths.
    thr : float
        Threshold for filtering prediction values.
    limit : int or None
        If provided, limits the processing to only the first 'limit' cells.
    """
    # Load ClinVar variant data and convert column types
    cln_var_df_path = '/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/data/cln_var_df_wihDup.parquet.gzip'
    cln_var_df = pd.read_parquet(cln_var_df_path, columns=['CHROM', 'POS'])
    cln_var_df = cln_var_df.rename(columns={"CHROM": "chrom", "POS": "pos"})
    cln_var_df['chrom'] = cln_var_df['chrom'].astype(np.int8)
    cln_var_df['pos'] = cln_var_df['pos'].astype(np.uint32)

    # Load row labels for pred1
    row_labels_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/row_labels.parquet.gzip'
    row_labels = pd.read_parquet(row_labels_path, columns=['chrom', 'pos'])
    row_labels['chrom'] = row_labels['chrom'].astype(np.int8)
    row_labels['pos'] = row_labels['pos'].astype(np.uint32)
    print(f'Row labels for pred1: {row_labels.shape}')

    # Define directories for predictions
    pred1_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/'
    pred1_files = os.listdir(pred1_path)
    
    inPath = f'/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/data/kmeans/uncert_gve_direction/{group}/pred200/'
    files = os.listdir(inPath)
    
    # Find the common set of cells from both directories
    cells1, _ = group_files(pred1_files)
    cells2, _ = group_files(files)
    cells = list(set(cells1) & set(cells2))
    print(f'Number of profiles: {len(cells)}')
    
    # If a limit is provided, only process the first 'limit' cells
    if limit is not None:
        cells = cells[:limit]
        print(f'Processing only {len(cells)} cells as limited by --limit parameter.')
    else:
        print(f'Number of profiles: {len(cells)}')
    
    # Define uncertainty level to use for enrichment (can be modified as needed)
    level = '_highCert'
    
    # Calculate ClinVar enrichment
    enrich_df = get_clinvar_enrich(cells, pred1_path, inPath, level, row_labels, cln_var_df, thr)
    
    # Save the enrichment DataFrame to a CSV file
    output_csv = f'/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/GVE_peak/data/enrich_ana/group{group}_clinVar_enrich.csv'
    enrich_df.to_csv(output_csv, index=False)
    print('Done! Results saved to', output_csv)


if __name__ == '__main__':
    # Parse command-line arguments for group, threshold, and optional limit
    parser = argparse.ArgumentParser(description="ClinVar Enrichment Analysis")
    parser.add_argument('--group', type=int, required=True, help="Group number (e.g., 2)")
    parser.add_argument('--thr', type=float, required=True, help="Threshold for filtering predictions (e.g., 0.1)")
    parser.add_argument('--limit', type=int, default=None, help="Optional: Process only the first N cells for testing")
    args = parser.parse_args()
    
    main(args.group, args.thr, args.limit)
