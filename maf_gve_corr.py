"""
Author: Sanjeeva Reddy Dodlapati
"""

import os
import re
import numpy as np
import pandas as pd
import time
import argparse

def rename_cells(cell_names):  
    """Cleans up cell names by splitting on '|' and removing '/' characters."""
    cells = []
    for cell in cell_names:
        splits = cell.split('|')
        cell = '_'.join(splits[:])
        cell = cell.replace('/', '')
        cells.append(cell)
    return cells

def main(args):
    group = args.group

    # --------------------------
    # Reading Minor Allele Frequency (MAF) data
    # --------------------------
    start_time = time.time()
    maf_dir = '/scratch/ml-csm/projects/fgenom/gve/data/human/MAF_chrom/processed/'
    maf_files = os.listdir(maf_dir)
    if '.ipynb_checkpoints' in maf_files:
        maf_files.remove('.ipynb_checkpoints')
    maf_files.sort()
    print(f'Number of MAF files to process: {len(maf_files)}')

    data_maf = pd.concat(
        [pd.read_parquet(os.path.join(maf_dir, file)) for file in maf_files],
        axis=0, ignore_index=True
    )
    end_time = time.time()
    print(f'Time to read MAF data: {(end_time - start_time)/60:.2f} minutes')

    # --------------------------
    # Reading row labels for one-time prediction data
    # --------------------------
    start_time = time.time()
    row_labels_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/row_labels.parquet.gzip'
    row_labels = pd.read_parquet(row_labels_path, columns=['chrom', 'pos'])
    row_labels['chrom'] = row_labels['chrom'].astype(np.int8)
    row_labels['pos'] = row_labels['pos'].astype(np.uint32)
    print(f'Row labels for pred1: {row_labels.shape}')
    end_time = time.time()
    print(f'Time to read row labels: {(end_time - start_time)/60:.2f} minutes')

    # --------------------------
    # Setting up paths for one-time and 150 prediction data
    # --------------------------
    pred1_path = os.path.join('/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/', str(group)) + os.sep
    pred150_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/mult_pred/top5M/{group}/pred150/'

    # Get list of 150 prediction files ending with _gve.tsv
    files_pred150 = [file for file in os.listdir(pred150_path) if re.match(r".*_gve\.tsv$", file)]
    files_pred150.sort()

    # Get columns from one of the 150 prediction files
    first_file = os.path.join(pred150_path, files_pred150[0])
    cols = list(pd.read_csv(first_file, sep='\t', nrows=0).columns)
    print(f'Number of columns in prediction file: {len(cols)}')

    # --------------------------
    # Calculate correlations for selected columns
    # --------------------------
    corr_dict = {}
    # For demonstration, we use only a subset of columns (e.g., columns 3 to 5)
    for col in cols[3:]:
        start_loop = time.time()
        cell = rename_cells([col])[0]
        
        # Read 150 predictions for the current cell from all relevant files
        try:
            data150 = pd.concat(
                [pd.read_csv(os.path.join(pred150_path, file), sep='\t', usecols=['chrom', 'pos', col])
                 for file in files_pred150],
                axis=0, ignore_index=True
            )
        except Exception as ex:
            print(f"Error reading 150-pred files for {cell}: {ex}")
            continue
        
        # Clean column names and convert data types
        data150.columns = rename_cells(data150.columns)
        data150["chrom"] = data150["chrom"].astype(np.int16)
        data150["pos"] = data150["pos"].astype(int)
        merged_df150 = pd.merge(data150, data_maf, on=["chrom", "pos"], how="inner")
        del data150
        
        try:
            corr150 = np.corrcoef(merged_df150[cell].abs().values, merged_df150['AF'].values)
        except Exception as ex:
            # If the values need flattening
            corr150 = np.corrcoef(np.concatenate(merged_df150[cell].abs().values), merged_df150['AF'].values)
            print(f'Had to concatenate gve column values for {cell}: {ex}')
        
        # Read one-time prediction data for the current cell
        file1 = cell + '_gve.parquet.gzip'
        try:
            data1 = pd.read_parquet(os.path.join(pred1_path, file1))
        except Exception as ex:
            print(f"Error reading one-time prediction file for {cell}: {ex}")
            continue
        
        # Insert row labels and filter based on threshold
        data1.insert(0, "chrom", row_labels["chrom"])
        data1.insert(1, "pos", row_labels["pos"])
        data1 = data1[data1[cell].abs() > 0.10].sort_values(by=cell, key=lambda x: x.abs(), ascending=False)
        merged_df1 = pd.merge(data1, data_maf, on=["chrom", "pos"], how="inner")
        del data1
        print(f'{cell}, size of pred1 data: {merged_df1.shape[0]}, size of pred150: {merged_df150.shape[0]}')
        
        try:
            corr1 = np.corrcoef(merged_df1[cell].abs().values, merged_df1['AF'].values)
        except Exception as ex:
            corr1 = np.corrcoef(np.concatenate(merged_df1[cell].abs().values), merged_df1['AF'].values)
            print(f'Had to concatenate gve column values for {cell}: {ex}')
            
        corr_dict[cell] = [corr1[0, 1], corr150[0, 1]]
        
        end_loop = time.time()
        print(f'corr1: {corr1[0, 1]}, corr150: {corr150[0, 1]}, Time: {(end_loop - start_loop)/60:.2f} minutes\n')

    # Save correlation results to CSV
    corr_df = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['pred1', 'pred150'])
    output_csv = os.path.join(
        '/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/GVE_peak/data/enrich_ana/',
        f'gve_maf_corr_gp{group}.csv'
    )
    corr_df.to_csv(output_csv)
    print(f'Results saved to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate PCC correlation between gve and MAF")
    parser.add_argument('--group', type=int, default=3, required=True, help='Group number')
    args = parser.parse_args()
    main(args)
