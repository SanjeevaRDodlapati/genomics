
# # import concurrent.futures
# # from multiprocessing import Manager


# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import os
# import pandas as pd
# import numpy as np
# import time

# # Global variables to hold large data (initialized later)
# row_labels = None
# cln_var_df = None

# def init_globals(shared_row_labels, shared_cln_var_df):
#     """Initialize global variables in worker processes."""
#     global row_labels, cln_var_df
#     row_labels = shared_row_labels
#     cln_var_df = shared_cln_var_df

# def process_file(file_name, pred1_Path):
#     """
#     Reads a parquet file, merges with global cln_var_df,
#     calculates enrichment, and returns (sig_enrich, NOT_sig_enrich).
#     """
#     global row_labels, cln_var_df
#     import time
#     import traceback

#     start_time = time.time()
#     cell = '_'.join(file_name.split('_')[:-1])
    
#     try:
#         print(f"Processing {cell} ...")

#         # Read parquet file
#         data = pd.read_parquet(os.path.join(pred1_Path, file_name))

#         # Insert chromosome/position from row_labels
#         data.insert(0, "chrom", row_labels["chrom"])
#         data.insert(1, "pos", row_labels["pos"])

#         # Merge with cln_var_df
#         merged_df = pd.merge(cln_var_df, data, on=["chrom", "pos"], how="inner")

#         # Calculate enrichment
#         sig_clin_var_count = (merged_df.iloc[:, -1] >= 0.10).sum()
#         not_sig_clin_var_count = merged_df.shape[0] - sig_clin_var_count

#         sig_count = (data.iloc[:, -1] >= 0.10).sum()
#         not_sig_count = data.shape[0] - sig_count

#         sig_enrich = round(sig_clin_var_count * 100 / sig_count, 2) if sig_count > 0 else np.nan
#         not_sig_enrich = round(not_sig_clin_var_count * 100 / not_sig_count, 2) if not_sig_count > 0 else np.nan

#         elapsed = time.time() - start_time
#         print(f"{cell}: sig_enrich={sig_enrich}, NOT_sig_enrich={not_sig_enrich}, time={elapsed/60:.2f} min")

#         return (cell, sig_enrich, not_sig_enrich)

#     except MemoryError:
#         print(f"🚨 MEMORY ERROR while processing {file_name}!")
#         return (cell, np.nan, np.nan)

#     except Exception as e:
#         print(f"❌ Error processing {file_name}: {str(e)}")
#         traceback.print_exc()  # Prints full error traceback
#         return (cell, np.nan, np.nan)





# def main():
#     global row_labels, cln_var_df  # Declare global variables

#     group = 2
#     print(f'Beginning the enrichment analysis for group {group}')

#     # Read row_labels
#     row_labels_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/row_labels.parquet.gzip'
#     row_labels = pd.read_parquet(row_labels_path, columns=['chrom', 'pos'])
#     row_labels['chrom'] = row_labels['chrom'].astype(np.int8)
#     row_labels['pos'] = row_labels['pos'].astype(np.uint32)

#     # Read cln_var_df globally
#     cln_var_df = pd.read_parquet('/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/data/cln_var_df_wihDup.parquet.gzip', columns=['CHROM', 'POS'])
#     # cln_var_df.drop(columns=['name'], inplace=True)
#     cln_var_df.rename(columns={"CHROM": "chrom", "POS": "pos"}, inplace=True)
#     cln_var_df['chrom'] = cln_var_df['chrom'].astype(np.int8)
#     cln_var_df['pos'] = cln_var_df['pos'].astype(np.uint32)

#     print(f'Row labels for pred1: {row_labels.shape}')

#     # Prepare file list (EXCLUDE row_labels file)
#     pred1_Path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/'
#     files = sorted(f for f in os.listdir(pred1_Path) if f != "row_labels.parquet.gzip")

#     # Parallel processing with progress bar
#     results = []
#     start_all = time.time()

#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = {executor.submit(process_file, f, pred1_Path): f for f in files}
#         for future in tqdm(as_completed(futures), total=len(files), desc="Processing files", unit="file"):
#             results.append(future.result())

#     print("All files processed in {:.2f} minutes".format((time.time() - start_all) / 60))

#     # Collect results into a DataFrame
#     enrichment_array = np.array(results)
#     enrich_df = pd.DataFrame(enrichment_array, columns=['cell', 'sig', 'NOT_sig'])
#     output_path = '/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/GVE_peak/data/enrich_ana/'
#     enrich_df.to_csv(output_path+f'pred1_group{group}_enrichment_in_clinvar.csv', index=False)

#     print("Done.")




# if __name__ == "__main__":
#     main()

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import time

# Global variables to hold large data (initialized later)
row_labels = None
cln_var_df = None

def process_file(file_name, pred1_Path):
    """
    Reads a parquet file, merges with global cln_var_df,
    calculates enrichment, and returns (sig_enrich, NOT_sig_enrich).
    """
    global row_labels, cln_var_df
    import time
    import traceback

    start_time = time.time()
    cell = '_'.join(file_name.split('_')[:-1])
    
    try:
        print(f"Processing {cell} ...")

        # Read parquet file
        data = pd.read_parquet(os.path.join(pred1_Path, file_name))

        # Insert chromosome/position from row_labels
        data.insert(0, "chrom", row_labels["chrom"])
        data.insert(1, "pos", row_labels["pos"])

        # Merge with cln_var_df
        merged_df = pd.merge(cln_var_df, data, on=["chrom", "pos"], how="inner")

        # Calculate enrichment
        sig_clin_var_count = (merged_df.iloc[:, -1] >= 0.10).sum()
        not_sig_clin_var_count = merged_df.shape[0] - sig_clin_var_count

        sig_count = (data.iloc[:, -1] >= 0.10).sum()
        not_sig_count = data.shape[0] - sig_count

        sig_enrich = round(sig_clin_var_count * 100 / sig_count, 2) if sig_count > 0 else np.nan
        not_sig_enrich = round(not_sig_clin_var_count * 100 / not_sig_count, 2) if not_sig_count > 0 else np.nan

        elapsed = time.time() - start_time
        print(f"{cell}: sig_enrich={sig_enrich}, NOT_sig_enrich={not_sig_enrich}, time={elapsed/60:.2f} min")

        return (cell, sig_enrich, not_sig_enrich)

    except MemoryError:
        print(f"🚨 MEMORY ERROR while processing {file_name}!")
        return (cell, np.nan, np.nan)

    except Exception as e:
        print(f"❌ Error processing {file_name}: {str(e)}")
        traceback.print_exc()  # Prints full error traceback
        return (cell, np.nan, np.nan)


def main(group):
    global row_labels, cln_var_df  # Declare global variables

    print(f'Beginning the enrichment analysis for group {group}')

    # Read row_labels
    row_labels_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/row_labels.parquet.gzip'
    row_labels = pd.read_parquet(row_labels_path, columns=['chrom', 'pos'])
    row_labels['chrom'] = row_labels['chrom'].astype(np.int8)
    row_labels['pos'] = row_labels['pos'].astype(np.uint32)

    # Read cln_var_df globally
    cln_var_df = pd.read_parquet('/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/data/cln_var_df_wihDup.parquet.gzip', columns=['CHROM', 'POS'])
    cln_var_df.rename(columns={"CHROM": "chrom", "POS": "pos"}, inplace=True)
    cln_var_df['chrom'] = cln_var_df['chrom'].astype(np.int8)
    cln_var_df['pos'] = cln_var_df['pos'].astype(np.uint32)

    print(f'Row labels for pred1: {row_labels.shape}')

    # Prepare file list (EXCLUDE row_labels file)
    pred1_Path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/{group}/'
    files = sorted(f for f in os.listdir(pred1_Path) if f != "row_labels.parquet.gzip")

    # Parallel processing with progress bar
    results = []
    start_all = time.time()

    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     futures = {executor.submit(process_file, f, pred1_Path): f for f in files}
    #     for future in tqdm(as_completed(futures), total=len(files), desc="Processing files", unit="file"):
    #         results.append(future.result())

    # Alternative code to automatically restart failed processes
    with ProcessPoolExecutor(max_workers=2) as executor:
        while len(files) > 0:
            futures = {executor.submit(process_file, f, pred1_Path): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Processing files", unit="file"):
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    print(f"⚠️ Retrying {futures[future]} due to failure...")
                    files.append(futures[future])  # Retry failed files


    print("All files processed in {:.2f} minutes".format((time.time() - start_all) / 60))

    # Collect results into a DataFrame
    enrichment_array = np.array(results)
    enrich_df = pd.DataFrame(enrichment_array, columns=['cell', 'sig', 'NOT_sig'])
    output_path = '/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/GVE_peak/data/enrich_ana/'
    enrich_df.to_csv(output_path+f'pred1_group{group}_enrichment_in_clinvar.csv', index=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enrichment analysis for a specified group number.")
    parser.add_argument("group", type=int, help="Group number for the enrichment analysis.")
    args = parser.parse_args()

    main(args.group)
