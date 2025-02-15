import gzip
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def regex_match(start_with, end_with, folder):
    m = re.search(start_with+end_with, folder)
    if m!=None:
        return True
    else:
        return False
    
def make_dir(dirname):
    if os.path.exists(dirname):
        print(f'Already Exists: {dirname}')
        return False
    else:
        os.makedirs(dirname)
        print(f'Created: {dirname}')
        return True



class VCFProcessor:
    def __init__(self, vcf_file_path, out_file_base, chunk_size=1_000_000):
        """
        Initialize the VCF Processor with input VCF file, output base name, and chunk size.
        """
        self.vcf_file_path = vcf_file_path
        self.out_file_base = out_file_base
        self.chunk_size = chunk_size
        self.file_list = []

    def stream_vcf_lines(self):
        """
        Generator that yields data rows (CHROM, POS, INFO) from a VCF file, skipping headers.
        """
        with gzip.open(self.vcf_file_path, "rt") as file_handle:
            for line in file_handle:
                if line.startswith('#'):
                    continue
                cols = line.strip().split('\t')
                yield cols[0], cols[1], cols[-1]  # CHROM, POS, INFO

    def build_df_chunk(self, batch):
        """
        Convert a batch of tuples (CHROM, POS, INFO) into a DataFrame with extracted AF values.
        """
        df_chunk = pd.DataFrame(batch, columns=["CHROM", "POS", "INFO"])
        df_chunk["AF"] = df_chunk["INFO"].str.extract(r'(?:^|;)AF=0([^;]+)')
        df_chunk["AF"] = pd.to_numeric(df_chunk["AF"], errors="coerce").astype('float32')
        df_chunk.dropna(subset=["AF"], inplace=True)
        df_chunk.drop(columns="INFO", inplace=True)

        df_chunk.rename(columns={"CHROM": "chrom", "POS": "pos"}, inplace=True)
        df_chunk["chrom"] = df_chunk["chrom"].str.replace(r"^chr", "", regex=True).astype(np.int16)
        df_chunk["pos"] = df_chunk["pos"].astype(int)

        return df_chunk

    def _write_parquet_chunk(self, df, chunk_id):
        """
        Write a DataFrame chunk to a Parquet file with gzip compression.
        """
        out_file_chunk = self._make_chunk_name(chunk_id)
        df.to_parquet(out_file_chunk, compression='gzip', engine='pyarrow')
        print(f"Wrote {out_file_chunk}")
        self.file_list.append(out_file_chunk)

    def _make_chunk_name(self, chunk_id):
        """
        Generate a chunk-specific parquet filename.
        """
        base, ext = os.path.splitext(self.out_file_base)
        base_gz, ext_gz = os.path.splitext(base)

        if ext_gz == '.gz':
            return f"{base_gz}_{chunk_id}{ext_gz}{ext}"
        return f"{base}_{chunk_id}{ext}"

    def combine_parquet_files(self):
        """
        Combine multiple Parquet chunk files into one final Parquet file.
        """
        print(f"Combining into {self.out_file_base}")
        tables = [pq.read_table(f) for f in self.file_list]
        combined = pa.concat_tables(tables)
        pq.write_table(combined, self.out_file_base, compression='gzip')
        print(f"Combined file written: {self.out_file_base}")

    def process_vcf(self):
        """
        Process the VCF file in chunks and convert it to a single compressed Parquet file.
        """
        batch = []
        chunk_id = 0

        for i, (chrom, pos, info) in enumerate(self.stream_vcf_lines()):
            batch.append((chrom, pos, info))
            if (i + 1) % self.chunk_size == 0:
                chunk_id += 1
                df_chunk = self.build_df_chunk(batch)
                self._write_parquet_chunk(df_chunk, chunk_id)
                batch.clear()

        # Handle leftover batch
        if batch:
            chunk_id += 1
            df_chunk = self.build_df_chunk(batch)
            self._write_parquet_chunk(df_chunk, chunk_id)
            batch.clear()

        # Combine chunk files
        self.combine_parquet_files()

        # Clean up temporary chunk files
        for f in self.file_list:
            os.remove(f)

# Example usage:
# processor = VCFProcessor("input.vcf.gz", "output.parquet.gz")
# processor.process_vcf()
