import pyarrow.parquet as pq
pf = pq.ParquetFile('data/bronze/statcast_2024-04-01_2024-04-14.parquet')
print('columns:', [f.name for f in pf.schema])
