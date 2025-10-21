import pandas as pd
import numpy as np
import bioframe as bf
import RADICL_clustering_class_definition
from importlib import reload
reload(RADICL_clustering_class_definition)


# %%
tbl5_gene_anno_file = "/storage/mathelierarea/processed/vipin/group/annotations/hg38/table5pENSTpCAT.filtered.gtf"

gene_anno_df =pd.read_csv(tbl5_gene_anno_file,sep="\t",comment="#",header=None)

gene_clean_df = (gene_anno_df 
 .iloc[:,[0,2,3,4,6,8]]
  .assign(gene_id=lambda df_:df_.iloc[:,5].str.extract(r'gene_id "((?:[^"\\]|\\.)*)"'))
  .assign(gene_type=lambda df_:df_.iloc[:,5].str.extract(r'gene_type "((?:[^"\\]|\\.)*)"'))
  .assign(gene_name=lambda df_:df_.iloc[:,5].str.extract(r'gene_name "((?:[^"\\]|\\.)*)"'))
  .assign(transcript_id=lambda df_:df_.iloc[:,5].str.extract(r'transcript_id "((?:[^"\\]|\\.)*)"'))

  .loc[:,[0,2,3,4,6,'gene_id','gene_type','gene_name','transcript_id']]
  .rename(columns={
      0:'chrom',
      2:'set',
      3:'start',
      4:'end',
      6:'strand'
  })
 )
gene_df = gene_clean_df.query('set=="gene"')
bg_gene_coord_df = bf.merge(gene_df.query("gene_type == 'protein_coding'"),on=['strand'])
#%%

gene_of_interest_id = "ENSG00000245532.10"
test_data_rep1 = 'TBD'
test_data_rep2 = 'TBD'
rep1_read_folder = test_data_rep1
rep2_read_folder = test_data_rep2
gene_tbl = gene_df.query("gene_id == @gene_of_interest_id")
gene_chromo = gene_tbl.chrom.to_list()[0]
# %%
rep1_gene_chrom_read_tbl = pd.read_csv(f"{rep1_read_folder}{gene_chromo}.txt",sep="\t",header=None)
rep2_gene_chrom_read_tbl = pd.read_csv(f"{rep2_read_folder}{gene_chromo}.txt",sep="\t",header=None)


# %%
rep1_RADICL_reads = RADICL_clustering_class_definition.RADICL_read_tbl(rep1_gene_chrom_read_tbl,gene_of_interest_id,gene_df)
rep1_RADICL_reads.get_read_tbl()
rep1_RADICL_reads.get_source_and_bg_read_ID()

rep2_RADICL_reads = RADICL_clustering_class_definition.RADICL_read_tbl(rep2_gene_chrom_read_tbl,gene_of_interest_id,gene_df)
rep2_RADICL_reads.get_read_tbl()
rep2_RADICL_reads.get_source_and_bg_read_ID()
#%%

target_chrom = "chrX"
rep1_target_chrom_clustering = RADICL_clustering_class_definition.RADICL_cluster(target_chrom,rep1_RADICL_reads)
rep1_target_chrom_clustering.produce_target_chromosome_read_tbls()
rep1_target_chrom_clustering.produce_HDBScan_cluster_tbl(5,10)
rep1_target_chrom_clustering.build_radicl_zscore(['corrected_lbg'],[10],[3],'lrc')

rep2_target_chrom_clustering = RADICL_clustering_class_definition.RADICL_cluster(target_chrom,rep2_RADICL_reads)
rep2_target_chrom_clustering.produce_target_chromosome_read_tbls()
rep2_target_chrom_clustering.produce_HDBScan_cluster_tbl(5,10)
rep2_target_chrom_clustering.build_radicl_zscore(['corrected_lbg'],[10],[3],'lrc')
#%%
target_chrom_merged_cluster = RADICL_clustering_class_definition.Merged_cluster(rep1_target_chrom_clustering,rep2_target_chrom_clustering)
target_chrom_merged_cluster.match_cluster()
target_chrom_merged_cluster.evaluate_overlap_significance(10)
target_chrom_merged_cluster.filter_significant_overlap_cluster(0.5,0.5)


# %%
