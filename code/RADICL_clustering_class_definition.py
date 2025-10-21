import pandas as pd
import numpy as np
import bioframe as bf
import hdbscan
import networkx as nx
import statsmodels.api as sm
import multiprocessing
from functools import partial
import scipy.stats as stats

class RADICL_read_tbl:
    def __init__(self, read_df,source_gene,anno_df) -> None:
        self.read_df = read_df
        self.DNA_read_df = None
        self.RNA_read_df = None
        self.source_gene_id = source_gene
        self.gene_source_df = None
        self.gene_source_chrom = None
        self.bg_df = None
        self.anno_df = anno_df
        self.source_read_ID = None
        self.bg_read_ID = None

    def get_read_tbl(self):
        print("Prepare read tables")
        tmp_chrom_read_df = (self.read_df
        .rename(columns = {0:'RNA_chrom',1:'RNA_start',2:'RNA_end',3:'DNA_chrom',4:'DNA_start',5:'DNA_end',6:'ID',8:'RNA_strand',9:'DNA_strand'})
        .loc[:,['RNA_chrom','RNA_start','RNA_end','DNA_chrom','DNA_start','DNA_end','RNA_strand','DNA_strand','ID']]
        )

        tmp_chrom_RNA_side_read_df = (tmp_chrom_read_df
        .loc[:,['RNA_chrom','RNA_start','RNA_end','RNA_strand','ID']]
        .rename(columns={"RNA_chrom":'chrom','RNA_start':'start','RNA_end':'end','RNA_strand':'strand'})
        )
        tmp_chrom_DNA_side_read_df = (tmp_chrom_read_df
        .loc[:,['DNA_chrom','DNA_start','DNA_end','ID']]
        .rename(columns={"DNA_chrom":'chrom','DNA_start':'start','DNA_end':'end'})
        )
        self.RNA_read_df = tmp_chrom_RNA_side_read_df
        self.DNA_read_df = tmp_chrom_DNA_side_read_df
    
    def get_source_and_bg_read_ID(self):
        self.gene_source_df = self.anno_df.query("gene_id == @self.source_gene_id")
        self.gene_source_chrom = self.gene_source_df.chrom.to_list()[0]
        self.bg_df = bf.merge(self.anno_df.query("gene_type == 'protein_coding'"),on=['strand'])
        self.source_read_ID = bf.overlap(self.RNA_read_df,self.gene_source_df,how='inner',on=['strand'],return_input=True).loc[:,'ID'].to_list()
        self.bg_read_ID = bf.overlap(self.RNA_read_df,self.bg_df,how='inner',on=['strand'],return_input=True).loc[:,'ID'].to_list()

class RADICL_cluster:
    def __init__(self,target_chrom,rep_RADICL_reads) -> None:
        self.rep_reads = rep_RADICL_reads
        self.source_chrom = rep_RADICL_reads.gene_source_chrom
        self.target_chrom = target_chrom
        self.source_chrom_gene_read_tbl = None
        self.target_chrom_gene_read_tbl = None
        self.target_chrom_bg_read_tbl = None
        self.source_chrom_label = None
        self.target_chrom_label = None
        self.hdb_graph = None
        self.hdb_summary_tbl = None
        self.res_tbl = None


    def produce_target_chromosome_read_tbls(self):
        self.target_chrom_gene_read_tbl = self.rep_reads.DNA_read_df.query("chrom == @self.target_chrom").query("ID in @self.rep_reads.source_read_ID")
        tmp_target_chrom_gene_read_ID_list = self.target_chrom_gene_read_tbl.ID.drop_duplicates().to_list()
        self.source_chrom_gene_read_tbl = self.rep_reads.RNA_read_df.query("ID in @tmp_target_chrom_gene_read_ID_list")
        ### Construct radicl bg
        bg_read_id_list = self.rep_reads.bg_read_ID
        self.target_chrom_bg_read_tbl = self.rep_reads.DNA_read_df.query("chrom == @self.target_chrom").query("ID in @bg_read_id_list")


    def collect_hdb_cluster_read(self):
        
        leaves = set([v for v, d in self.hdb_graph.out_degree() if d == 0])
        HDB_clusters = [v for v, d in self.hdb_graph.out_degree() if d > 0]

        cl_read_idx = [list(nx.descendants(self.hdb_graph,i).intersection(leaves)) for i in HDB_clusters]
        cl_read_tbl = pd.DataFrame({"HDB_cluster":HDB_clusters,"read_id_set":cl_read_idx})
        node_depth = nx.shortest_path_length(self.hdb_graph,source=cl_read_tbl.HDB_cluster.min())
        cl_read_tbl = (cl_read_tbl
                       .assign(lvl = lambda df_: [node_depth[i] for i in df_.HDB_cluster.to_list()])
                       .assign(norm_lvl = lambda df_: df_.lvl/df_.lvl.max()))

        return(cl_read_tbl)

    def build_radicl_zscore(self,vars,vars_df,vars_degree,target_var):

        x_spline = self.hdb_summary_tbl[vars].to_numpy(dtype=float)
        y = self.hdb_summary_tbl[target_var].to_numpy(dtype=float)
        bs = sm.gam.BSplines(x_spline, df=vars_df, degree=vars_degree)

        chr_gam = sm.GLMGam(y,smoother=bs)
        chr_gam_res = chr_gam.fit()
        gam_infl = chr_gam_res.get_influence()
        bs_tranform_exog = bs.transform(self.hdb_summary_tbl[vars].to_numpy())
        tmp_rng = chr_gam_res.get_distribution(exog=bs_tranform_exog)
        mod_pvalue = tmp_rng.sf(self.hdb_summary_tbl[target_var].to_numpy())
        new_data_tbl = self.hdb_summary_tbl.assign(zscore = gam_infl.resid_studentized,pvalue = mod_pvalue)
        self.res_tbl =  new_data_tbl

    def produce_HDBScan_cluster_tbl(self,min_clust_size,njobs):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_clust_size,
                            metric='euclidean',
                            core_dist_n_jobs=njobs)
        clusterer.fit(self.target_chrom_gene_read_tbl.loc[:,['start']])
        self.hdb_graph = clusterer.condensed_tree_.to_networkx()
        tmp_cl_tbl = self.collect_hdb_cluster_read()
        long_cl_read_tbl = tmp_cl_tbl.explode('read_id_set')

        long_cl_read_tbl = (long_cl_read_tbl
                            .assign(start = lambda df_:self.target_chrom_gene_read_tbl.reset_index().start.to_numpy()[df_.read_id_set.to_numpy(dtype=int)])
                            )

        tmp_chr_hdb_summary_tbl = (long_cl_read_tbl
                                        .groupby('HDB_cluster')
                                        .agg(start = ('start','min'),
                                            end=('start','max'),
                                            rc = ('start','count'),
                                            )
                                        .assign(w= lambda df_:df_.end - df_.start)
                                    )
        
        tmp_chr_hdb_summary_tbl = (tmp_chr_hdb_summary_tbl
                                    .assign(lrc = lambda df_: np.log10(df_.rc))
                                    )

        tmp_chr_hdb_summary_tbl = tmp_chr_hdb_summary_tbl.reset_index().merge(tmp_cl_tbl)
        tmp_chr_hdb_summary_tbl = bf.count_overlaps(tmp_chr_hdb_summary_tbl.assign(chrom =self.target_chrom),self.target_chrom_bg_read_tbl).rename(columns={'count':'bg_count'})
        self.hdb_summary_tbl = tmp_chr_hdb_summary_tbl.assign(lbg = lambda df_: np.log10(df_.bg_count)).assign(corrected_lbg = lambda df_: np.where(df_.bg_count.lt(1),0,df_.lbg))

def get_CvM_pvalue(i,tbl):
    return stats.cramervonmises_2samp(tbl.iloc[i].rep1_start,tbl.iloc[i].rep2_start).pvalue

def get_matched_cluster_agreement_pvalue(rep_match_tbl,rep1_target_chrom_clustering,rep2_target_chrom_clustering,njobs):
    tmp_rep1_read_start_tbl = (rep_match_tbl.loc[:,['HDB_cluster_rep1']]
    .drop_duplicates()
    .merge(rep1_target_chrom_clustering.hdb_summary_tbl.loc[:,['HDB_cluster','read_id_set']],left_on = ['HDB_cluster_rep1'],right_on = ['HDB_cluster'])
    .drop('HDB_cluster',axis=1)
    .assign(DNA_start = lambda df_: df_.apply(lambda x:rep1_target_chrom_clustering.target_chrom_gene_read_tbl.start.iloc[x.read_id_set].to_numpy(),axis=1))
    .drop('read_id_set',axis=1)
    .rename(columns={'DNA_start':"rep1_start"})
    )

    tmp_rep2_read_start_tbl = (rep_match_tbl.loc[:,['HDB_cluster_rep2']]
    .drop_duplicates()
    .merge(rep2_target_chrom_clustering.hdb_summary_tbl.loc[:,['HDB_cluster','read_id_set']],left_on = ['HDB_cluster_rep2'],right_on = ['HDB_cluster'])
    .drop('HDB_cluster',axis=1)
    .assign(DNA_start = lambda df_: df_.apply(lambda x:rep2_target_chrom_clustering.target_chrom_gene_read_tbl.start.iloc[x.read_id_set].to_numpy(),axis=1))
    .drop('read_id_set',axis=1)
    .rename(columns={'DNA_start':"rep2_start"})
    )

    rep_match_read_coord_tbl = (rep_match_tbl
    .loc[:,['HDB_cluster_rep1','HDB_cluster_rep2','jaccard']]
    .merge(tmp_rep1_read_start_tbl)
    .merge(tmp_rep2_read_start_tbl)
    )

    with multiprocessing.Pool(processes=njobs) as pool:
            # Using map_async method to perform square operation on all numbers parallely
            read_agreement_pvalue = pool.map(partial(get_CvM_pvalue, tbl = rep_match_read_coord_tbl),
                                            range(rep_match_read_coord_tbl.shape[0])) 
    return rep_match_read_coord_tbl.assign(cvm_pvalue = read_agreement_pvalue).loc[:,['HDB_cluster_rep1','HDB_cluster_rep2','jaccard','cvm_pvalue']]   

class Merged_cluster:
    def __init__(self,rep1_cluster,rep2_cluster) -> None:
        self.rep1 = rep1_cluster
        self.rep2 = rep2_cluster
        self.rep1_matching_tbl = None
        self.rep2_matching_tbl = None
        self.rep1_match_pvalue_tbl = None
        self.rep2_match_pvalue_tbl = None
        self.robust_cluster_match_tbl = None
        self.robust_cluster_coord_tbl = None
        self.inter_rep_match_tbl = None
    
    def match_cluster(self):

        rep_overlap_df = (bf.overlap(self.rep1.res_tbl.loc[:,['chrom','start','end','HDB_cluster']],
                                     self.rep2.res_tbl.loc[:,['chrom','start','end','HDB_cluster']],
                                     return_overlap=True,
                                     how='inner',suffixes=['_rep1','_rep2'])
        .assign(inter_w=lambda df_:(df_.overlap_end-df_.overlap_start),
                end_point=lambda df_:df_[["end_rep1","end_rep2"]].values.tolist(),
                start_points = lambda df_:df_[["start_rep1","start_rep2"]].values.tolist())
        .assign(jaccard = lambda df_:df_.inter_w/(df_.end_point.apply(max) - df_.start_points.apply(min)))
        .drop(['end_point','start_points'],axis=1)
        )

        rep1_max_jaccard_idx = (rep_overlap_df
                .groupby(['chrom_rep1','start_rep1','end_rep1','HDB_cluster_rep1'])
                .jaccard.idxmax()
                )

        rep2_max_jaccard_idx = (rep_overlap_df
                .groupby(['chrom_rep2','start_rep2','end_rep2','HDB_cluster_rep2'])
                .jaccard.idxmax()
                )

        self.rep1_matching_tbl = rep_overlap_df.iloc[rep1_max_jaccard_idx,:]
        self.rep2_matching_tbl = rep_overlap_df.iloc[rep2_max_jaccard_idx,:]
        
    def evaluate_overlap_significance(self,njobs):
        self.rep1_match_pvalue_tbl = get_matched_cluster_agreement_pvalue(self.rep1_matching_tbl,self.rep1,self.rep2,njobs)
        self.rep2_match_pvalue_tbl = get_matched_cluster_agreement_pvalue(self.rep2_matching_tbl,self.rep1,self.rep2,njobs)


    def filter_significant_overlap_cluster(self,CvM_thresh,specificity_thresh):
        
        rep1_matched_cluster_ID_list = self.rep1_match_pvalue_tbl.query("cvm_pvalue > @CvM_thresh").HDB_cluster_rep1.drop_duplicates().to_list()
        rep2_matched_cluster_ID_list = self.rep2_match_pvalue_tbl.query("cvm_pvalue > @CvM_thresh").HDB_cluster_rep2.drop_duplicates().to_list()

        rep1_robust_cluster_tbl = (self.rep1
                                    .res_tbl.loc[:,['HDB_cluster','chrom','start','end','w','zscore','pvalue']]
                                    .query("HDB_cluster in @rep1_matched_cluster_ID_list")
                                    .assign(rep = "rep1")
                                    .merge(self.rep1_match_pvalue_tbl,left_on='HDB_cluster',right_on="HDB_cluster_rep1")
                                    .drop('HDB_cluster',axis=1)
                                    .query("HDB_cluster_rep2 in @rep2_matched_cluster_ID_list")
                                    .merge(self.rep2.res_tbl.loc[:,['HDB_cluster','pvalue']].rename(columns={'HDB_cluster':'HDB_cluster_rep2','pvalue':'pvalue_rep2'}))
                                    .query('pvalue < @specificity_thresh and pvalue_rep2 < @specificity_thresh')
                                    .loc[:,['chrom','start','end','zscore','pvalue','rep','HDB_cluster_rep1','HDB_cluster_rep2','jaccard','cvm_pvalue']]
                                )
        rep2_robust_cluster_tbl = (self.rep2
                                    .res_tbl.loc[:,['HDB_cluster','chrom','start','end','w','zscore','pvalue']]
                                    .query("HDB_cluster in @rep2_matched_cluster_ID_list")
                                    .assign(rep = "rep2")
                                    .merge(self.rep2_match_pvalue_tbl,left_on='HDB_cluster',right_on="HDB_cluster_rep2")
                                    .drop('HDB_cluster',axis=1)
                                    .query("HDB_cluster_rep1 in @rep1_matched_cluster_ID_list")
                                    .merge(self.rep1.res_tbl.loc[:,['HDB_cluster','pvalue']].rename(columns={'HDB_cluster':'HDB_cluster_rep1','pvalue':'pvalue_rep1'}))
                                    .query('pvalue < @specificity_thresh and pvalue_rep1 < @specificity_thresh')
                                    .loc[:,['chrom','start','end','zscore','pvalue','rep','HDB_cluster_rep1','HDB_cluster_rep2','jaccard','cvm_pvalue']]
                                )
        if (rep1_robust_cluster_tbl.shape[0] > 0 and rep2_robust_cluster_tbl.shape[0]):
            self.robust_cluster_match_tbl = bf.cluster(pd.concat([rep1_robust_cluster_tbl.drop('HDB_cluster_rep2',axis=1).rename(columns={'HDB_cluster_rep1':'HDB_cluster'}),
                                                                  rep2_robust_cluster_tbl.drop('HDB_cluster_rep1',axis=1).rename(columns={'HDB_cluster_rep2':'HDB_cluster'})])).assign(w = lambda df_: df_.cluster_end -df_.start).sort_values('w')

            self.robust_cluster_coord_tbl = (self.robust_cluster_match_tbl
                            .loc[:,['chrom','cluster_start','cluster_end','cluster']]
                            .drop_duplicates()
                            .rename(columns = {'cluster_start':'start','cluster_end':"end"})
                            .assign(w = lambda df_: df_.end - df_.start)
                            )
            
            self.inter_rep_match_tbl = (pd.concat([rep2_robust_cluster_tbl
                                        .loc[:,['HDB_cluster_rep1','HDB_cluster_rep2','cvm_pvalue','jaccard']]
                                        .merge(self.rep1.res_tbl.loc[:,['HDB_cluster','chrom','start','end','zscore','pvalue','norm_lvl']]
                                               .rename(columns = {'start':'start_rep1','end':'end_rep1','pvalue':'pvalue_rep1','zscore':'zscore_rep1','norm_lvl':'norm_lvl_rep1'}),left_on = 'HDB_cluster_rep1',right_on='HDB_cluster').drop('HDB_cluster',axis=1)
                                        .merge(self.rep2.res_tbl.loc[:,['HDB_cluster','start','end','zscore','pvalue','norm_lvl']]
                                               .rename(columns = {'start':'start_rep2','end':'end_rep2','pvalue':'pvalue_rep2','zscore':'zscore_rep2','norm_lvl':'norm_lvl_rep2'}),left_on = 'HDB_cluster_rep2',right_on='HDB_cluster').drop('HDB_cluster',axis=1)
                                        ,
                                        rep1_robust_cluster_tbl
                                        .loc[:,['HDB_cluster_rep1','HDB_cluster_rep2','cvm_pvalue','jaccard']]
                                        .merge(self.rep1.res_tbl.loc[:,['HDB_cluster','chrom','start','end','zscore','pvalue','norm_lvl']]
                                               .rename(columns = {'start':'start_rep1','end':'end_rep1','pvalue':'pvalue_rep1','zscore':'zscore_rep1','norm_lvl':'norm_lvl_rep1'}),left_on = 'HDB_cluster_rep1',right_on='HDB_cluster').drop('HDB_cluster',axis=1)
                                        .merge(self.rep2.res_tbl.loc[:,['HDB_cluster','start','end','zscore','pvalue','norm_lvl']]
                                               .rename(columns = {'start':'start_rep2','end':'end_rep2','pvalue':'pvalue_rep2','zscore':'zscore_rep2','norm_lvl':'norm_lvl_rep2'}),left_on = 'HDB_cluster_rep2',right_on='HDB_cluster').drop('HDB_cluster',axis=1)
                                        ]).drop_duplicates()
                                        .assign(start = lambda df_: df_.apply(lambda row: min(row.start_rep1,row.start_rep2),axis=1),
                                                end = lambda df_: df_.apply(lambda row: max(row.end_rep1,row.end_rep2),axis=1))
                                        .assign(w = lambda df_: df_.end -df_.start))

            
        else:
            print("no significant and replicable cluster")
