# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions
## Covariance based DMR calling

Scripts for covariance differential methylation calling, currently setup for using .beta files although support for coverage or bedgraph files can be added in future. I would also like to included Methrix object intergratition, this should be hopefully easy as meta data and files are all included in a h5ad array.

Annotation based on rank and difference between samples. Other methods to be developped

working data in : 
- wgbs : /rds/user/tm739/rds-massie-nutgKVmnlpI/Deconvolution/Toby/atlas/Prostate_Atlas_hg38
- targeted : /home/tm739/rds/rds-massie-nutgKVmnlpI/Deconvolution/Toby/wgbstools_data/ICGC/pilot

todo list:

- Methrix object input, modify main function to take it as an input.
- Bed and Cov input
- Annotations through R (functions already written but need to create rpy2 versions)
- Multi-Omic intergration. Allow additional data sources to be overlapped with data for higher convidence regions, currently scRNA-seq has been tested (do this in R as current? or move to python?)
- Network propagation for DMRs? Would be interesting to see if this can be used to refine regions futher. (probably more useful in Tum vs Ben comps)

<img width="1360" height="1010" alt="GSTP1_plot" src="https://github.com/user-attachments/assets/b733b5a6-230c-454d-9a79-5037cefa1803" />

