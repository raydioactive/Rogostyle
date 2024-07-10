#PSEUDOTIME
library(monocle3)
library(SeuratDisk)
library(SeuratData)
library(Seurat)
cell_metadata <- data.frame(
  sentence = result_df_combined$sentence,
  genre = result_df_combined$genre,
  row.names = paste0("Sentence", seq_along(result_df_combined$sentence))
)
gene_annotation <-data.frame(gene_short_name = paste("position", 1:768, sep = "_"))
str(cds)
embedding_positive_mat<-embedding_matrix-(min(embedding_matrix))
cds <- new_cell_data_set(t(embedding_positive_mat),
                         cell_metadata = cell_metadata,
                         gene_metadata=gene_annotation)
rownames(cds)<-rowData(cds)$gene_short_name
rowData(cds)$gene_name <- rownames(cds)
rowData(cds)$gene_short_name <- rowData(cds)$gene_name
cds <- preprocess_cds(cds, num_dim = 100)
cds <- reduce_dimension(cds)
cds <- cluster_cells(cds)
cds <- learn_graph(cds)
cds <- order_cells(cds)

cds_Stylo <- new_cell_data_set((df_genemat),
                         cell_metadata = cell_metadata)
rowData(cds_Stylo)$gene_name <- rownames(cds_Stylo)
rowData(cds_Stylo)$gene_short_name <- rowData(cds_Stylo)$gene_name
cds_Stylo <- preprocess_cds(cds_Stylo, num_dim = 100)
cds_Stylo <- reduce_dimension(cds_Stylo, reduction_method = "PCA")
cds_Stylo <- cluster_cells(cds_Stylo)
cds_Stylo <- learn_graph(cds_Stylo)
cds_Stylo <- order_cells(cds_Stylo)
plot_cells(cds_Stylo, color_cells_by = "genre", cell_size = 1,reduction_method = "PCA")
plot_cells(cds_Stylo, color_cells_by = "pseudotime",cell_size = 1,)
stylo_graph_test_res <- graph_test(cds_Stylo, neighbor_graph="knn", cores=8)
plot_cells(cds_Stylo, genes="gerund",
           show_trajectory_graph=FALSE,
           label_cell_groups=TRUE,
           label_leaves=FALSE)
plot_cells(cds_Stylo, genes="personal_pronoun",
           show_trajectory_graph=FALSE,
           label_cell_groups=TRUE,
           label_leaves=FALSE)
plot_cells(cds_Stylo, genes="reflexive_pronoun",
           show_trajectory_graph=FALSE,
           label_cell_groups=TRUE,
           label_leaves=FALSE)
plot_cells(cds_Stylo, genes="superlatives",
           show_trajectory_graph=FALSE,
           label_cell_groups=TRUE,
           label_leaves=FALSE)
plot_cells(cds_Stylo, genes="conditional_clauses",
           show_trajectory_graph=FALSE,
           label_cell_groups=FALSE,
           label_leaves=FALSE)


plot_cells(cds, color_cells_by = "genre", cell_size = 1)
plot_cells(cds, color_cells_by = "pseudotime",cell_size = 1)
##MORANS
pr_graph_test_res <- graph_test(cds, neighbor_graph="knn", cores=8)
pr_deg_ids <- row.names(subset(pr_graph_test_res, q_value < 0.05))
##
gene_module_df <- find_gene_modules(cds[pr_deg_ids,], resolution=1e-2)
  cell_group_df <- tibble::tibble(cell=row.names(colData(cds)),
                                cell_group=partitions(cds)[colnames(cds)])
agg_mat <- aggregate_gene_expression(cds, gene_module_df, cell_group_df)
row.names(agg_mat) <- stringr::str_c("Module ", row.names(agg_mat))
colnames(agg_mat) <- stringr::str_c("Partition ", colnames(agg_mat))

pheatmap::pheatmap(agg_mat, cluster_rows=TRUE, cluster_cols=TRUE,
                   scale="column", clustering_method="ward.D2",
                   fontsize=6)
library(dplyr)
plot_cells(cds,
           genes=gene_module_df %>% filter(module %in% c(1,2,3,4)),
           group_cells_by="partition",
           color_cells_by="partition",
           show_trajectory_graph=FALSE)
plot_cells(cds, genes=c("position_18", "position_55", "position_302", "position_207"),
           show_trajectory_graph=FALSE,
           label_cell_groups=FALSE,
           label_leaves=FALSE)
