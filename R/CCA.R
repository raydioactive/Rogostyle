library(CCA)

# Extract the CLS embeddings and convert to a matrix
cls_embeddings_matrix <- embedding_matrix

# Convert the stylometry embeddings to a matrix
stylometry_embeddings_matrix <- t(sapply(result_df_combined$stylometry_embedding, function(x) unlist(x)))

# Get the number of dimensions in the stylometry embedding
num_stylometry_dimensions <- ncol(stylometry_embeddings_matrix)

# Perform PCA on CLS embeddings to reduce dimensionality
cls_pca <- prcomp(cls_embeddings_matrix, scale. = TRUE)
cls_pca_scores <- cls_pca$x[, 1:num_stylometry_dimensions]

# Perform regularized CCA on the PCA scores and stylometry embeddings
cca_result <- rcc(cls_pca_scores, stylometry_embeddings_matrix, lambda1 = 0.1, lambda2 = 0.1)

# Print the summary of regularized CCA results
summary(cca_result)

# Get the canonical correlations
canonical_correlations <- cca_result$cor

# Get the canonical variates (linear combinations)
cls_variates <- cca_result$xcoef
stylometry_variates <- cca_result$ycoef

# Determine the number of significant canonical correlations
num_significant <- sum(canonical_correlations > 0.01)  # Adjust the threshold as needed

# Print the significant canonical correlations and their corresponding variates
for (i in 1:num_significant) {
  cat("Canonical Correlation", i, ":", canonical_correlations[i], "\n")
  cat("CLS Embedding Variate", i, ":\n")
  print(cls_variates[, i])
  cat("Stylometry Embedding Variate", i, ":\n")
  print(stylometry_variates[, i])
  cat("\n")
}
