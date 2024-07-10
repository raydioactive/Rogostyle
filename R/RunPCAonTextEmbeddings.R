RunPCAOnTextEmbeddings <- function(embeddings, npcs = 50, ...) {
  # Validate embeddings input
  if (!is.matrix(embeddings) && !is.data.frame(embeddings)) {
    stop("Embeddings must be a matrix or a dataframe")
  }

  # Convert dataframe to matrix if necessary
  if (is.data.frame(embeddings)) {
    embeddings <- as.matrix(embeddings)
  }

  # Check if embeddings have proper dimensions
  if (ncol(embeddings) < npcs) {
    warning("The number of PCs requested is greater than the number of features in embeddings. Adjusting npcs to the number of features.")
    npcs <- ncol(embeddings)
  }

  # Run PCA using the adapted Seurat function
  pca_result <- RunPCA.default(
    object = embeddings,
    npcs = npcs,
    ...
  )

  return(pca_result)
}


