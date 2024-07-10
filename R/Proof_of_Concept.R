#StoreEmbeddings
library(reticulate)
Doc<-process_text_with_cltk("~/Arma.txt")
#
SpeechSents<-Doc$sentences
#separate embeddings
SpeechSentEmbeds<-Doc$sentence_embeddings
#Create Data Frame with the String, Embedding and Genre
Sentence_DF<-data.frame(Sent = Doc$sentences_strings, Embeddings = I(SpeechSentEmbeds), genre = c("Speech","Speech","Speech","Speech","Speech","Speech","Speech","Speech","Speech","Speech","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Letter","Philosophy","Philosophy","Philosophy","Philosophy","Philosophy","Philosophy","Philosophy"))
# Create the embedding matrix
embedding_matrix <- do.call(rbind, SpeechSentEmbeds)
#Calculate Principal components of variance in embedding matrix
pca_results <- prcomp(embedding_matrix, rank. = 50)
#PlotPCA
plot(pca_results$x[,1:2], col = as.factor(Sentence_DF$genre), pch = 19)
legend("topleft", legend = unique(Sentence_DF$genre), col = 1:length(unique(Sentence_DF$genre)), pch = 19)
## VARIANCE EXPLAINED
# Load necessary library
library(ggplot2)

# Extract the variance explained by each principal component
variance_explained <- pca_results$sdev^2
total_variance <- sum(variance_explained)
variance_explained_percent <- variance_explained / total_variance * 100

# Create a data frame for plotting
pca_variance_df <- data.frame(PC = 1:length(variance_explained_percent),
                              Variance = variance_explained_percent)

# Create the plot
ggplot(pca_variance_df, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "blue") +
  theme_minimal() +
  xlab("Principal Component") +
  ylab("Percentage of Variance Explained") +
  ggtitle("Variance Explained by Each Principal Component")
#UMAP
library(uwot)
set.seed(123) # For reproducibility
umap_results <- umap(embedding_matrix,, n_neighbors = 2, learning_rate = 0.5)
plot(umap_results[,1], umap_results[,2], col = as.factor(Sentence_DF$genre), pch = 19, xlab = "UMAP1", ylab = "UMAP2")

legend("top", legend = unique(Sentence_DF$genre), col = 1:length(unique(Sentence_DF$genre)), pch = 19)

