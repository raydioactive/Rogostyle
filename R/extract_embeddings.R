library(reticulate)
use_condaenv("cltk", required = TRUE)
###
Extract_text_embeddings_to_df_cltk_nogenre <- function(file_path) {
  # Read the text file
  text_data <- readLines(file_path)

  # Concatenate the lines into a single string
  text_data <- paste(text_data, collapse = " ")

  # Define a Python function that uses CLTK
  analyze_text <- py_run_string("
def analyze_text_with_cltk(text):
    from cltk import NLP
    cltk_nlp = NLP(language='lat')
    cltk_nlp.pipeline.processes.pop(-1) # Pop last two processes as per the original requirement
    cltk_nlp.pipeline.processes.pop(-1)
    doc = cltk_nlp.analyze(text)
    return doc
  ")

  # Call the Python function from R and store the result
  cltk_doc <- analyze_text$analyze_text_with_cltk(text_data)

  # Extract sentences and embeddings
  sentences <- cltk_doc$sentences_strings
  embeddings <- cltk_doc$sentence_embeddings

  # Create a dataframe
  data_df <- data.frame(
    sentence = sentences,
    embedding = I(embeddings) # Use I() to store list in dataframe
  )

  return(data_df)
}

letterembeddingsdf<-Extract_text_embeddings_to_df_cltk(file_path = "~/Rogostyle/texts/333kletterchars.txt",
                                                       genre = "Letters")
letterselected_rows_1551<- sample(nrow(letterembeddingsdf), 1551)
letterselected_rows_100<- sample(nrow(letterembeddingsdf), 500)
letterembeddingsdf_100 <- letterembeddingsdf[letterselected_rows_100, ]

length(letterembeddingsdf$sentence)

speechembeddingsdf<-Extract_text_embeddings_to_df_cltk(file_path = "~/Rogostyle/texts/333kspeechchars.txt",
                                                       genre = "Speeches")
speechselected_rows_100<- sample(nrow(speechembeddingsdf), 500)
speechembeddingsdf_100 <- speechembeddingsdf[speechselected_rows_100, ]

length(speechembeddingsdf$sentence)

philembeddingsdf<-Extract_text_embeddings_to_df_cltk(file_path = "~/Rogostyle/texts/333kphilchars.txt",
                                                       genre = "Philosophies")
philelected_rows_1551<- sample(nrow(philembeddingsdf), 1551)
philembeddingsdf_1551 <- philembeddingsdf[philelected_rows_1551, ]
philelected_rows_100<- sample(nrow(philembeddingsdf), 500)
philembeddingsdf_100 <- philembeddingsdf[philelected_rows_100, ]

length(philembeddingsdf$sentence)
allinonedf<-Extract_text_embeddings_to_df_cltk_nogenre(file_path = "~/Rogostyle/texts/999kchars.txt")
allinonedf$genre<-c(rep("letters", length(letterembeddingsdf$sentence)),rep("speeches", length(speechembeddingsdf$sentence)),rep("philosophies", length(philembeddingsdf$sentence)))
merged_df<-rbind(letterembeddingsdf_1551,speechembeddingsdf,philembeddingsdf_1551)
merged_df_100<-rbind(letterembeddingsdf_100,speechembeddingsdf_100,philembeddingsdf_100)
merged_df_100 <- subset(merged_df_100, sentence != ".")
###

embedding_matrix <- do.call(rbind, merged_df_100$embedding)
#Calculate Principal components of variance in embedding matrix
pca_results <- prcomp(embedding_matrix, rank. = 100)
#PlotPCA
plot(pca_results$x[,1:2], col = as.factor(merged_df_100$genre), pch = 19)
legend("bottomleft", legend = unique(merged_df_100$genre), col = 1:length(unique(merged_df_100$genre)), pch = 19)
###
library(plotly)
plot_df <- data.frame(PC1 = pca_results$x[,1],
                      PC2 = pca_results$x[,2],
                      Sentence = merged_df_100$sentence, # Replace with the name of your sentence column
                      Genre = as.factor(merged_df_100$genre))

# Create the Plotly plot
plot_ly(plot_df, x = ~PC1, y = ~PC2, type = 'scatter', mode = 'markers',
        text = ~Sentence, # Display the sentence on hover
        color = ~Genre, # Color by genre
        marker = list(size = 10)) %>%
  layout(title = 'PCA Plot',
         xaxis = list(title = 'PC1'),
         yaxis = list(title = 'PC2'),
         hovermode = 'closest')
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
umap_results <- umap(embedding_matrix, n_neighbors =10, learning_rate = 0.5,n_components = 50,n_epochs = 500)
plot(umap_results[,1], umap_results[,2], col = as.factor(merged_df_100$genre), pch = 19, xlab = "UMAP1", ylab = "UMAP2")

legend("topleft", legend = unique(merged_df$genre), col = 1:length(unique(merged_df$genre)), pch = 19)
# Create a new dataframe for plotting
umap_df <- data.frame(UMAP1 = umap_results[,1],
                      UMAP2 = umap_results[,2],
                      Sentence = merged_df_100$sentence,
                      Genre = as.factor(merged_df_100$genre))

# Create the Plotly plot
plot_ly(umap_df, x = ~UMAP1, y = ~UMAP2, type = 'scatter', mode = 'markers',
        text = ~Sentence, # Display the sentence on hover
        color = ~Genre, # Color by genre
        marker = list(size = 10)) %>%
  layout(title = 'UMAP Plot',
         xaxis = list(title = 'UMAP1'),
         yaxis = list(title = 'UMAP2'),
         hovermode = 'closest')

##TSNE
library(Rtsne)
unique_embedding_matrix <- unique(embedding_matrix)

# Set seed for reproducibility
set.seed(123)

# Run t-SNE
tsne_results <- Rtsne(embedding_matrix, dims = 2, perplexity = 10,
                      theta =1, check_duplicates = F ,pca = F,
                      num_threads = 0, verbose = TRUE, max_iter = 2000,
                      normalize = T,mom_switch_iter = 500,stop_lying_iter = 1000)


# Plotting
plot(tsne_results$Y[,1], tsne_results$Y[,2], col = as.factor(merged_df_100$genre), pch = 19, xlab = "t-SNE 1", ylab = "t-SNE 2")
## plotly
tsne_df <- data.frame(tsne1 = tsne_results$Y[,1],
                      tsne2 = tsne_results$Y[,2],
                      Sentence = merged_df_100$sentence,
                      Genre = as.factor(merged_df_100$genre))
# Create the Plotly plot
plot_ly(tsne_df, x = ~tsne1, y = ~tsne2, type = 'scatter', mode = 'markers',
        text = ~Sentence, # Display the sentence on hover
        color = ~Genre, # Color by genre
        marker = list(size = 10)) %>%
  layout(title = 'TSNE Plot',
         xaxis = list(title = 'tsne1'),
         yaxis = list(title = 'tsne2'),
         hovermode = 'closest')
