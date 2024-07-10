###RESTART-> LATINBERT
library(reticulate)

use_condaenv("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert", required = TRUE)
use_python("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert/bin/python")
source_python("~/Rogostyle/gen_berts.py")
#
generate_embeddings <- function(sentences, genre) {
  # Create a LatinBERT object with the specified paths
  bert <- LatinBERT(tokenizerPath = "/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder",
                    bertPath = "/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/")

  # Generate embeddings for the sentences
  bert_sents <- bert$get_berts(sentences)

  # Extract the token and embedding from each sentence
  embeddings <- lapply(bert_sents, function(sent) {
    lapply(sent, function(token_bert) {
      list(token = token_bert[[1]], embedding = token_bert[[2]])
    })
  })

  # Create a dataframe with the sentence strings, embeddings, and genre
  df <- data.frame(
    sentence = sentences,
    embedding = I(embeddings),
    genre = genre,
    stringsAsFactors = FALSE
  )

  return(df)
}
#
tokenize_sentences <- function(text, language = "lat") {
  # Use the reticulate::py_run_string() to define Python code
  py_run_string("
from cltk.sentence.sentence import PunktSentenceTokenizer

def tokenize_text(input_text, language):
    tokenizer = PunktSentenceTokenizer(language=language)
    sentences = tokenizer.tokenize(input_text)
    return sentences
  ")

  # Call the Python function
  sentences <- py$tokenize_text(text, language)

  return(sentences)
}
tokenize_sentences_from_file <- function(file_path, language = "lat") {
  # Use reticulate to define Python code
  py_run_string("
from cltk.sentence.sentence import PunktSentenceTokenizer

def tokenize_text_from_file(file_path, language):
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
    tokenizer = PunktSentenceTokenizer(language=language)
    sentences = tokenizer.tokenize(input_text)
    return sentences
  ")

  # Call the Python function with the file path
  sentences <- py$tokenize_text_from_file(file_path, language)

  return(sentences)
}

letter_sentences <- tokenize_sentences_from_file(("/slipstream_old/home/juliusherzog/Rogostyle/texts/333kletterchars.txt"))
phil_sentences<-tokenize_sentences_from_file(("/slipstream_old/home/juliusherzog/Rogostyle/texts/333kphilchars.txt"))
speech_sentences<-tokenize_sentences_from_file(("/slipstream_old/home/juliusherzog/Rogostyle/texts/333kspeechchars.txt"))

# Create dataframe
speech_df <- data.frame(Sentence = unlist(speech_sentences), genre = "Speech", stringsAsFactors = FALSE)
letters_df <- data.frame(Sentence = unlist(letter_sentences), genre = "Letter", stringsAsFactors = FALSE)
phil_df <- data.frame(Sentence = unlist(phil_sentences), genre = "Philosophies", stringsAsFactors = FALSE)
combined_df<-rbind(speech_df,letters_df,phil_df)
# View the dataframe
View(combined_df)

variedbyone = c("Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio dicendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum artium studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic A. Licinius fructum a me repetere prope suo iure debet.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio dicendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum artium studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope suo iure debet.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio dicendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum artium studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope eius iure debet.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio dicendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum artium studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope eius iure potest.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio loquendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum artium studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope eius iure potest.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio loquendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum scientiarum studiis ac disciplina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope eius iure potest.",
                "Si quid est in me ingeni, iudices, quod sentio quam sit exiguum, aut si qua exercitatio loquendi in qua me non infitior mediocriter esse versatum, aut si huiusce rei ratio aliqua ab optimarum scientiarum studiis ac doctrina profecta, a qua ego nullum confiteor aetatis meae tempus abhorruisse, earum rerum omnium vel in primis hic B. Licinius fructum a me repetere prope eius iure potest.",
                "Nam quoad longissime potest mens mea respicere spatium praeteriti temporis et pueritiae memoriam recordari ultimam, inde usque repetens, hunc video mihi principem et ad suscipiendam et ad ingrediendam rationem horum studiorum exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium praeteriti temporis et pueritiae memoriam recordari ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam rationem horum studiorum exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium praeteriti temporis et pueritiae memoriam recordari ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum studiorum exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium futuri temporis et pueritiae memoriam recordari ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum studiorum exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium futuri temporis et pueritiae memoriam recordari ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum artium exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium futuri temporis et adolescentiae memoriam recordari ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum artium exstitisse.",
                "Nam quoad longissime potest mens mea respicere spatium futuri temporis et adolescentiae memoriam revocare ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum artium exstitisse.",
                'Nam quoad longissime potest mens mea respicere spatium futuri temporis et adolescentiae memoriam revocare ultimam, inde usque repetens, hunc video mihi ducem et ad suscipiendam et ad ingrediendam viam horum artium fuisse.',
                "Quod si haec vox huius hortatu praeceptisque conformata non nullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et alios servare possemus, huic profecto ipsi quantum est situm in nobis et opem et salutem ferre debemus.",
                "Quod si haec vox huius hortatu praeceptisque conformata non nullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et opem et salutem ferre debemus.",
                "Quod si haec vox huius hortatu praeceptisque conformata non nullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et opem et salutem offerre debemus.",
                "Quod si haec vox eius hortatu praeceptisque conformata non nullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et opem et salutem offerre debemus.",
                "Quod si haec vox eius hortatu praeceptisque conformata non nullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et auxilium et salutem offerre debemus.",
                "Quod si haec vox eius hortatu praeceptisque conformata nonnullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et auxilium et salutem offerre debemus.",
                "Quod si haec vox eius hortatu praeceptisque conformata nonnullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et auxilium et salutem praebere debemus.",
                "Quod si haec vox eius hortatu praeceptisque formata nonnullis aliquando saluti fuit, a quo id accepimus, quo ceteris opitulari et multos servare possemus, huic profecto ipsi quantum est situm in nobis et auxilium et salutem praebere debemus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod alia quaedam in hoc facultas sit ingeni, neque haec dicendi ratio aut disciplina, ne nos quidem huic uni studio penitus umquam dediti fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod alia quaedam in hoc facultas sit ingeni, neque haec dicendi ratio aut disciplina, ne nos quidem huic uni studio penitus umquam dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod alia quaedam in hoc facultas sit ingeni, neque haec scribendi ratio aut disciplina, ne nos quidem huic uni studio penitus dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod aliae quaedam in hoc facultas sit ingeni, neque haec scribendi ratio aut disciplina, ne nos quidem huic uni studio penitus dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod aliae quaedam in hoc facultas sit ingeni, neque haec scribendi ratio aut disciplina, sed nos quidem huic uni studio penitus dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quod aliae quaedam in hoc facultas sit ingeni, nec haec scribendi ratio aut disciplina, sed nos quidem huic uni studio penitus dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quia aliae quaedam in hoc facultas sit ingeni, nec haec scribendi ratio aut disciplina, sed nos quidem huic uni studio penitus dedicati fuimus.",
                "Ac ne quis a nobis hoc ita dici forte miretur, quia aliae quaedam in hoc facultas sit ingeni, nec haec scribendi ratio aut disciplina, sed nos quidem huic multi studio penitus dedicati fuimus.")



speech_vector <- rep("speech", nrow(speech_df))
letter_vector <- rep("letter", nrow(letters_df))
phil_vector <- rep("phil", nrow(phil_df))

result_df_combined <- generate_embeddings(combined_df$Sentence, c(speech_vector,letter_vector,phil_vector))
variedbyone_df<-generate_embeddings(variedbyone, c(rep("1", 7),rep("2",8), rep("3",8), rep("4",8)))

print(result_df_)
result_df_combined <- subset(result_df_combined, sentence != ".")
saveRDS(result_df_combined, file = "result_df_spchletterphil.rds")
###
result_df_combined$cls_embedding <- lapply(result_df_combined$embedding, function(embed_list) {
  # Search for the [CLS] token in the list and extract its embedding
  cls_embed <- NULL
  for (embed in embed_list) {
    if (embed$token == "[CLS]") {
      cls_embed <- embed$embedding
      break  # Stop the loop once the [CLS] embedding is found
    }
  }
  return(cls_embed)
})
##
#result_df_combined$cls_embedding<-list(result_df_combined$cls_embedding)
embedding_matrix <- do.call(rbind, result_df_combined$cls_embedding)
embedding_as_genemat<-t(embedding_matrix)
#Calculate Principal components of variance in embedding matrix
pca_results <- prcomp(embedding_matrix, rank. = 50)
#PlotPCA
plot(pca_results$x[,1:2], col = as.factor(result_df_combined$genre), pch = 19)
legend("bottomleft", legend = unique(result_df_combined$genre), col = 1:length(unique(result_df_combined$genre)), pch = 19)
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
## 3D PCA
library(plotly)

library(stats)
components <- pca_results[["x"]]
components <- data.frame(components)
components$PC2 <- -components$PC2
components$PC3 <- -components$PC3
components = cbind(components, result_df_combined$sentence)
tot_explained_variance_ratio <- summary(pca_results)[["importance"]]['Proportion of Variance',]
tot_explained_variance_ratio <- 100 * sum(tot_explained_variance_ratio)
tit = 'Total Explained Variance = 99.48'
fig <- plot_ly(components, x = ~PC1, y = ~PC2, z = ~PC3, color = ~result_df_combined$genre, colors = c('#636EFA','#EF553B','#00CC96') ) %>%
  add_markers(size = 12)
fig <- fig %>%
  layout(
    title = tit,
    scene = list(bgcolor = "#e5ecf6")
  )
####!!!!
fig
##

#UMAP
library(uwot)
set.seed(123) # For reproducibility
umap_results <- umap(embedding_matrix, n_neighbors = 1000, learning_rate = 10, metric = "correlation")
plot(umap_results[,1], umap_results[,2], col = as.factor(result_df_combined$genre), pch = 19, xlab = "UMAP1", ylab = "UMAP2")


legend("bottomleft", legend = unique(result_df_combined$genre), col = 1:length(unique(result_df_combined$genre)), pch = 19)
###
library(dplyr)
library(plotly)
saveRDS(umap_df, file = "UMAP_df.rds")
# Create a new dataframe for plotting
umap_df <- data.frame(UMAP1 = umap_results[,1],
                      UMAP2 = umap_results[,2],
                      Sentence = result_df_combined$sentence,
                      Genre = as.factor(result_df_combined$genre))
umap_df <- readRDS(file = "UMAP_df.rds")
# Create the Plotly plot
plot_ly(umap_df, x = ~UMAP1, y = ~UMAP2, type = 'scatter', mode = 'markers',
        text = ~Sentence, # Display the sentence on hover
        color = ~Genre, # Color by genre
        marker = list(size = 10)) %>%
  layout(title = 'UMAP Plot',
         xaxis = list(title = 'UMAP1'),
         yaxis = list(title = 'UMAP2'),
         hovermode = 'closest')

## 3D TSNE
library(Rtsne)
library(plotly)
library(dplyr)
set.seed(123)  # For reproducibility
tsne_result <- Rtsne(embedding_matrix, dims = 3, perplexity = 10, verbose = TRUE, max_iter = 2000 ,check_duplicates = F,num_threads = 0,eta = 1000)
tsne_result_2D <- Rtsne(embedding_matrix, dims = 2, perplexity = 10, verbose = TRUE, max_iter = 1000 ,check_duplicates = F,num_threads = 0,eta = 1000)
# Extract the 3D t-SNE embeddings
tsne_embeddings <- tsne_result$Y

# Create a 3D scatter plot using plotly
plot_data <- data.frame(
  x = tsne_embeddings[, 1],
  y = tsne_embeddings[, 2],
  z = tsne_embeddings[, 3],
  Sentence = result_df_combined$sentence,
  Genre = as.factor(result_df_combined$genre)
)
library(plotly)
plot_ly(plot_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
        text = ~Sentence, # Display the sentence on hover
        color = ~Genre, # Color by cluster
        marker = list(size = 4, opacity = 0.5)) %>%
  layout(scene = list(xaxis = list(title = "t-SNE Dimension 1"),
                      yaxis = list(title = "t-SNE Dimension 2"),
                      zaxis = list(title = "t-SNE Dimension 3")),


         title = "3D t-SNE Embedding Plot")
#### CLUSTERING
# Extract the 3D t-SNE coordinates
tsne_coords <- plot_data[, c("x", "y", "z")]

# Determine the optimal number of clusters using the elbow method
wss <- sapply(1:20, function(k) {
  kmeans(tsne_coords, centers = k, nstart = 20, iter.max = 100)$tot.withinss
})
plot(1:20, wss, type = "b", xlab = "Number of Clusters", ylab = "Within Sum of Squares")

# Choose the number of clusters based on the elbow point
k = 5  # Adjust this based on  observation

# Perform k-means clustering on the t-SNE coordinates
kmeans_result <- kmeans(tsne_coords, centers = k, nstart = 10, iter.max = 100)

# Get the cluster assignments for each sentence
cluster_assignments <- kmeans_result$cluster

# Add the cluster assignments to the plot_data data frame
plot_data$Cluster <- as.factor(cluster_assignments)

library(dplyr)
# Create the 3D t-SNE plot with clusters using plotly
plot_ly(plot_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
        text = ~paste("Sentence: ", Sentence, "Genre: ", Genre, "Cluster: ", Cluster),
        color = ~Cluster, # Color by cluster
        marker = list(size = 5, opacity = 0.8)) %>%
  layout(scene = list(xaxis = list(title = "t-SNE Dimension 1"),
                      yaxis = list(title = "t-SNE Dimension 2"),
                      zaxis = list(title = "t-SNE Dimension 3")),
         title = "3D t-SNE Embedding Plot with Clusters")

#By GENRE
plot_ly(plot_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
        text = ~paste("Sentence: ", Sentence, "Genre: ", Genre, "Cluster: ", Cluster),
        color = ~Genre, # Color by cluster
        marker = list(size = 5, opacity = 0.8)) %>%
  layout(scene = list(xaxis = list(title = "t-SNE Dimension 1"),
                      yaxis = list(title = "t-SNE Dimension 2"),
                      zaxis = list(title = "t-SNE Dimension 3")),
         title = "3D t-SNE Embedding Plot with Clusters")

#### Pie Charts
create_cluster_pie_charts <- function(plot_data, n_clusters) {
  par(mfrow = c(1, n_clusters))  # Arrange the plots in a 1xn grid

  for (i in 1:n_clusters) {
    cluster_data <- plot_data[plot_data$Cluster == i, ]
    cluster_genre_counts <- table(cluster_data$Genre)

    # Create pie chart for the current cluster
    pie(cluster_genre_counts, main = paste("Cluster", i), col = rainbow(length(cluster_genre_counts)))
    legend("topright", legend = names(cluster_genre_counts), fill = rainbow(length(cluster_genre_counts)), cex = 0.8)
  }
}

create_cluster_pie_charts(plot_data, k)
#### Louvain_clustering
library(igraph)
library(cluster)
#
# Number of neighbors
k <- 5

# Compute the distance matrix
dist_matrix <- as.matrix(dist(tsne_coords))

# Find the k-nearest neighbors for each point
knn <- apply(dist_matrix, 1, order)[1:(k+1), ]

# Create an adjacency matrix
adj_matrix <- matrix(0, nrow = nrow(dist_matrix), ncol = ncol(dist_matrix))
for (i in 1:nrow(dist_matrix)) {
  adj_matrix[i, knn[, i]] <- 1
  adj_matrix[knn[, i], i] <- 1 # Ensure symmetry
}

# Convert the adjacency matrix to a graph
graph <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
#louvain
set.seed(123) # For reproducibility
louvain_clusters <- cluster_louvain(graph)$membership
# Add Louvain cluster assignments to the plot_data dataframe
plot_data$LouvainCluster <- as.factor(louvain_clusters)

# Create the 3D t-SNE plot with Louvain clusters using plotly
plot_ly(plot_data, x = ~x, y = ~y, z = ~z, type = "scatter3d", mode = "markers",
        text = ~paste("Sentence: ", Sentence, "Genre: ", Genre, "Louvain Cluster: ", LouvainCluster),
        color = ~LouvainCluster, # Color by Louvain cluster
        marker = list(size = 5, opacity = 0.8)) %>%
  layout(scene = list(xaxis = list(title = "t-SNE Dimension 1"),
                      yaxis = list(title = "t-SNE Dimension 2"),
                      zaxis = list(title = "t-SNE Dimension 3")),
         title = "3D t-SNE Embedding Plot with Louvain Clusters")

###Centroids, identify most representative sentences

# Calculate the centroids of each cluster
cluster_centroids <- aggregate(cbind(x, y, z) ~ LouvainCluster, data = plot_data, FUN = mean)
# Function to calculate Euclidean distance
euclidean_dist <- function(x1, y1, z1, x2, y2, z2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
}

# Initialize a vector to store the index of the closest sentence for each cluster
closest_sentences_indices <- numeric(nrow(cluster_centroids))

for (i in 1:nrow(cluster_centroids)) {
  # Extract current cluster centroid
  centroid <- cluster_centroids[i,]

  # Subset the plot_data for the current cluster
  cluster_data <- subset(plot_data, LouvainCluster == centroid$LouvainCluster)

  # Calculate distances from each point in the cluster to the centroid
  distances <- euclidean_dist(cluster_data$x, cluster_data$y, cluster_data$z,
                              centroid$x, centroid$y, centroid$z)

  # Find the index of the minimum distance
  closest_sentences_indices[i] <- which.min(distances)
}

# Extract the most representative sentences
representative_sentences <- plot_data[closest_sentences_indices, "Sentence"]
for (i in 1:length(representative_sentences)) {
  cat("Cluster", i, "representative sentence:", representative_sentences[i], "\n\n")
}



####
library(DESeq2)
cts<-embedding_as_genemat
coldata = data.frame(condition = result_df_combined$genre)
colnames(cts)<-rep(1:length(coldata$condition))
coldata = data.frame(condition = result_df_combined$genre)
coldata$condition<-as.factor(coldata$condition)
rownames(coldata)<-rep(1:length(coldata$condition))

##### MAST
library(MAST)
library(SingleCellExperiment)
library(dplyr)
sce <- SingleCellExperiment(assays = t(embedding_matrix),
                            colData = coldata)
sca <- SceToSingleCellAssay(sce, check_sanity = FALSE)
zlmCond <- zlm(formula = as.formula(paste("~", "condition")), sca = sca, parallel = TRUE)
summaryCond <- summary(zlmCond, doLRT = TRUE, parallel = TRUE)
View(summaryCond$datatable)
##sceSTYLO

counts_data <- assays(cds_Stylo)[["counts"]]  # Adjust as needed if using a different assay like "norm_counts"
rowData <- rowData(cds_Stylo)
colData <- colData(cds_Stylo)
reducedDimsData <- reducedDims(cds_Stylo)  # This may include PCA, tSNE, UMAP dimensions etc.

# Create the SCE object
scestylo <- SingleCellExperiment(
  assays = list(counts = counts_data),
  rowData = rowData,
  colData = colData
)
scastylo <- SceToSingleCellAssay(scestylo, check_sanity = FALSE)
zlmCond <- zlm(formula = as.formula(paste("~", "genre")), sca = scastylo, parallel = TRUE)
summaryCond <- summary(zlmCond, doLRT = TRUE, parallel = TRUE)
View(summaryCond$datatable)


##BARPLOTS
library(tidyr)
library(ggplot2)
row_indices <- c(1,8,16,24)  # Replace with your desired row indices

# Extract the selected embeddings from the 'embedding' column
selected_embeddings <- variedbyone_df$cls_embedding[row_indices]

# Create a list to store the individual plots
plot_list <- list()

# Iterate over each selected embedding and create a separate plot
for (i in seq_along(selected_embeddings)) {
  # Convert the current embedding to a data frame
  df <- data.frame(Position = seq_along(selected_embeddings[[i]]), Value = selected_embeddings[[i]])

  # Create the barplot using ggplot2
  plot <- ggplot(df, aes(x = Position, y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(x = "Position", y = "Value", title = paste("Embedding", row_indices[i])) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

  # Add the plot to the plot_list
  plot_list[[i]] <- plot
}


# Arrange the plots in a grid using gridExtra
library(gridExtra)
grid.arrange(grobs = plot_list, ncol = 2)

### p316 is strongly changed in speech and phil relative to letters: look at sentences on the opposite ends of that spectrum

library(plotly)

create_plots_for_embedding_position <- function(position) {
  # Extract the value at the specified position for each row's cls_embedding
  values_at_position <- sapply(result_df_combined$cls_embedding, function(embedding) embedding[position])

  # Find the row index with the minimum value at the specified position
  min_index <- which.min(values_at_position)
  max_index <- which.max(values_at_position)

  # Retrieve the row from the dataframe
  row_with_min_value <- result_df_combined[min_index, ]
  row_with_max_value <- result_df_combined[max_index, ]

  # Display the sentence associated with the minimum value
  cat("Sentence with minimum value at position", position, ":\n", row_with_min_value$sentence, "\n\n")
  cat("Sentence with maximum value at position", position, ":\n", row_with_max_value$sentence, "\n\n")

  # Create an interactive scatter plot
  fig <- plot_ly(data = result_df_combined, x = values_at_position, y = rep(1, length(values_at_position)),
                 type = 'scatter', mode = 'markers',
                 text = ~sentence, # Assuming the column containing sentences is named 'sentence'
                 hoverinfo = 'text', marker = list(color = 'blue'))
  fig <- fig %>% layout(title = paste('Interactive Scatter Plot of Values at Position', position),
                        xaxis = list(title = paste('Value at Position', position)),
                        yaxis = list(title = '', tickmode = 'array', tickvals = c(1), ticktext = c(''), showticklabels = FALSE))

  # Show the plot
  fig %>% print()

  # Create a histogram of these values
  #hist(values_at_position, main = paste("Histogram of Values at Position", position), xlab = "Value", ylab = "Frequency", col = "blue")
}


# Example usage of the function
create_plots_for_embedding_position(385)


### Louvain Clustering of the TSNE Plot - new DEG analysis grouping similar sentences.
