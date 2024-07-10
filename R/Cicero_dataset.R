library(tidyr)
result_df_spchletterphil<-readRDS(file = "result_df_spchletterphil.rds")
result_df_combined = result_df_spchletterphil
# Unnest the embedding column
result_df_unnested <- result_df_combined %>%
  unnest(embedding)

# Convert the embedding to a matrix or data frame
embedding_matrix <- as.matrix(result_df_unnested[, -c(1, 2, 3)])
cicero_works <- data.frame(
  title = unique(Cicero_dataset$`Corpus Name`),
  stringsAsFactors = FALSE
)

# Create a function to classify the works
classify_work <- function(title) {
  if (grepl("Epistulae|Letters", title)) {
    return("Letters")
  } else if (grepl("De |Disputationes|Academica|Topica|Paradoxa", title)) {
    return("Philosophy")
  } else {
    return("Speech")
  }
}

# Apply the classification function to the titles
cicero_works$category <- sapply(cicero_works$title, classify_work)
Cicero_dataset$category <- cicero_works$category
Letters_Cdata = Cicero_dataset[Cicero_dataset$category == "Letters",]
Speech_Cdata = Cicero_dataset[Cicero_dataset$category == "Speech",]
Phil_Cdata = Cicero_dataset[Cicero_dataset$category == "Philosophy",]
Averaged<-data.frame(Speech=colMeans(Speech_Cdata[1:26]), Philosophy=colMeans(Phil_Cdata[1:26]) , Letters=colMeans(Letters_Cdata[1:26]))
Averaged$Philosophy_vs_Speech_LFC <- log2(Averaged$Philosophy / Averaged$Speech)

# Log fold change: Letters vs Speech
Averaged$Letters_vs_Speech_LFC <- log2(Averaged$Letters / Averaged$Speech)

# Log fold change: Letters vs Philosophy
Averaged$Letters_vs_Philosophy_LFC <- log2(Averaged$Letters / Averaged$Philosophy)

#### Per sentence:
#### Per sentence:

library(reticulate)

use_condaenv("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert", required = TRUE)
use_python("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert/bin/python")
source_python("~/Rogostyle/stylometry.py")

sentences <- result_df_combined$sentence

# Initialize an empty list to store the stylometry results
stylometry_results <- list()

# Iterate over each sentence
for (sentence in sentences) {
  # Call the calculate_stylometry function
  result <- calculate_stylometry(sentence)

  # Append the result to the list
  stylometry_results <- append(stylometry_results, list(result))
}

# Add the stylometry results as a new column in the data frame
result_df_combined$stylometry_embedding <- stylometry_results

# Print the updated data frame
View(result_df_combined)


#Stylometry matrix
library(dplyr)
library(tidyr)
df_list <- lapply(result_df_combined$stylometry_embedding, function(x) unlist(x, recursive = FALSE))
df <- do.call(rbind, df_list)
View(df)
df_mat<-as.matrix(df)
df_genemat<-t(df_mat)
colnames(df_genemat)<-paste0("Sentence", seq_along(result_df_combined$sentence)
# Convert the row-bound list into a data frame
df <- (t(df))
colnames(df) <- names(df_list[[1]])
