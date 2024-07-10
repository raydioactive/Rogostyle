## PROSE VS POETRY

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

Prose_sentences <- tokenize_sentences_from_file(("/slipstream_old/home/juliusherzog/Rogostyle/texts/Prose_short.txt"))
Poetry_sentences<-tokenize_sentences_from_file(("/slipstream_old/home/juliusherzog/Rogostyle/texts/Poetry.txt"))

# Create dataframe
Prose_sentences

Prose_df <- data.frame(Sentence = unlist(Prose_sentences), genre = "Prose", stringsAsFactors = FALSE)
Poetry_df <- data.frame(Sentence = unlist(Poetry_sentences), genre = "Poetry", stringsAsFactors = FALSE)

combined_pp_df<-rbind(Prose_df,Poetry_df)
# View the dataframe

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



Prose_vector <- rep("Prose", nrow(Prose_df))
Poetry_vector <- rep("Poetry", nrow(Poetry_df))


PP_df_combined <- generate_embeddings(combined_pp_df$Sentence, c(Prose_vector,Poetry_vector))
variedbyone_df<-generate_embeddings(variedbyone, c(rep("1", 7),rep("2",8), rep("3",8), rep("4",8)))
print(result_df_)


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
embedding_matrix <- do.call(rbind, variedbyone_df$cls_embedding)
embedding_as_genemat<-t(embedding_matrix)
#Calculate Principal components of variance in embedding matrix
pca_results <- prcomp(embedding_matrix, rank. = 50)
#PlotPCA
plot(pca_results$x[,1:2], col = as.factor(result_df_combined$genre), pch = 19)
legend("topleft", legend = unique(result_df_combined$genre), col = 1:length(unique(result_df_combined$genre)), pch = 19)
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
umap_results <- umap(embedding_matrix, n_neighbors = 31, learning_rate = 5)
plot(umap_results[,1], umap_results[,2], col = as.factor(variedbyone_df$genre), pch = 19, xlab = "UMAP1", ylab = "UMAP2")


legend("bottomleft", legend = unique(variedbyone_df$genre), col = 1:length(unique(variedbyone_df$genre)), pch = 19)
###
library(dplyr)
library(plotly)
saveRDS(umap_df, file = "UMAP_df.rds")
# Create a new dataframe for plotting
umap_df <- data.frame(UMAP1 = umap_results[,1],
                      UMAP2 = umap_results[,2],
                      Sentence = variedbyone_df$sentence,
                      Genre = as.factor(variedbyone_df$genre))
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
#### DESEQ2
library(DESeq2)
cts<-embedding_as_genemat
colnames(cts)<-rep(1:31)
coldata = data.frame(condition = c(rep("Sent_1", 7),rep("Sent_2",8), rep("Sent_3",8), rep("Sent4",8)))
coldata$condition<-as.factor(coldata$condition)
rownames(coldata)<-rep(1:length(coldata$condition))

##### MAST
library(MAST)
library(SingleCellExperiment)
library(dplyr)
sce <- SingleCellExperiment(assays = embedding_as_genemat,
                            colData = coldata)
sca <- SceToSingleCellAssay(sce, check_sanity = FALSE)
zlmCond <- zlm(formula = as.formula(paste("~", "condition")), sca = sca, parallel = TRUE)
summaryCond <- summary(zlmCond, doLRT = TRUE, parallel = TRUE)
View(summaryCond$datatable)
##### LIME




