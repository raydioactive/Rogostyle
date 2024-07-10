library(reticulate)
library(dplyr)

use_condaenv("/slipstream_old/home/juliusherzog/miniconda3/envs/cltk", required = TRUE)
use_python("/slipstream_old/home/juliusherzog/miniconda3/envs/cltk/bin/python")
# Load the Python script that generates embeddings

# Create an R function to generate embeddings and return a dataframe
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

process_text_with_cltk <- function(file_path) {
  # Read the text file
  text_data <- readLines(file_path)

  # Concatenate the lines into a single string
  text_data <- paste(text_data, collapse = " ")

  # Define a Python function that uses CLTK
  analyze_text <- py_run_string("
def analyze_text_with_cltk(text):
    from cltk import NLP
    cltk_nlp = NLP(language='lat')
    cltk_nlp.pipeline.processes.pop(-1)
    print(cltk_nlp.pipeline.processes)
    doc = cltk_nlp.analyze(text)
    return doc
  ")

  # Call the Python function from R and store the result
  cltk_doc <- analyze_text$analyze_text_with_cltk(text_data)
  # Process cltk_doc as needed
  #

  return(cltk_doc)
}
Speech_cltk_doc<-process_text_with_cltk("~/Rogostyle/texts/5kspeechchars.txt")

sentences<-Speech_cltk_doc$sentences_strings
saveRDS(sentences, file = "~/Rogostyle/sentences.rds")

