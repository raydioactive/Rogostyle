library(reticulate)
library(dplyr)
use_condaenv("/slipstream_old/home/juliusherzog/miniconda3/envs/cltk", required = TRUE)


Extract_text_embeddings_to_df_bert <- function(file_path) {
  # Read the text file
  text_data <- readLines(file_path, warn = T)

  # Concatenate the lines into a single string
  text_data <- paste(text_data, collapse = " ")

  # Define paths for tokenizer and BERT model
  tokenizer_path <- "/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder"
  bert_model_path <- "/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/"

  # Use reticulate to source Python script and load LatinBERT
  source_python("/slipstream_old/home/juliusherzog/Rogostyle/Berts_req.py")
  latinBERT = LatinBERT(tokenizerPath="/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder", bertPath="/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/")
  latinBERT <- LatinBERT(tokenizerPath=tokenizer_path, bertPath=bert_model_path)

  # Assuming get_berts method returns a list of embeddings for each sentence
  embeddings <- latinBERT$get_berts(list(text_data))

  # Extract sentences and embeddings
  sentences <- sapply(embeddings, function(x) x[[1]][1])
  embeddings_list <- lapply(embeddings, function(x) x[[1]][2])

  # Create a dataframe
  data_df <- data.frame(sentence = sentences, stringsAsFactors = FALSE)
  data_df$embedding <- I(embeddings_list)

  return(data_df)
}

Extract_text_embeddings_to_df_bert("/slipstream_old/home/juliusherzog/Rogostyle/chunks/Arma")


# Source the Python script
source_python("/slipstream_old/home/juliusherzog/Rogostyle/Berts_req.py")

# Define R function to read text, process it with CLTK, and generate embeddings with BERT
extract_text_embeddings_to_df_bert <- function(file_path, chunk_dir, embeddings_dir) {
  # Read the text file
  text_data <- (file_path)

  # Concatenate the lines into a single string
  text_data <- paste(text_data, collapse = " ")

  # Assuming your Python environment has cltk installed and set up
  # Process text with CLTK (adapted as a Python function if needed)
  # Skipping CLTK processing steps here for brevity

  # Assuming `latinBERT` is an instance of your LatinBERT class from the Python script
  # and `generate_and_store_embeddings` is a function in the same script
  latinBERT_instance <- py$LatinBERT(tokenizerPath="/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder", bertPath="/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/")
  # Generate and store BERT embeddings
  generate_and_store_embeddings(chunk_dir, latinBERT, embeddings_dir)

  # Load embeddings (assuming embeddings are stored as .pkl files)
  embeddings <- list.files(embeddings_dir, full.names = TRUE) %>%
    lapply(function(x) py_run_string(paste0("pickle.load(open('", x, "', 'rb'))"))) %>%
    do.call(rbind, .)

  return(embeddings)
}

# Example call
# Adjust the paths as necessary
embeddings_df <- extract_text_embeddings_to_df_bert("/slipstream_old/home/juliusherzog/Rogostyle/chunks/Arma",
                                                    "/slipstream_old/home/juliusherzog/Rogostyle/chunks/",
                                                    "/slipstream_old/home/juliusherzog/Rogostyle/embeddings/")
