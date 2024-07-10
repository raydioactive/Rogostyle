#Load reticulate, use the conda environment that has cltk library
library(reticulate)
use_condaenv("cltk", required = TRUE)
# Define function to process text with cltk, return cltk_doc

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

Arma_Doc<-process_text_with_cltk("~/Arma.txt")

