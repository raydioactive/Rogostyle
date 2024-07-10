#PreProcess Text:
library(stringr)
preprocess_text <- function(text) {
  # Convert the encoding to UTF-8
  text <- iconv(text, to = "UTF-8", sub = "byte")

  # Proceed with preprocessing
  text <- tolower(text)  # Convert to lowercase
  text <- gsub("[^a-z.]", " ", text)  # Remove all non-alphabet characters except periods
  text <- gsub("\\s+", " ", text)  # Replace multiple spaces with a single space
  return(trimws(text))
}
directory_path <- "~/latinbert/Cicero_texts/philosophies/"

# List all text files in the directory
file_paths <- list.files(directory_path, pattern = "\\.txt$", full.names = TRUE)
# Loop over each file, preprocess, and optionally save the cleaned text
for (file_path in file_paths) {
  # Read the file content
  text_content <- readLines(file_path, warn = FALSE)

  # Preprocess the text
  cleaned_text <- sapply(text_content, preprocess_text, USE.NAMES = FALSE)
  cleaned_text <- paste(cleaned_text, collapse = "\n")

  # Save the cleaned text back to the file or to a new file
  # Uncomment the line below to overwrite the original file
  # writeLines(cleaned_text, file_path)

  # Uncomment the line below to save to a new file (e.g., appending "_cleaned" to the filename)
  new_file_path <- str_replace(file_path, "\\.txt$", "_cleaned.txt")
  writeLines(cleaned_text, new_file_path)
}
