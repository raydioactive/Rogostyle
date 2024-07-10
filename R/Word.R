setClass(
  "Word",
  slots = c(
    index_char_start = "numeric",
    index_char_stop = "numeric",
    index_token = "numeric",
    index_sentence = "numeric",
    string = "character",
    pos = "character",
    lemma = "character",
    stem = "character", # Allowing NULL
    scansion = "character", # Allowing NULL
    xpos = "character",
    upos = "character",
    dependency_relation = "character",
    governor = "numeric",
    features = "list", # Holds named character vectors
    category = "list", # Holds named character vectors
    stop = "logical",
    named_entity = "character", # Allowing NULL
    syllables = "list", # Allowing NULL
    phonetic_transcription = "character", # Allowing NULL
    definition = "character"
  ),
  prototype = list(
    stem = NULL,
    scansion = NULL,
    named_entity = NULL,
    syllables = NULL,
    phonetic_transcription = NULL
  )
)
