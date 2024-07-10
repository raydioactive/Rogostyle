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
    stem = "character", # If stem is always NULL, you can remove this slot
    scansion = "character", # If scansion is always NULL, you can remove this slot
    xpos = "character",
    upos = "character",
    dependency_relation = "character",
    governor = "numeric",
    features = "list",
    category = "list",
    stop = "logical",
    named_entity = "character", # If named_entity is always NULL, you can remove this slot
    syllables = "character", # If syllables is always NULL, you can remove this slot
    phonetic_transcription = "character", # If phonetic_transcription is always NULL, you can remove this slot
    definition = "character" # If definition is always NULL, you can remove this slot
  )
)

setClass(
  "Sentence",
  slots = c(
    words = "list", # List of Word objects
    index = "numeric"
))
