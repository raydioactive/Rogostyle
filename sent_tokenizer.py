from cltk.sentence.sentence import PunktSentenceTokenizer

# Load your text document
with open('text_document.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Initialize the tokenizer for Latin
tokenizer = PunktSentenceTokenizer(language='lat')

# Tokenize the document
sentences = tokenizer.tokenize(text)

# Print or save the sentences
print(sentences)
