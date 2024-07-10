import csv
from cltk.sentence.sentence import PunktSentenceTokenizer

# Define the paths to your prose and poetry text files
prose_file = '/slipstream_old/home/juliusherzog/Rogostyle/texts/Prose.txt'
poetry_file = '/slipstream_old/home/juliusherzog/Rogostyle/texts/Poetry.txt'

# Initialize the tokenizer for Latin
tokenizer = PunktSentenceTokenizer(language='lat')

# Function to process a text file and return sentences and genre
def process_text(file_path, genre):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    if genre == 'poetry':
        # Replace newline characters and indents with spaces in the poetry text
        text = re.sub(r'\n\s+', ' ', text)

    sentences = tokenizer.tokenize(text)
    return [(sentence, genre) for sentence in sentences]

# Process prose and poetry texts
prose_sentences = process_text(prose_file, 'prose')
poetry_sentences = process_text(poetry_file, 'poetry')

# Combine sentences from both genres
dataset = prose_sentences + poetry_sentences

# Save the dataset as a CSV file
csv_file = 'dataset.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Sentence', 'Genre'])  # Write header
    writer.writerows(dataset)  # Write data rows

print(f"Dataset saved as {csv_file}")
