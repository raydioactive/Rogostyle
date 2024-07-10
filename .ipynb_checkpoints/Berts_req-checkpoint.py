#Use slurm to generate these berts in a script:

#import
import pickle
import os
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatinBERT():

	def __init__(self, tokenizerPath=None, bertPath=None):
		encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
		self.wp_tokenizer = LatinTokenizer(encoder)
		self.model = BertLatin(bertPath=bertPath)
		self.model.to(device)

	def get_batches(self, sentences, max_batch, tokenizer):

			maxLen=0
			for sentence in sentences:
				length=0
				for word in sentence:
					toks=tokenizer.tokenize(word)
					length+=len(toks)

				if length> maxLen:
					maxLen=length

			all_data=[]
			all_masks=[]
			all_labels=[]
			all_transforms=[]

			for sentence in sentences:
				tok_ids=[]
				input_mask=[]
				labels=[]
				transform=[]

				all_toks=[]
				n=0
				for idx, word in enumerate(sentence):
					toks=tokenizer.tokenize(word)
					all_toks.append(toks)
					n+=len(toks)

				cur=0
				for idx, word in enumerate(sentence):
					toks=all_toks[idx]
					ind=list(np.zeros(n))
					for j in range(cur,cur+len(toks)):
						ind[j]=1./len(toks)
					cur+=len(toks)
					transform.append(ind)

					tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

					input_mask.extend(np.ones(len(toks)))
					labels.append(1)

				all_data.append(tok_ids)
				all_masks.append(input_mask)
				all_labels.append(labels)
				all_transforms.append(transform)

			lengths = np.array([len(l) for l in all_data])

			# Note sequence must be ordered from shortest to longest so current_batch will work
			ordering = np.argsort(lengths)
			
			ordered_data = [None for i in range(len(all_data))]
			ordered_masks = [None for i in range(len(all_data))]
			ordered_labels = [None for i in range(len(all_data))]
			ordered_transforms = [None for i in range(len(all_data))]
			

			for i, ind in enumerate(ordering):
				ordered_data[i] = all_data[ind]
				ordered_masks[i] = all_masks[ind]
				ordered_labels[i] = all_labels[ind]
				ordered_transforms[i] = all_transforms[ind]

			batched_data=[]
			batched_mask=[]
			batched_labels=[]
			batched_transforms=[]

			i=0
			current_batch=max_batch

			while i < len(ordered_data):

				batch_data=ordered_data[i:i+current_batch]
				batch_mask=ordered_masks[i:i+current_batch]
				batch_labels=ordered_labels[i:i+current_batch]
				batch_transforms=ordered_transforms[i:i+current_batch]

				max_len = max([len(sent) for sent in batch_data])
				max_label = max([len(label) for label in batch_labels])

				for j in range(len(batch_data)):
					
					blen=len(batch_data[j])
					blab=len(batch_labels[j])

					for k in range(blen, max_len):
						batch_data[j].append(0)
						batch_mask[j].append(0)
						for z in range(len(batch_transforms[j])):
							batch_transforms[j][z].append(0)

					for k in range(blab, max_label):
						batch_labels[j].append(-100)

					for k in range(len(batch_transforms[j]), max_label):
						batch_transforms[j].append(np.zeros(max_len))

				batched_data.append(torch.LongTensor(batch_data))
				batched_mask.append(torch.FloatTensor(batch_mask))
				batched_labels.append(torch.LongTensor(batch_labels))
				batched_transforms.append(torch.FloatTensor(batch_transforms))

				bsize=torch.FloatTensor(batch_transforms).shape
				
				i+=current_batch

				# adjust batch size; sentences are ordered from shortest to longest so decrease as they get longer
				if max_len > 100:
					current_batch=12
				if max_len > 200:
					current_batch=6

			return batched_data, batched_mask, batched_transforms, ordering


	def get_berts(self, raw_sents):
		sents=convert_to_toks(raw_sents)
		batch_size=32
		batched_data, batched_mask, batched_transforms, ordering=self.get_batches(sents, batch_size, self.wp_tokenizer)
	        # Debug: Print lengths of tokenized sentences
		for sent in sents:
			print(f"Tokenized sentence length: {len(sent)}")
			if len(sent) > 512:
				print("Warning: Sentence exceeds 512 tokens.")

		ordered_preds=[]
		for b in range(len(batched_data)):
			size=batched_transforms[b].shape
			b_size=size[0]
			berts=self.model.forward(batched_data[b], attention_mask=batched_mask[b], transforms=batched_transforms[b])
			berts=berts.detach()
			berts=berts.cpu()
			for row in range(b_size):
				ordered_preds.append([np.array(r) for r in berts[row]])

		preds_in_order = [None for i in range(len(sents))]


		for i, ind in enumerate(ordering):
			preds_in_order[ind] = ordered_preds[i]


		bert_sents=[]

		for idx, sentence in enumerate(sents):
			bert_sent=[]

			bert_sent.append(("[CLS]", preds_in_order[idx][0] ))

			for t_idx in range(1, len(sentence)-1):
				token=sentence[t_idx]
				
				pred=preds_in_order[idx][t_idx]
				bert_sent.append((token, pred ))

			bert_sent.append(("[SEP]", preds_in_order[idx][len(sentence)-1] ))

			bert_sents.append(bert_sent)

		return bert_sents




class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4

		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text):
		tokens=text.split(" ")
		wp_tokens=[]
		for token in tokens:

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens

def convert_to_toks(sents):

	sent_tokenizer = SentenceTokenizer()
	word_tokenizer = WordTokenizer()

	all_sents=[]

	for data in sents:
		text=data.lower()

		sents=sent_tokenizer.tokenize(text)
		for sent in sents:
			tokens=word_tokenizer.tokenize(sent)
			filt_toks=[]
			filt_toks.append("[CLS]")
			for tok in tokens:
				if tok != "":
					filt_toks.append(tok)
			filt_toks.append("[SEP]")

			all_sents.append(filt_toks)

	return all_sents




class BertLatin(nn.Module):

	def __init__(self, bertPath=None):
		super(BertLatin, self).__init__()

		self.bert = BertModel.from_pretrained(bertPath)
		self.bert.eval()
		
	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)
		sequence_outputs, pooled_outputs = self.bert.forward(input_ids, token_type_ids=None, attention_mask=attention_mask)
		all_layers=sequence_outputs
		out=torch.matmul(transforms,all_layers)
		return out

### With this class existing, I can now generate these embeddings and save to my 

def write_chunks_to_files(text, tokenizer, max_chunk_size, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sentences = text.split('.')
    chunk_id = 0
    current_chunk = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        # Add special tokens
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]

        if len(current_chunk) + len(tokenized_sentence) > max_chunk_size:
            with open(os.path.join(output_dir, f'chunk_{chunk_id}.txt'), 'w', encoding='utf-8') as file:
                file.write(' '.join(current_chunk))
            chunk_id += 1
            current_chunk = tokenized_sentence
        else:
            current_chunk.extend(tokenized_sentence)

    if current_chunk:
        with open(os.path.join(output_dir, f'chunk_{chunk_id}.txt'), 'w', encoding='utf-8') as file:
            file.write(' '.join(current_chunk))


latinBERT = LatinBERT(tokenizerPath="/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder", bertPath="/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/")



#with open('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/allletters.txt', 'r', encoding='utf-8', errors = "ignore" ) as letter:
#    letter = letter.read()
#with open('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/allphilosophies.txt', 'r', encoding='utf-8', errors = "ignore" ) as phil:
#    phil_txt = phil.read()
#with open('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/allspeeches.txt', 'r', encoding='utf-8', errors = "ignore" ) as speech:
#    speech_txt = speech.read()
#
#write_chunks_to_files(speech_txt, latinBERT.wp_tokenizer, 256, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/speeches/')
#write_chunks_to_files(letter, latinBERT.wp_tokenizer, 256, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/letters/')
#write_chunks_to_files(phil_txt, latinBERT.wp_tokenizer, 256, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/philosophies/')

def generate_and_store_embeddings(chunk_dir, latinBERT, embeddings_dir):
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    for chunk_file_name in os.listdir(chunk_dir):
        chunk_file_path = os.path.join(chunk_dir, chunk_file_name)
        if os.path.isfile(chunk_file_path):  # Ensure it's a file
            try:
                with open(chunk_file_path, 'r', encoding='utf-8') as file:
                    chunk_text = file.read()
                    print(f"Processing {chunk_file_name}: Length = {len(chunk_text.split())}")
                    embedding = latinBERT.get_berts([chunk_text])

                embedding_file_name = f'embedding_{chunk_file_name}'
                embedding_file_path = os.path.join(embeddings_dir, embedding_file_name)
                with open(embedding_file_path, 'wb') as file:
                    pickle.dump(embedding, file)
            except Exception as e:
                print(f"Error processing {chunk_file_name}: {e}")
                continue
# generate_and_store_embeddings('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/speeches/', latinBERT, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/speech_embeddings/')
# generate_and_store_embeddings('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/philosophies/', latinBERT, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/phil_embeddings/')
# generate_and_store_embeddings('/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/letters/', latinBERT, '/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/letter_embeddings/')



if __name__ == "__main__":
     chunk_dir = "/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/speeches/"
     embeddings_dir = "/slipstream_old/home/juliusherzog/latinbert/Cicero_texts/chunked_texts/speech_embeddings/"
     latinBERT = LatinBERT("/slipstream_old/home/juliusherzog/latinbert/models/subword_tokenizer_latin/latin.subword.encoder", "/slipstream_old/home/juliusherzog/latinbert/models/latin_bert/")
     generate_and_store_embeddings(chunk_dir, latinBERT, embeddings_dir)











