import re
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def normalize_text(text: str) -> str:
	text = text.lower()
	text = re.sub(r"(?m)^\s*\d+[.,]{1,2}\s*", "", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def tokenize_words_and_punctuation(text: str):
	return re.findall(r"\w+|[.,!?;:]", text, flags=re.UNICODE)


def map_tokens_for_language_model(tokens, train_word_frequency, minimum_word_frequency):
	mapped_tokens = []
	for token in tokens:
		if train_word_frequency.get(token, 0) >= minimum_word_frequency:
			mapped_tokens.append(token)
		else:
			mapped_tokens.append("<unk>")
	return mapped_tokens


def build_vocab(train_tokens):
	unique_words = sorted(set(train_tokens))
	if "<unk>" not in unique_words:
		unique_words = ["<unk>"] + unique_words
	return unique_words


class NGramTextDataset(Dataset):
	def __init__(self, tokens, context_size, word_to_index):
		self.samples = []
		unknown_index = word_to_index["<unk>"]

		for token_position in range(context_size, len(tokens)):
			context_words = tokens[token_position - context_size:token_position]
			target_word = tokens[token_position]

			context_indices = [word_to_index.get(word, unknown_index) for word in context_words]
			target_index = word_to_index.get(target_word, unknown_index)
			self.samples.append((context_indices, target_index))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		context_indices, target_index = self.samples[index]
		return (
			torch.tensor(context_indices, dtype=torch.long),
			torch.tensor(target_index, dtype=torch.long)
		)


class NGramLanguageModel(nn.Module):
	def __init__(self, vocabulary_size, context_size, embedding_dimension=128, hidden_dimension=256):
		super().__init__()
		self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
		self.hidden_layer = nn.Linear(context_size * embedding_dimension, hidden_dimension)
		self.output_layer = nn.Linear(hidden_dimension, vocabulary_size)
		self.activation = nn.ReLU()

	def forward(self, context_indices):
		embedded = self.embedding(context_indices)
		flattened = embedded.view(embedded.size(0), -1)
		hidden_state = self.activation(self.hidden_layer(flattened))
		logits = self.output_layer(hidden_state)
		return logits


class NGramTrainer:
	def __init__(self, model, device):
		self.model = model
		self.device = device
		self.loss_function = nn.CrossEntropyLoss()

	def train(self, train_loader, number_of_epochs=12, learning_rate=0.002):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		training_losses = []

		for epoch in range(number_of_epochs):
			self.model.train()
			total_loss = 0.0

			for context_batch, target_batch in train_loader:
				context_batch = context_batch.to(self.device)
				target_batch = target_batch.to(self.device)

				optimizer.zero_grad()
				logits = self.model(context_batch)
				loss = self.loss_function(logits, target_batch)
				loss.backward()
				optimizer.step()

				total_loss += loss.item()

			average_loss = total_loss / len(train_loader)
			training_losses.append(average_loss)

		return training_losses

	def calculate_perplexity(self, data_loader):
		self.model.eval()
		total_loss = 0.0

		with torch.no_grad():
			for context_batch, target_batch in data_loader:
				context_batch = context_batch.to(self.device)
				target_batch = target_batch.to(self.device)
				logits = self.model(context_batch)
				loss = self.loss_function(logits, target_batch)
				total_loss += loss.item()

		average_loss = total_loss / len(data_loader)
		perplexity = torch.exp(torch.tensor(average_loss)).item()
		return perplexity

	def generate_sentence(
		self,
		seed_words,
		word_to_index,
		index_to_word,
		context_size,
		max_new_tokens=30,
		temperature=0.9,
		stop_tokens=None
	):
		if stop_tokens is None:
			stop_tokens = {".", "!", "?"}

		self.model.eval()

		if isinstance(seed_words, str):
			generated_words = seed_words.lower().split()
		else:
			generated_words = [word.lower() for word in seed_words]

		unknown_index = word_to_index["<unk>"]

		with torch.no_grad():
			for _ in range(max_new_tokens):
				current_context_words = generated_words[-context_size:]
				if len(current_context_words) < context_size:
					current_context_words = ["<unk>"] * (context_size - len(current_context_words)) + current_context_words

				current_context_indices = [word_to_index.get(word, unknown_index) for word in current_context_words]
				context_tensor = torch.tensor([current_context_indices], dtype=torch.long, device=self.device)

				logits = self.model(context_tensor).squeeze(0)
				scaled_logits = logits / temperature
				probabilities = torch.softmax(scaled_logits, dim=0)

				probabilities[unknown_index] = 0.0
				probability_sum = probabilities.sum()
				if probability_sum > 0:
					probabilities = probabilities / probability_sum
				else:
					probabilities = torch.softmax(scaled_logits, dim=0)

				next_word_index = torch.multinomial(probabilities, num_samples=1).item()
				next_word = index_to_word[next_word_index]
				generated_words.append(next_word)

				if next_word in stop_tokens:
					break

		generated_sentence = " ".join(generated_words)
		generated_sentence = re.sub(r"\s+([.,!?;:])", r"\1", generated_sentence)
		return generated_sentence


def run(
	dataset_file_path="truyen_kieu_data.txt",
	n_gram_size=2,
	minimum_word_frequency=2,
	number_of_epochs=12,
	learning_rate=0.002,
	batch_size=512,
	random_seed=42,
	seed_examples=None
):
	torch.manual_seed(random_seed)
	random.seed(random_seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	raw_text = Path(dataset_file_path).read_text(encoding="utf-8")
	clean_text = normalize_text(raw_text)
	all_tokens = tokenize_words_and_punctuation(clean_text)

	split_index = int(0.8 * len(all_tokens))
	train_tokens = all_tokens[:split_index]
	test_tokens = all_tokens[split_index:]

	train_word_frequency = Counter(train_tokens)
	train_tokens_for_model = map_tokens_for_language_model(
		train_tokens,
		train_word_frequency,
		minimum_word_frequency
	)
	test_tokens_for_model = map_tokens_for_language_model(
		test_tokens,
		train_word_frequency,
		minimum_word_frequency
	)

	context_size = n_gram_size - 1
	vocabulary = build_vocab(train_tokens_for_model)
	word_to_index = {word: index for index, word in enumerate(vocabulary)}
	index_to_word = {index: word for word, index in word_to_index.items()}

	train_dataset = NGramTextDataset(train_tokens_for_model, context_size, word_to_index)
	test_dataset = NGramTextDataset(test_tokens_for_model, context_size, word_to_index)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	model = NGramLanguageModel(
		vocabulary_size=len(vocabulary),
		context_size=context_size,
		embedding_dimension=128,
		hidden_dimension=256
	).to(device)

	trainer = NGramTrainer(model=model, device=device)
	training_losses = trainer.train(
		train_loader=train_loader,
		number_of_epochs=number_of_epochs,
		learning_rate=learning_rate
	)
	test_perplexity = trainer.calculate_perplexity(test_loader)

	if seed_examples is None:
		seed_examples = [["trăm", "năm"], ["người", "đâu"], ["một", "đời"], ["chữ", "tài"]]

	generated_sentences = []
	for seed_words in seed_examples:
		generated_sentence = trainer.generate_sentence(
			seed_words=seed_words,
			word_to_index=word_to_index,
			index_to_word=index_to_word,
			context_size=context_size,
			max_new_tokens=28,
			temperature=0.9
		)
		generated_sentences.append((seed_words, generated_sentence))

	return {
		"device": str(device),
		"token_count": len(all_tokens),
		"train_token_count": len(train_tokens),
		"test_token_count": len(test_tokens),
		"vocabulary_size": len(vocabulary),
		"train_sample_count": len(train_dataset),
		"test_sample_count": len(test_dataset),
		"training_losses": training_losses,
		"test_perplexity": test_perplexity,
		"generated_sentences": generated_sentences
	}

if __name__ == "__main__":
	run()