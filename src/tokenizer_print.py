import random

# Load or create your tokenizer
tokenizer = ChessTokenizer.from_pretrained("path/to/tokenizer")
# Or build from dataset:
# tokenizer = ChessTokenizer.build_vocab_from_dataset()

# Get all tokens (excluding special tokens)
vocab = tokenizer.get_vocab()
special_tokens = {tokenizer.PAD_TOKEN, tokenizer.BOS_TOKEN, 
                  tokenizer.EOS_TOKEN, tokenizer.UNK_TOKEN}
regular_tokens = [token for token in vocab.keys() 
                  if token not in special_tokens]

# Sample random tokens
num_samples = 20
sampled_tokens = random.sample(regular_tokens, num_samples)

# Print them
print("Randomly sampled tokens:")
for token in sampled_tokens:
    token_id = vocab[token]
    print(f"  {token} (ID: {token_id})")

# Or sample by IDs and convert back to tokens
random_ids = random.sample(range(4, tokenizer.vocab_size), num_samples)
print("\nRandom tokens by ID:")
for token_id in random_ids:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    print(f"  ID {token_id}: {token}")