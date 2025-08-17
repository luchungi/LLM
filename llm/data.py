
import mlx.data as dx
from tokenizers import Tokenizer

def chunk_story(text, tokenizer, eos_token, pad_token, context_len):
    """
    Chunk a single story into fixed-length sequences with proper EOS and padding.
    Returns a list of token chunks.
    """
    pad_token_id = tokenizer.token_to_id(pad_token)
    eos_token_id = tokenizer.token_to_id(eos_token)
    # print(f"Padding token ID: {pad_token_id}, EOS token ID: {eos_token_id}")

    # Tokenize the story
    tokens = tokenizer.encode(text).ids

    # Add EOS token to the end of the story
    tokens.append(eos_token_id)

    # Create chunks of fixed length
    chunks = []
    for i in range(0, len(tokens), context_len):
        chunk = tokens[i:i + context_len]

        # If this is the last chunk and it's not full length, pad it
        if len(chunk) < context_len:
            padding_length = context_len - len(chunk)
            chunk = chunk + [pad_token_id] * padding_length

        chunks.append(chunk)

    return chunks

def data_to_array_of_dict(dataset, name='seq'):
    return [{name: sequence} for sequence in dataset]