
import os
import json
import inspect
import mlx.data as dx
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

def train_tokenizer(dataset, vocab_size, special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]):
    """
    Train a tokenizer on the given dataset and save it to a file.

    Args:
        dataset: The dataset containing text data.
        tokenizer_file: Path to save the trained tokenizer.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to include in the tokenizer.
        batch_size: Number of samples per batch for training.
    """
        # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace(replacement=" ")
    tokenizer.decoder = MetaspaceDecoder(replacement=" ")

    # Configure the trainer with a vocabulary size and special tokens
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Train the tokenizer on our text file
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))
    # tokenizer.train(['./data/tinystories_data.txt'], trainer)
    print("Training complete.")
    return tokenizer

def encode_story(text, tokenizer, sos_token, eos_token):
    sos_token_id = tokenizer.token_to_id(sos_token)
    eos_token_id = tokenizer.token_to_id(eos_token)
    story_tokens = tokenizer.encode(text).ids
    count = len(story_tokens)
    tokens = [sos_token_id] + story_tokens + [eos_token_id]
    return tokens, count

def chunk_story(text, tokenizer, sos_token, eos_token, context_len, unfinished_chunk=None, padding=False, pad_token=None):
    """
    Chunk a single story into fixed-length sequences with proper EOS and padding.

    Parameters:
        text: The story text to tokenize and chunk
        tokenizer: The tokenizer to use
        sos_token: Start of sequence token
        eos_token: End of sequence token
        pad_token: Padding token
        context_len: The fixed length for each chunk
        unfinished_chunk: Optional list of tokens from previous call that needs to be completed

    Returns:
        chunks: List of complete chunks (each of length context_len)
        unfinished_chunk: Remaining tokens that don't fill a complete chunk (or None)
        count: Number of actual story tokens processed (excluding special tokens)
    """
    sos_token_id = tokenizer.token_to_id(sos_token)
    eos_token_id = tokenizer.token_to_id(eos_token)
    if padding and pad_token is None:
        raise ValueError("Padding is enabled but no pad_token provided.")
    else:
        pad_token_id = tokenizer.token_to_id(pad_token)

    # Tokenize the story
    story_tokens = tokenizer.encode(text).ids
    count = len(story_tokens)

    if padding:
        tokens = [sos_token_id]  # Start with SOS token
        tokens = tokens + story_tokens

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

    else:
        # Start with unfinished chunk or create a new one
        chunks = []
        current_tokens = []

        if unfinished_chunk is not None:
            # Continue from unfinished chunk
            current_tokens = unfinished_chunk.copy()
            current_tokens.append(sos_token_id)  # Ensure SOS token is at the start
        else:
            # Start a new chunk with SOS token
            current_tokens = [sos_token_id]

        # Process story tokens
        token_index = 0
        while token_index < len(story_tokens):
            # Add tokens to the current chunk until it reaches context_len
            space_left = context_len - len(current_tokens)

            if space_left > 0:
                # Add as many tokens as will fit
                tokens_to_add = story_tokens[token_index:token_index + space_left]
                current_tokens.extend(tokens_to_add)
                token_index += len(tokens_to_add)

                # If we've reached context_len, save this chunk and start a new one
                if len(current_tokens) == context_len:
                    chunks.append(current_tokens)
                    current_tokens = []  # Start a new chunk with SOS token
            else:
                # Current chunk is full, save it and start a new one
                chunks.append(current_tokens)
                current_tokens = []  # Start a new chunk with SOS token

        # Handle the last chunk
        if len(current_tokens) > 0:
            if current_tokens[-1] != eos_token_id:  # Only add EOS if not already there
                # Check if there's space for EOS token
                if len(current_tokens) < context_len:
                    current_tokens.append(eos_token_id)
                else:
                    # remove the first token to make space for EOS
                    current_tokens.pop(0)
                    current_tokens.append(eos_token_id)

            # If the last chunk is full, add it to chunks
            if len(current_tokens) == context_len:
                chunks.append(current_tokens)
                unfinished_chunk = None
            else:
                # Return as unfinished_chunk (don't pad it)
                unfinished_chunk = current_tokens
        else:
            unfinished_chunk = None

    return chunks, unfinished_chunk, count

def pack_stories(lists, max_len, padding_id):
    """
    Combine variable-length lists to maximize utilization of fixed length slots.

    Args:
        lists: List of lists with varying lengths
        max_len: Maximum allowed length
        padding_id: Token ID to use for padding

    Returns:
        List of padded lists, each of length exactly max_len
    """
    # Step 1: Filter out lists that are longer than max_len
    valid_lists = [lst for lst in lists if len(lst) <= max_len]

    # Step 2: Sort by length in descending order for better bin packing
    valid_lists.sort(key=len, reverse=True)

    # Step 3: Best-fit bin packing algorithm
    result = []

    for lst in valid_lists:
        # Find the best fit (list in result that would have least space left after combining)
        best_fit_idx = -1
        min_space_left = max_len + 1

        for i, res_lst in enumerate(result):
            space_left = max_len - len(res_lst) - len(lst)
            if space_left >= 0 and space_left < min_space_left:
                best_fit_idx = i
                min_space_left = space_left

        if best_fit_idx != -1:
            # Combine with the best fit
            result[best_fit_idx].extend(lst)
        else:
            # Start a new list
            result.append(lst.copy())

    # Step 4: Pad all lists to max_len
    for i in range(len(result)):
        padding_needed = max_len - len(result[i])
        result[i].extend([padding_id] * padding_needed)

    return result

def data_to_array_of_dict(dataset, name='seq'):
    return [{name: sequence} for sequence in dataset]

def create_dict_parameters(vars):
    """
    Create a dictionary of parameters from the given variables.
    """
    exclude_list = ['In', 'Out', 'TOKENIZER_FILE', 'CHUNK_FILE', 'DICT_LABEL']
    params = {k: v for k, v in vars.items() if not k.startswith('_') and not callable(v) and v is not None and not inspect.ismodule(v)}
    for item in exclude_list:
        params.pop(item, None)
    return params