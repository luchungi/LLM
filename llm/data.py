
import mlx.data as dx
from tokenizers import Tokenizer

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

# def chunk_story(text, tokenizer, sos_token, eos_token, pad_token, context_len):
#     """
#     Chunk a single story into fixed-length sequences with proper EOS and padding.
#     Returns a list of token chunks.
#     """
#     sos_token_id = tokenizer.token_to_id(sos_token)
#     eos_token_id = tokenizer.token_to_id(eos_token)
#     pad_token_id = tokenizer.token_to_id(pad_token)
#     # print(f"Padding token ID: {pad_token_id}, EOS token ID: {eos_token_id}")

#     # Tokenize the story
#     tokens = [sos_token_id]  # Start with SOS token
#     story_tokens = tokenizer.encode(text).ids
#     count = len(story_tokens)
#     tokens = tokens + story_tokens

#     # Add EOS token to the end of the story
#     tokens.append(eos_token_id)

#     # Create chunks of fixed length
#     chunks = []
#     for i in range(0, len(tokens), context_len):
#         chunk = tokens[i:i + context_len]

#         # If this is the last chunk and it's not full length, pad it
#         if len(chunk) < context_len:
#             padding_length = context_len - len(chunk)
#             chunk = chunk + [pad_token_id] * padding_length

#         chunks.append(chunk)

#     return chunks, count

def chunk_story(text, tokenizer, sos_token, eos_token, context_len, unfinished_chunk=None):
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

    # Tokenize the story
    story_tokens = tokenizer.encode(text).ids
    count = len(story_tokens)

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

def data_to_array_of_dict(dataset, name='seq'):
    return [{name: sequence} for sequence in dataset]