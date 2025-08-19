import math
import mlx.core as mx
import mlx.nn as nn

# implement the sinusoidal embedding as described in the paper "Attention Is All You Need"

def create_positional_encoding(seq_len: int, embed_dim: int):
    """
    Creates a positional encoding tensor of shape (1, seq_len, embed_dim) efficiently.

    Args:
        seq_len (int): The sequence length.
        embed_dim (int): The embedding dimension. Must be even.

    Returns:
        mx.array: The positional encoding tensor.
    """
    if embed_dim % 2 != 0:
        raise ValueError("The embedding dimension embed_dim must be an even number.")

    # `pos` represents the position in the sequence.
    pos = mx.arange(seq_len).reshape(-1, 1) # Shape: (seq_len, 1)

    # `i` represents the dimension index. We only need embed_dim/2 values because the same frequency is used for each sin/cos pair.
    div_term_indices = mx.arange(0, embed_dim, 2) # Shape: (embed_dim/2,)

    # Calculate the denominator (the inverse frequencies).
    inv_freq = 1.0 / (10000**(div_term_indices / embed_dim)) # Shape: (embed_dim/2,)

    # Calculate the arguments for sin and cos using broadcasting.
    # `position` (seq_len, 1) broadcasts with `inv_freq` (embed_dim/2,)
    angles = pos * inv_freq # shape: (seq_len, embed_dim/2)

    # Apply sin to even indices and cos to odd indices.
    pe = mx.zeros((seq_len, embed_dim))
    pe[:, 0::2] = mx.sin(angles) # Even columns
    pe[:, 1::2] = mx.cos(angles) # Odd columns

    # Step 5: Add the batch dimension to get the final shape (1, seq_len, embed_dim).
    pe = pe[None, :, :]

    return pe

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        super(SinusoidalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Create a matrix of shape (1, max_len, embed_dim)
        self.pos_embed = create_positional_encoding(max_len, embed_dim)

    def __call__(self, x):
        # x is of shape (batch_size, seq_len, embed_dim)
        # Ensure x is within the range of max_len
        if x.shape[1] > self.max_len:
            raise ValueError("Input sequence length > max_len.")

        # add sinusoidal embeddings to the input
        # print(f'x.shape: {x.shape}, pos_embed.shape: {self.pos_embed.shape}')
        return x + self.pos_embed[:, :x.shape[1], :]

class SwiGLU(nn.Module):
    """
    A SwiGLU layer, which is a modern and effective activation function found in many large language models like Llama and Mixtral.
    """
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        # The gate projection layer
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=False)
        # The up projection layer
        self.up_proj = nn.Linear(dims, hidden_dims, bias=False)
        # The down projection layer
        self.down_proj = nn.Linear(hidden_dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Calculate the gate and the up projection in parallel
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply the SwiGLU activation
        fused_swiglu = nn.silu(gate) * up

        # Apply the final down projection
        return self.down_proj(fused_swiglu)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()

        # Ensure dims are divisible by num_heads
        if embed_dims % num_heads != 0:
            raise ValueError(
                "The embedding dimension `dims` must be divisible by `num_heads`."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        # 1. Create a single, fused linear layer for Q, K, and V.
        # It projects from `dims` to `3 * dims` because it handles all three matrices.
        # This is more efficient than having separate linear layers for Q, K, and V
        # but it restricts the model to use the same dimension for all three.
        self.qkv_proj = nn.Linear(embed_dims, 3 * embed_dims, bias=False)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=False)

    def __call__(self, x: mx.array, mask: mx.array = None):
        batch_size, seq_len, embed_dim = x.shape

        # Perform one matrix multiplication to get Q, K, and V combined.
        qkv = self.qkv_proj(x)

        # Split the result into Q, K, and V.
        # This is a zero-cost operation that just creates views of the tensor.
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention.
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to get shapes ready for attention score calculation.
        # (batch_size, num_heads, seq_len, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # --- Standard Attention Logic ---
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # print(f"Attention scores: {scores[0, 0, :, :]}")  # Print the first head's scores for the first batch

        if mask is not None:
            scores = scores + mask
        # print(f"Attention scores after mask: {scores[0, 0, :, :]}")

        scores = mx.softmax(scores, axis=-1)
        # print(f"Attention scores after softmax: {scores[0, 0, :, :]}")
        # print(f"Values before attention: {values[0, 0, :, :]}") # Print the first head's values for the first batch
        # print(f"Values shape: {values.shape}")  # Should be (batch_size, num_heads, seq_len, head_dim)
        output = (scores @ values).transpose(0, 2, 1, 3)
        # print(f"Output.shape after attention: {output.shape}")  # Should be (batch_size, seq_len, num_heads, head_dim)
        # print(f"Output before attention reshape: {output[0, :, 0, :]}")  # Print the first head's output for the first batch
        output = output.reshape(batch_size, seq_len, embed_dim)
        # print(f"Output after attention reshape: {output[0, :, :]}")
        # print(f"Output shape after attention reshape: {output.shape}")

        return self.out_proj(output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_head: int, mlp_dim: int, max_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.mlp_dim = mlp_dim
        self.max_len = max_len

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dims=embed_dim, num_heads=n_head)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            SwiGLU(mlp_dim, mlp_dim),
            nn.Linear(mlp_dim, embed_dim)
        )

    def __call__(self, x: mx.array, mask: mx.array = None):
        # print(f'Input shape: {x.shape}')
        x = self.ln1(x)
        attn_output = self.mha(x, mask)
        # print(f'After MultiHeadAttention, attn_output.shape: {attn_output.shape}')
        x = x + attn_output
        x = self.ln2(x)
        x = x + self.ffn(x)
        # print(f'After FeedForward Network, x.shape: {x.shape}')
        return x

class TransformerEncoder(nn.Module):
    '''
    Simple transformer encoder without the use of mx built in layers
    with Multi-Head Self-Attention (MHSA) and a feed-forward network
    '''
    def __init__(self, embed_dim: int, n_head: int, num_layers: int, max_len: int = 512, mlp_dim: int = 2048):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.max_len = max_len
        self.pos_embed = SinusoidalEmbedding(embed_dim, max_len)
        self.layers = [TransformerEncoderLayer(embed_dim, n_head, mlp_dim, max_len) for _ in range(num_layers)]

    def __call__(self, x, mask: mx.array = None):
        # x is of shape (batch_size, seq_len, vocab_dim)
        # print(f'Input x.shape: {x.shape}')

        # add sinusoidal embeddings
        x = self.pos_embed(x)

        # go through each layer of the transformer encoder
        for layer in self.layers:
            x = layer(x, mask)
        return x

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_dim: int, embed_dim: int = 512, n_head: int = 8, num_layers: int = 6, mlp_dim: int = 2048, max_len: int = 512):
        super(SmallLanguageModel, self).__init__()
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.max_len = max_len

        # self.softmax = nn.Softmax(axis=-1)
        # self.transformer_encoder = TransformerEncoder(vocab_dim, embed_dim, n_head, num_layers, max_len, kq_dim, mlp_dim)
        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        self.transformer_layer = TransformerEncoder(embed_dim, n_head, num_layers, max_len, mlp_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, vocab_dim, bias=False),  # Output projection layer
            nn.Softmax()  # Softmax to convert logits to probabilities
        )

    def __call__(self, x, mask: mx.array = None):
        # x is of shape (batch_size, seq_len)
        # print(f'Input x.shape: {x.shape}')
        # Convert input indices to embeddings
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed
        # print(f'After embedding, x.shape: {x.shape}')
        # Pass through the transformer encoder layer
        x = self.transformer_layer(x, mask)
        # print(f'After transformer layer, x.shape: {x.shape}')
        # Pass through the output projection layer
        x = self.output_proj(x)  # Shape: (batch_size, seq_len, vocab_dim)
        # print(f'After output projection, x.shape: {x.shape}')
        return x

def count_parameters(x):
    total_params = 0
    if isinstance(x, dict):
        for key, value in x.items():
            if isinstance(value, mx.array):
                total_params += value.size
            elif isinstance(value, dict):
                total_params += count_parameters(value)
    return total_params

def create_causal_mask_triu(L: int):
    # Create a boolean matrix where the upper triangle (excluding the diagonal) is True
    mask = mx.triu(mx.ones((L, L)), k=1).astype(mx.bool_)
    return mx.where(mask, -1e9, 0.0)[None, None, :, :]  # Add batch and head dimensions

def loss_fn(model: nn.Module, input: mx.array, target: mx.array, pad_token_id: int):
    """
    Computes the cross-entropy loss for the model's predictions against the target labels.

    Args:
        model (nn.Module): The transformer model.
        input (mx.array): Input sequence of shape (batch_size, seq_len, embed_dim).
        target (mx.array): Target sequence of shape (batch_size, seq_len).
        pad_token_id (int): The ID of the padding token.

    Returns:
        mx.array: The computed loss.
    """
    # Create mask to prevent attention to future tokens
    mask = create_causal_mask_triu(input.shape[1])
    logits = model(input, mask)

    # Compute the cross-entropy loss including padded tokens
    loss = nn.losses.cross_entropy(logits, target, reduction='none')

    # Create a padding mask to ignore padded tokens in the loss calculation
    padding_mask = (target != pad_token_id).astype(mx.float32)
    # print(loss.shape, padding_mask.shape)  # Debugging: Check shapes of loss and padding_mask
    # print(f"Padding mask: {padding_mask[:2, :]}")  # Debugging: Check the first sequence's padding mask

    # Apply the padding mask to the loss
    loss = loss * padding_mask

    return loss.mean()

def generate_story(model, tokenizer, prompt, max_length, eos_token_id=None, temp=None):
    '''
    Generates a story using the model and tokenizer based on the provided prompt.
    '''
    # print(f"Generating story with prompt: {prompt}")
    model.eval()  # Set the model to evaluation mode
    tokens = tokenizer.encode(prompt).ids
    input_ids = mx.array(tokens).reshape(1, -1)  # Reshape for batch size of 1

    for i in range(max_length-len(tokens)):
        mask = create_causal_mask_triu(input_ids.shape[1])
        logits = model(input_ids, mask)
        next_token_logits = logits[:, -1, :]  # Get the logits for the last token
        if temp is None:
            next_token_id = mx.argmax(next_token_logits, axis=-1).astype(mx.int32)
        else:
            next_token_logits = next_token_logits / temp
            next_token_id = mx.random.categorical(next_token_logits, num_samples=1).astype(mx.int32)[0]
        # print(f"Next token ID: {next_token_id} with shape {next_token_id.shape}")
        # print(tokenizer.decode(next_token_id.tolist()), end='', flush=True)
        # if (i+1) % 50 == 0:
        #     print()

        if eos_token_id is not None and next_token_id == eos_token_id:
            break  # Stop if we hit the eos token

        # print(input_ids.shape, next_token_id.reshape(1, 1).shape)
        input_ids = mx.concat([input_ids, next_token_id.reshape(1, 1)], axis=1)  # Append the new token
    story = tokenizer.decode(input_ids[0].tolist())
    # print a new line after every 30 words in story
    words = story.split()
    for i in range(0, len(words), 30):
        print(' '.join(words[i:i+30]), end=' ')
        if (i + 30) % 30 == 0:
            print()
    print()  # Print a newline at the end of the story
    print('-' * 20)