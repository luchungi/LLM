import math
import mlx.core as mx
import mlx.nn as nn

class SinusoidalEmbedding(nn.Module):
    '''
    Implement the sinusoidal embedding as described in the paper "Attention Is All You Need"
    '''
    def __init__(self, embed_dim: int, max_len: int):
        super(SinusoidalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Create a matrix of shape (1, max_len, embed_dim)
        self.pos_embed = self.create_positional_encoding(max_len, embed_dim)

    def __call__(self, x):
        return self.pos_embed[:, :x.shape[1], :]

    def create_positional_encoding(self, seq_len: int, embed_dim: int):
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
    def __init__(self, embed_dim: int, num_heads: int, qk_head_dim: int, v_head_dim: int):
        super().__init__()

        # # Ensure dims are divisible by num_heads
        # if embed_dims % num_heads != 0:
        #     raise ValueError(
        #         "The embedding dimension `dims` must be divisible by `num_heads`."
        #     )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_dim = num_heads * qk_head_dim  # Total dimension for query and key
        self.v_dim = num_heads * v_head_dim  # Total dimension for value

        self.q_proj = nn.Linear(embed_dim, self.qk_dim, bias=False)  # Query projection
        self.k_proj = nn.Linear(embed_dim, self.qk_dim, bias=False)  # Key projection
        self.v_proj = nn.Linear(embed_dim, self.v_dim, bias=False)  # Value projection

        # Output projection layer
        self.out_proj = nn.Linear(self.v_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array):
        batch_size, seq_len, embed_dim = x.shape

        queries = self.q_proj(x)  # Shape: (batch_size, seq_len, qk_dim)
        keys = self.k_proj(x)  # Shape: (batch_size, seq_len, qk_dim)
        values = self.v_proj(x)  # Shape: (batch_size, seq_len, v_dim)

        # Reshape for multi-head attention.
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.v_head_dim)

        # Transpose to get shapes ready for attention score calculation.
        # (batch_size, num_heads, seq_len, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # --- Standard Attention Logic ---
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.qk_dim)
        # print(f"Attention scores: {scores[0, 0, :, :]}")  # Print the first head's scores for the first batch

        scores = scores + self.create_causal_mask_triu(seq_len)  # Add causal mask
        # print(f"Attention scores after mask: {scores[0, 0, :, :]}")

        scores = mx.softmax(scores, axis=-1)
        # print(f"Attention scores after softmax: {scores[0, 0, :, :]}")
        # print(f"Values before attention: {values[0, 0, :, :]}") # Print the first head's values for the first batch
        # print(f"Values shape: {values.shape}")  # Should be (batch_size, num_heads, seq_len, head_dim)
        output = (scores @ values).transpose(0, 2, 1, 3)
        # print(f"Output.shape after attention: {output.shape}")  # Should be (batch_size, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.v_dim)
        # print(f"Output before attention reshape: {output[0, :, 0, :]}")  # Print the first head's output for the first batch
        # print(f"Output after attention reshape: {output[0, :, :]}")
        # print(f"Output shape after attention reshape: {output.shape}")
        return self.out_proj(output)

    def create_causal_mask_triu(self, L: int):
        # Create a boolean matrix where the upper triangle (excluding the diagonal) is True
        mask = mx.triu(mx.ones((L, L)), k=1).astype(mx.bool_)
        return mx.where(mask, -1e9, 0.0)[None, None, :, :]  # Add batch and head dimensions

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_head: int, qk_head_dim: int, v_head_dim: int, mlp_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.mlp_dim = mlp_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, n_head, qk_head_dim, v_head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Activation function
            # SwiGLU(mlp_dim, mlp_dim),
            nn.Linear(mlp_dim, embed_dim)
        )

    def __call__(self, x: mx.array):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_dim: int, embed_dim: int = 512, n_head: int = 8, num_layers: int = 6, qk_head_dim: int = 32, v_head_dim: int = 64, mlp_dim: int = 2048, max_len: int = 512):
        super(SmallLanguageModel, self).__init__()
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_dim = mlp_dim
        self.max_len = max_len

        # self.softmax = nn.Softmax(axis=-1)
        # self.transformer_encoder = TransformerEncoder(vocab_dim, embed_dim, n_head, num_layers, max_len, kq_dim, mlp_dim)
        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        self.pos_embed = SinusoidalEmbedding(embed_dim, max_len)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_head, qk_head_dim, v_head_dim, mlp_dim) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_dim, bias=False)  # Output projection layer

    def __call__(self, x):
        # x is of shape (batch_size, seq_len)
        # print(f'Input x.shape: {x.shape}')
        # Convert input indices to embeddings
        x = self.embedding(x) + self.pos_embed(x)  # Shape: (batch_size, seq_len, embed_dim)
        # print(f'After embedding, x.shape: {x.shape}')
        # Pass through transformer blocks
        x = self.transformer_blocks(x)  # Shape: (batch_size, seq_len, embed_dim)
        # print(f'After transformer blocks, x.shape: {x.shape}')
        # Apply output projection and softmax
        x = self.output_proj(self.layer_norm(x))  # Shape: (batch_size, seq_len, vocab_dim)
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
            elif isinstance(value, list):
                for item in value:
                    total_params += count_parameters(item)
    return total_params

def loss_fn(model: nn.Module, input: mx.array, target: mx.array, pad_token_id: int = None):
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
    logits = model(input)
    # Reshape logits to (batch_size * seq_len, vocab_dim) and target to
    # (batch_size * seq_len) for cross-entropy loss calculation
    logits = logits.reshape(-1, logits.shape[-1])  # Shape: (batch_size * seq_len, vocab_dim)
    target = target.reshape(-1)  # Shape: (batch_size * seq_len)
    # print(f"Logits shape: {logits.shape}")  # Debugging: Check logits shape
    # print(f"Target shape: {target.shape}")  # Debugging: Check target shape

    if pad_token_id is not None:
        # Create a mask to ignore padding tokens in the loss calculation
        loss = nn.losses.cross_entropy(logits, target, reduction='none')
        mask = (target != pad_token_id)
        loss = (loss * mask).sum() / mx.sum(mask)
    else:
    # Compute the cross-entropy loss including padded tokens
        loss = nn.losses.cross_entropy(logits, target, reduction='mean')

    return loss

def generate_story(model, tokenizer, prompt, max_length, eos_token_id=None, temp=None):
    '''
    Generates a story using the model and tokenizer based on the provided prompt.
    '''
    # print(f"Generating story with prompt: {prompt}")
    model.eval()  # Set the model to evaluation mode
    tokens = tokenizer.encode(prompt).ids
    input_ids = mx.array(tokens).reshape(1, -1)  # Reshape for batch size of 1

    for i in range(max_length-len(tokens)):
        logits = model(input_ids)
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