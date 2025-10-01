import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism"""

    def __init__(self, d_model, num_heads, dropout=0.1, is_causal=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(1), query.size(0)
        key_seq_len = key.size(0)

        # Linear projections and split into heads
        Q = self.w_q(query)  # (seq_len, batch_size, d_model)
        K = self.w_k(key)    # (key_seq_len, batch_size, d_model)
        V = self.w_v(value)  # (key_seq_len, batch_size, d_model)

        # Store pre-split Q, K, V for snapshots
        Q_pre_split = Q.clone()
        K_pre_split = K.clone()
        V_pre_split = V.clone()

        # Reshape for multi-head: (batch_size * num_heads, seq_len, d_k)
        Q = Q.contiguous().view(seq_len, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        K = K.contiguous().view(key_seq_len, batch_size * self.num_heads, self.d_k).transpose(0, 1)
        V = V.contiguous().view(key_seq_len, batch_size * self.num_heads, self.d_k).transpose(0, 1)

        # Store multi-head split tensors
        Q_multi_head = Q.view(batch_size, self.num_heads, seq_len, self.d_k).transpose(0, 1)
        K_multi_head = K.view(batch_size, self.num_heads, key_seq_len, self.d_k).transpose(0, 1)
        V_multi_head = V.view(batch_size, self.num_heads, key_seq_len, self.d_k).transpose(0, 1)

        # Attention scores: (batch_size * num_heads, seq_len, key_seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)

        # Reshape scores for mask application and storage
        scores = scores.view(batch_size, self.num_heads, seq_len, key_seq_len)

        # Store attention scores before softmax
        scores_before_softmax = scores.clone()

        # Create and store mask tensor if causal
        mask_tensor = None
        scores_before_mask = None
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, key_seq_len) * float('-inf'), diagonal=1)
            causal_mask = causal_mask.to(scores.device)
            mask_tensor = causal_mask
            scores_before_mask = scores.clone()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, key_seq_len)
            scores = scores + causal_mask

        # Apply provided mask
        if mask is not None:
            if mask.dim() == 3:
                if mask.size(1) == seq_len and mask.size(2) == key_seq_len:
                    mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, key_seq_len)
                    mask = mask.repeat(1, self.num_heads, 1, 1)
                elif mask.size(1) == mask.size(2) and seq_len != key_seq_len:
                    if seq_len < mask.size(1):
                        mask = mask[:, :seq_len, :key_seq_len]
                    else:
                        new_mask = torch.ones(batch_size, seq_len, key_seq_len, device=mask.device)
                        min_seq = min(seq_len, mask.size(1))
                        min_key_seq = min(key_seq_len, mask.size(2))
                        new_mask[:, :min_seq, :min_key_seq] = mask[:, :min_seq, :min_key_seq]
                        mask = new_mask
                    mask = mask.unsqueeze(1)
                    mask = mask.repeat(1, self.num_heads, 1, 1)

            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = self.softmax(scores)
        attention_weights_after_softmax = attention_weights.clone()
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_weights = attention_weights.view(batch_size * self.num_heads, seq_len, key_seq_len)
        V = V.view(batch_size * self.num_heads, key_seq_len, self.d_k)

        context = torch.bmm(attention_weights, V)  # (batch_size * num_heads, seq_len, d_k)

        # Store multi-head output before concatenation
        multi_head_output_before_concat = context.clone()

        # Reshape context back to (seq_len, batch_size, d_model)
        context = context.view(batch_size, self.num_heads, seq_len, self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = context.transpose(0, 1)  # (seq_len, batch_size, d_model)

        # Store concatenated multi-head output (Snapshot #13, #28, #35)
        multi_head_concat_output = context.clone()

        output = self.w_o(context)

        return (output, attention_weights.view(batch_size, self.num_heads, seq_len, key_seq_len),
                Q_pre_split, K_pre_split, V_pre_split, scores_before_softmax,
                attention_weights_after_softmax, Q_multi_head, K_multi_head, V_multi_head,
                mask_tensor, scores_before_mask, multi_head_concat_output)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        ff_input = x.clone()
        ff_layer1 = self.linear1(x)
        ff_layer1_act = self.relu(ff_layer1)
        ff_layer1_drop = self.dropout(ff_layer1_act)
        ff_layer2 = self.linear2(ff_layer1_drop)
        return ff_layer2, ff_input, ff_layer1, ff_layer2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # BREAKPOINT: Encoder layer input (Snapshot #6)
        encoder_block_input = x.clone()

        # Self-attention sublayer with residual connection
        (attn_output, attn_weights, Q, K, V, scores_before_softmax,
         attention_weights_after_softmax, Q_multi_head, K_multi_head,
         V_multi_head, _, _, multi_head_concat_output) = self.self_attention(x, x, x, mask)

        # Residual connection
        residual_after_attn = x + self.dropout(attn_output)
        # Layer normalization
        norm_after_attn = self.norm1(residual_after_attn)

        # Feed-forward sublayer with residual connection
        ff_output, ff_input, ff_layer1, ff_layer2 = self.feed_forward(norm_after_attn)

        residual_after_ff = norm_after_attn + self.dropout(ff_output)
        encoder_block_output = self.norm2(residual_after_ff)

        return (encoder_block_output, attn_weights, Q, K, V, scores_before_softmax,
                attention_weights_after_softmax, Q_multi_head, K_multi_head, V_multi_head,
                encoder_block_input, residual_after_attn, norm_after_attn, ff_input,
                ff_layer1, ff_layer2, residual_after_ff, multi_head_concat_output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout, is_causal=True)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # BREAKPOINT: Decoder layer input (Snapshot #20)
        decoder_block_input = x.clone()

        # Masked self-attention sublayer
        (masked_attn_output, masked_attn_weights, masked_Q, masked_K, masked_V,
         masked_scores_before_softmax, masked_attention_weights_after_softmax,
         masked_Q_multi_head, masked_K_multi_head, masked_V_multi_head,
         mask_tensor, masked_scores_before_mask, masked_multi_head_concat_output) = self.masked_self_attention(x, x, x, tgt_mask)

        # Residual connection and normalization after masked self-attention
        residual_after_masked_attn = x + self.dropout(masked_attn_output)
        norm_after_masked_attn = self.norm1(residual_after_masked_attn)

        # Cross-attention sublayer
        cross_src_mask = None
        if src_mask is not None:
            batch_size = x.size(1)
            tgt_len = x.size(0)
            src_len = encoder_output.size(0)

            if src_mask.size(1) == src_mask.size(2) and src_mask.size(1) == src_len:
                cross_src_mask = src_mask[:, :tgt_len, :]
            else:
                cross_src_mask = src_mask

        (cross_attn_output, cross_attn_weights, cross_Q, cross_K, cross_V,
         cross_scores_before_softmax, cross_attention_weights_after_softmax,
         cross_Q_multi_head, cross_K_multi_head, cross_V_multi_head, _, _,
         cross_multi_head_concat_output) = self.cross_attention(
            norm_after_masked_attn, encoder_output, encoder_output, cross_src_mask)

        # Residual connection and normalization after cross-attention
        residual_after_cross_attn = norm_after_masked_attn + self.dropout(cross_attn_output)
        norm_after_cross_attn = self.norm2(residual_after_cross_attn)

        # Feed-forward sublayer
        ff_output, ff_input, ff_layer1, ff_layer2 = self.feed_forward(norm_after_cross_attn)

        residual_after_ff = norm_after_cross_attn + self.dropout(ff_output)
        decoder_block_output = self.norm3(residual_after_ff)

        return (decoder_block_output, masked_attn_weights, cross_attn_weights,
                masked_Q, masked_K, masked_V, masked_scores_before_softmax,
                masked_attention_weights_after_softmax, masked_Q_multi_head,
                masked_K_multi_head, masked_V_multi_head, mask_tensor,
                masked_scores_before_mask, cross_Q, cross_K, cross_V,
                cross_scores_before_softmax, cross_attention_weights_after_softmax,
                cross_Q_multi_head, cross_K_multi_head, cross_V_multi_head,
                decoder_block_input, residual_after_masked_attn, norm_after_masked_attn,
                residual_after_cross_attn, norm_after_cross_attn, ff_input,
                ff_layer1, ff_layer2, residual_after_ff,
                masked_multi_head_concat_output, cross_multi_head_concat_output)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src_tokens, src_mask=None):
        # Token embedding
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attention_weights = []
        all_intermediates = []

        for i, layer in enumerate(self.layers):
            if i == 0:  # Only store intermediates for first layer for snapshots
                layer_output = layer(x, src_mask)
                (x, attn_weights, Q, K, V, scores_before_softmax,
                 attention_weights_after_softmax, Q_multi_head, K_multi_head, V_multi_head,
                 encoder_block_input, residual_after_attn, norm_after_attn, ff_input,
                 ff_layer1, ff_layer2, residual_after_ff, multi_head_concat_output) = layer_output

                layer_intermediates = {
                    'attn_weights': attn_weights,
                    'Q': Q, 'K': K, 'V': V,
                    'scores_before_softmax': scores_before_softmax,
                    'attention_weights_after_softmax': attention_weights_after_softmax,
                    'Q_multi_head': Q_multi_head, 'K_multi_head': K_multi_head, 'V_multi_head': V_multi_head,
                    'encoder_block_input': encoder_block_input,
                    'residual_after_attn': residual_after_attn,
                    'norm_after_attn': norm_after_attn,
                    'ff_input': ff_input,
                    'ff_layer1': ff_layer1,
                    'ff_layer2': ff_layer2,
                    'residual_after_ff': residual_after_ff,
                    'multi_head_concat_output': multi_head_concat_output  # Snapshot #13
                }
                all_intermediates.append(layer_intermediates)
            else:
                # For subsequent layers, just get the main output
                layer_output = layer(x, src_mask)
                x = layer_output[0]  # First element is the main output
                attn_weights = layer_output[1]  # Second element is attention weights

            attention_weights.append(attn_weights)

        return x, attention_weights, all_intermediates


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt_tokens, encoder_output, src_mask=None, tgt_mask=None):
        # Token embedding
        x = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        masked_attention_weights = []
        cross_attention_weights = []
        all_intermediates = []

        for i, layer in enumerate(self.layers):
            if i == 0:  # Only store intermediates for first layer for snapshots
                layer_output = layer(x, encoder_output, src_mask, tgt_mask)
                (x, masked_attn_weights, cross_attn_weights, masked_Q, masked_K, masked_V,
                 masked_scores_before_softmax, masked_attention_weights_after_softmax,
                 masked_Q_multi_head, masked_K_multi_head, masked_V_multi_head, mask_tensor,
                 masked_scores_before_mask, cross_Q, cross_K, cross_V, cross_scores_before_softmax,
                 cross_attention_weights_after_softmax, cross_Q_multi_head, cross_K_multi_head,
                 cross_V_multi_head, decoder_block_input, residual_after_masked_attn,
                 norm_after_masked_attn, residual_after_cross_attn, norm_after_cross_attn,
                 ff_input, ff_layer1, ff_layer2, residual_after_ff,
                 masked_multi_head_concat_output, cross_multi_head_concat_output) = layer_output

                layer_intermediates = {
                    'masked_attn_weights': masked_attn_weights,
                    'cross_attn_weights': cross_attn_weights,
                    'masked_Q': masked_Q, 'masked_K': masked_K, 'masked_V': masked_V,
                    'masked_scores_before_softmax': masked_scores_before_softmax,
                    'masked_attention_weights_after_softmax': masked_attention_weights_after_softmax,
                    'masked_Q_multi_head': masked_Q_multi_head,
                    'masked_K_multi_head': masked_K_multi_head,
                    'masked_V_multi_head': masked_V_multi_head,
                    'mask_tensor': mask_tensor,
                    'masked_scores_before_mask': masked_scores_before_mask,
                    'cross_Q': cross_Q, 'cross_K': cross_K, 'cross_V': cross_V,
                    'cross_scores_before_softmax': cross_scores_before_softmax,
                    'cross_attention_weights_after_softmax': cross_attention_weights_after_softmax,
                    'cross_Q_multi_head': cross_Q_multi_head,
                    'cross_K_multi_head': cross_K_multi_head,
                    'cross_V_multi_head': cross_V_multi_head,
                    'decoder_block_input': decoder_block_input,
                    'residual_after_masked_attn': residual_after_masked_attn,
                    'norm_after_masked_attn': norm_after_masked_attn,
                    'residual_after_cross_attn': residual_after_cross_attn,
                    'norm_after_cross_attn': norm_after_cross_attn,
                    'ff_input': ff_input,
                    'ff_layer1': ff_layer1,
                    'ff_layer2': ff_layer2,
                    'residual_after_ff': residual_after_ff,
                    'masked_multi_head_concat_output': masked_multi_head_concat_output,  # Snapshot #28
                    'cross_multi_head_concat_output': cross_multi_head_concat_output     # Snapshot #35
                }
                all_intermediates.append(layer_intermediates)
            else:
                # For subsequent layers, just get the main outputs
                layer_output = layer(x, encoder_output, src_mask, tgt_mask)
                x = layer_output[0]  # Main output
                masked_attn_weights = layer_output[1]  # Masked attention weights
                cross_attn_weights = layer_output[2]   # Cross attention weights

            masked_attention_weights.append(masked_attn_weights)
            cross_attention_weights.append(cross_attn_weights)

        return x, masked_attention_weights, cross_attention_weights, all_intermediates


class DebuggableTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_encoder_layers=2,
                 num_decoder_layers=2, d_ff=512, max_seq_length=1000, dropout=0.1):
        super(DebuggableTransformer, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_encoder_layers, d_ff, max_seq_length, dropout
        )

        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, num_decoder_layers, d_ff, max_seq_length, dropout
        )

        # Final projection layer
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        # Encoder forward pass
        encoder_output, enc_attn_weights, enc_intermediates = self.encoder(src_tokens, src_mask)

        # Decoder forward pass
        decoder_output, dec_masked_attn_weights, dec_cross_attn_weights, dec_intermediates = self.decoder(
            tgt_tokens, encoder_output, src_mask, tgt_mask)

        # Final projection
        output_logits = self.output_projection(decoder_output)

        return {
            'logits': output_logits,
            'encoder_output': encoder_output,
            'decoder_output': decoder_output,
            'enc_attention_weights': enc_attn_weights,
            'dec_masked_attention_weights': dec_masked_attn_weights,
            'dec_cross_attention_weights': dec_cross_attn_weights,
            'enc_intermediates': enc_intermediates,
            'dec_intermediates': dec_intermediates
        }


def create_sample_data(vocab_size=1000, src_seq_length=8, tgt_seq_length=7, batch_size=1):
    """Create sample input and target data from programming domain"""
    torch.manual_seed(42)

    src_tokens = torch.tensor([[100, 201, 302, 403, 504, 605, 706, 807]]).transpose(0, 1)  # (seq_len, batch_size)
    tgt_tokens = torch.tensor([[150, 251, 352, 453, 554, 655, 756]]).transpose(0, 1)  # (seq_len, batch_size)

    src_mask = torch.ones(batch_size, src_seq_length, src_seq_length)
    tgt_mask = torch.ones(batch_size, tgt_seq_length, tgt_seq_length)

    return src_tokens, tgt_tokens, src_mask, tgt_mask


def debug_transformer_flow():
    """Main debugging function with all 43 snapshots explicitly captured"""
    print("Initializing Transformer Model for Debugging...")

    # Model parameters matching task requirements
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    src_seq_length = 8
    tgt_seq_length = 7
    batch_size = 1

    model = DebuggableTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff
    )

    src_tokens, tgt_tokens, src_mask, tgt_mask = create_sample_data(
        vocab_size, src_seq_length, tgt_seq_length, batch_size
    )

    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")

    model.eval()

    print("\nStarting forward pass - set breakpoints and use PyCharm debugger to inspect tensor shapes...")
    print("All 43 snapshots will be explicitly captured and available for inspection.")

    with torch.no_grad():
        # ==================================================
        # BREAKPOINT 1: For Input & Embedding snapshots (1-5)
        # SNAPSHOT 1: Raw input tokens
        snapshot_1 = src_tokens
        # SNAPSHOT 2: Target tokens
        snapshot_2 = tgt_tokens

        # EMBEDDING LAYERS
        # SNAPSHOT 3: Embedding weight matrix (slice)
        snapshot_3 = model.encoder.token_embedding.weight[:5, :5]  # 5x5 slice

        # Input embeddings
        src_embeddings = model.encoder.token_embedding(src_tokens) * math.sqrt(d_model)
        # SNAPSHOT 4: Input embeddings after lookup
        snapshot_4 = src_embeddings

        # Positional encoding
        src_embeddings_pe = model.encoder.positional_encoding(src_embeddings)
        # SNAPSHOT 5: Embeddings after adding positional encoding
        snapshot_5 = src_embeddings_pe

        # ==================================================
        # BREAKPOINT 2: Before encoder processing
        # ENCODER LAYERS (First layer for snapshots 6-19)
        print("\n--- Processing Encoder Layers ---")

        # Full model forward pass to get all intermediates
        full_output = model(src_tokens, tgt_tokens, src_mask, tgt_mask)

        # Extract encoder intermediates from first layer
        enc_intermediates = full_output['enc_intermediates'][0] if full_output['enc_intermediates'] else {}

        # SNAPSHOT 6: Encoder block input tensor
        snapshot_6 = enc_intermediates.get('encoder_block_input', None)

        # SNAPSHOT 7: Self-attention queries (Q)
        snapshot_7 = enc_intermediates.get('Q', None)
        # SNAPSHOT 8: Self-attention keys (K)
        snapshot_8 = enc_intermediates.get('K', None)
        # SNAPSHOT 9: Self-attention values (V)
        snapshot_9 = enc_intermediates.get('V', None)

        # SNAPSHOT 10: Attention score matrix before softmax
        snapshot_10 = enc_intermediates.get('scores_before_softmax', None)
        # SNAPSHOT 11: Attention score matrix after softmax
        snapshot_11 = enc_intermediates.get('attention_weights_after_softmax', None)

        # SNAPSHOT 12: Multi-head split (Q/K/V split)
        snapshot_12 = enc_intermediates.get('Q_multi_head', None)

        # SNAPSHOT 13: Multi-head attention output after concatenation
        snapshot_13 = enc_intermediates.get('multi_head_concat_output', None)

        # SNAPSHOT 14: Residual connection tensors
        snapshot_14 = enc_intermediates.get('residual_after_attn', None)

        # SNAPSHOT 15: Layer normalization output
        snapshot_15 = enc_intermediates.get('norm_after_attn', None)

        # SNAPSHOT 16: Feed-forward input
        snapshot_16 = enc_intermediates.get('ff_input', None)
        # SNAPSHOT 17: Feed-forward first linear layer output
        snapshot_17 = enc_intermediates.get('ff_layer1', None)
        # SNAPSHOT 18: Feed-forward second linear layer output
        snapshot_18 = enc_intermediates.get('ff_layer2', None)

        # SNAPSHOT 19: Encoder block final output tensor
        snapshot_19 = enc_intermediates.get('residual_after_ff', None)

        # ==================================================
        # BREAKPOINT 3: Before decoder processing
        # DECODER LAYERS (First layer for snapshots 20-40)
        print("\n--- Processing Decoder Layers ---")

        # Extract decoder intermediates from first layer
        dec_intermediates = full_output['dec_intermediates'][0] if full_output['dec_intermediates'] else {}

        # SNAPSHOT 20: Decoder block input tensor
        snapshot_20 = dec_intermediates.get('decoder_block_input', None)

        # SNAPSHOT 21: Masked self-attention queries (Q)
        snapshot_21 = dec_intermediates.get('masked_Q', None)
        # SNAPSHOT 22: Masked self-attention keys (K)
        snapshot_22 = dec_intermediates.get('masked_K', None)
        # SNAPSHOT 23: Masked self-attention values (V)
        snapshot_23 = dec_intermediates.get('masked_V', None)

        # SNAPSHOT 24: Masked attention scores before mask
        snapshot_24 = dec_intermediates.get('masked_scores_before_mask', None)
        # SNAPSHOT 25: Mask tensor
        snapshot_25 = dec_intermediates.get('mask_tensor', None)
        # SNAPSHOT 26: Masked attention scores after mask + softmax
        snapshot_26 = dec_intermediates.get('masked_attention_weights_after_softmax', None)

        # SNAPSHOT 27: Masked self-attention multi-head split
        snapshot_27 = dec_intermediates.get('masked_Q_multi_head', None)

        # SNAPSHOT 28: Masked self-attention multi-head concatenated output
        snapshot_28 = dec_intermediates.get('masked_multi_head_concat_output', None)

        # SNAPSHOT 29: Residual + normalization after masked self-attention
        snapshot_29 = dec_intermediates.get('norm_after_masked_attn', None)

        # SNAPSHOT 30: Cross-attention queries (from decoder)
        snapshot_30 = dec_intermediates.get('cross_Q', None)
        # SNAPSHOT 31: Cross-attention keys (from encoder)
        snapshot_31 = dec_intermediates.get('cross_K', None)
        # SNAPSHOT 32: Cross-attention values (from encoder)
        snapshot_32 = dec_intermediates.get('cross_V', None)

        # SNAPSHOT 33: Cross-attention score matrix before softmax
        snapshot_33 = dec_intermediates.get('cross_scores_before_softmax', None)
        # SNAPSHOT 34: Cross-attention score matrix after softmax
        snapshot_34 = dec_intermediates.get('cross_attention_weights_after_softmax', None)

        # SNAPSHOT 35: Cross-attention output after concatenation
        snapshot_35 = dec_intermediates.get('cross_multi_head_concat_output', None)

        # SNAPSHOT 36: Residual + normalization after cross-attention
        snapshot_36 = dec_intermediates.get('norm_after_cross_attn', None)

        # SNAPSHOT 37: Decoder feed-forward input
        snapshot_37 = dec_intermediates.get('ff_input', None)
        # SNAPSHOT 38: Feed-forward first linear layer output
        snapshot_38 = dec_intermediates.get('ff_layer1', None)
        # SNAPSHOT 39: Feed-forward second linear layer output
        snapshot_39 = dec_intermediates.get('ff_layer2', None)

        # SNAPSHOT 40: Decoder block final output tensor
        snapshot_40 = dec_intermediates.get('residual_after_ff', None)

        # ==================================================
        # BREAKPOINT 4: For Final Output snapshots (41-43)
        # FINAL OUTPUT
        # SNAPSHOT 41: Decoder final sequence output (before projection)
        snapshot_41 = full_output['decoder_output']

        # SNAPSHOT 42: Logits after final linear projection
        snapshot_42 = full_output['logits']

        # SNAPSHOT 43: Logits slice (first few values for one token)
        snapshot_43 = full_output['logits'][0, 0, :10]  # First batch, first token, first 10 values

        # ==================================================

        all_snapshots = {
            # Input & Embedding (Snapshots 1-5)
            1: {'name': 'Raw input tokens', 'shape': snapshot_1.shape if snapshot_1 is not None else None, 'expected_shape': (src_seq_length, batch_size)},
            2: {'name': 'Target tokens', 'shape': snapshot_2.shape if snapshot_2 is not None else None, 'expected_shape': (tgt_seq_length, batch_size)},
            3: {'name': 'Embedding weight matrix slice', 'shape': snapshot_3.shape if snapshot_3 is not None else None, 'expected_shape': (5, 5)},
            4: {'name': 'Input embeddings after lookup', 'shape': snapshot_4.shape if snapshot_4 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            5: {'name': 'Embeddings after positional encoding', 'shape': snapshot_5.shape if snapshot_5 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},

            # Encoder (Snapshots 6-19)
            6: {'name': 'Encoder block input tensor', 'shape': snapshot_6.shape if snapshot_6 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            7: {'name': 'Self-attention queries (Q)', 'shape': snapshot_7.shape if snapshot_7 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            8: {'name': 'Self-attention keys (K)', 'shape': snapshot_8.shape if snapshot_8 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            9: {'name': 'Self-attention values (V)', 'shape': snapshot_9.shape if snapshot_9 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            10: {'name': 'Attention scores before softmax', 'shape': snapshot_10.shape if snapshot_10 is not None else None, 'expected_shape': (batch_size, num_heads, src_seq_length, src_seq_length)},
            11: {'name': 'Attention scores after softmax', 'shape': snapshot_11.shape if snapshot_11 is not None else None, 'expected_shape': (batch_size, num_heads, src_seq_length, src_seq_length)},
            12: {'name': 'Multi-head split (Q)', 'shape': snapshot_12.shape if snapshot_12 is not None else None, 'expected_shape': (num_heads, batch_size, src_seq_length, d_model // num_heads)},
            13: {'name': 'Multi-head attention output after concatenation', 'shape': snapshot_13.shape if snapshot_13 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            14: {'name': 'Residual connection after attention', 'shape': snapshot_14.shape if snapshot_14 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            15: {'name': 'Layer normalization output', 'shape': snapshot_15.shape if snapshot_15 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            16: {'name': 'Feed-forward input', 'shape': snapshot_16.shape if snapshot_16 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            17: {'name': 'Feed-forward first layer output', 'shape': snapshot_17.shape if snapshot_17 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_ff)},
            18: {'name': 'Feed-forward second layer output', 'shape': snapshot_18.shape if snapshot_18 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            19: {'name': 'Encoder block final output', 'shape': snapshot_19.shape if snapshot_19 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},

            # Decoder (Snapshots 20-40)
            20: {'name': 'Decoder block input tensor', 'shape': snapshot_20.shape if snapshot_20 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            21: {'name': 'Masked self-attention queries (Q)', 'shape': snapshot_21.shape if snapshot_21 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            22: {'name': 'Masked self-attention keys (K)', 'shape': snapshot_22.shape if snapshot_22 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            23: {'name': 'Masked self-attention values (V)', 'shape': snapshot_23.shape if snapshot_23 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            24: {'name': 'Masked attention scores before mask', 'shape': snapshot_24.shape if snapshot_24 is not None else None, 'expected_shape': (batch_size, num_heads, tgt_seq_length, tgt_seq_length)},
            25: {'name': 'Mask tensor', 'shape': snapshot_25.shape if snapshot_25 is not None else None, 'expected_shape': (tgt_seq_length, tgt_seq_length)},
            26: {'name': 'Masked attention scores after softmax', 'shape': snapshot_26.shape if snapshot_26 is not None else None, 'expected_shape': (batch_size, num_heads, tgt_seq_length, tgt_seq_length)},
            27: {'name': 'Masked self-attention multi-head split', 'shape': snapshot_27.shape if snapshot_27 is not None else None, 'expected_shape': (num_heads, batch_size, tgt_seq_length, d_model // num_heads)},
            28: {'name': 'Masked self-attention multi-head concatenated output', 'shape': snapshot_28.shape if snapshot_28 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            29: {'name': 'Residual + norm after masked attention', 'shape': snapshot_29.shape if snapshot_29 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            30: {'name': 'Cross-attention queries (from decoder)', 'shape': snapshot_30.shape if snapshot_30 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            31: {'name': 'Cross-attention keys (from encoder)', 'shape': snapshot_31.shape if snapshot_31 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            32: {'name': 'Cross-attention values (from encoder)', 'shape': snapshot_32.shape if snapshot_32 is not None else None, 'expected_shape': (src_seq_length, batch_size, d_model)},
            33: {'name': 'Cross-attention scores before softmax', 'shape': snapshot_33.shape if snapshot_33 is not None else None, 'expected_shape': (batch_size, num_heads, tgt_seq_length, src_seq_length)},
            34: {'name': 'Cross-attention scores after softmax', 'shape': snapshot_34.shape if snapshot_34 is not None else None, 'expected_shape': (batch_size, num_heads, tgt_seq_length, src_seq_length)},
            35: {'name': 'Cross-attention output after concatenation', 'shape': snapshot_35.shape if snapshot_35 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            36: {'name': 'Residual + norm after cross-attention', 'shape': snapshot_36.shape if snapshot_36 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            37: {'name': 'Decoder feed-forward input', 'shape': snapshot_37.shape if snapshot_37 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            38: {'name': 'Feed-forward first layer output', 'shape': snapshot_38.shape if snapshot_38 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_ff)},
            39: {'name': 'Feed-forward second layer output', 'shape': snapshot_39.shape if snapshot_39 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            40: {'name': 'Decoder block final output', 'shape': snapshot_40.shape if snapshot_40 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},

            # Final Output (Snapshots 41-43)
            41: {'name': 'Decoder final output before projection', 'shape': snapshot_41.shape if snapshot_41 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, d_model)},
            42: {'name': 'Logits after final projection', 'shape': snapshot_42.shape if snapshot_42 is not None else None, 'expected_shape': (tgt_seq_length, batch_size, vocab_size)},
            43: {'name': 'Logits slice (first token)', 'shape': snapshot_43.shape if snapshot_43 is not None else None, 'expected_shape': (10,)},
        }

        print(f"\nFinal output logits shape: {full_output['logits'].shape}")
        print("All 43 snapshots have been captured and are available for inspection.")

        # Print shapes for verification
        print("\n=== Snapshot Shapes Verification ===")
        captured_count = 0
        for snapshot_num, snapshot_info in all_snapshots.items():
            actual_shape = snapshot_info['shape']
            expected_shape = snapshot_info['expected_shape']
            if actual_shape is not None:
                status = "✓" if actual_shape == expected_shape else "✗"
                print(f"Snapshot #{snapshot_num:2d} {status}: {snapshot_info['name']}")
                print(f"     Expected: {expected_shape}")
                print(f"     Actual:   {actual_shape}")
                captured_count += 1
            else:
                print(f"Snapshot #{snapshot_num:2d} ✗: {snapshot_info['name']} - NOT CAPTURED")

        print(f"\nCaptured {captured_count}/43 snapshots successfully!")

        snapshot_variables = {}
        for i in range(1, 44):
            var_name = f"snapshot_{i}"
            snapshot_variables[var_name] = locals().get(var_name)

        return full_output, all_snapshots, snapshot_variables


def inspect_model_parameters(model):
    """Inspect model parameters for debugging"""
    print("\n=== Model Parameter Inspection ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel configuration:")
    print(f"d_model: {model.encoder.d_model}")
    print(f"Number of encoder layers: {len(model.encoder.layers)}")
    print(f"Number of decoder layers: {len(model.decoder.layers)}")
    print(f"Number of attention heads: {model.encoder.layers[0].self_attention.num_heads}")


if __name__ == "__main__":
    print("Transformer Debugging Script")
    print("=" * 50)
    print("Model configured with:")
    print("- 2 encoder layers, 2 decoder layers")
    print("- 4 attention heads")
    print("- 128 embedding dimension")
    print("- 5-12 token sequences")
    print("=" * 50)

    vocab_size = 1000
    model = DebuggableTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    )

    inspect_model_parameters(model)

    try:
        final_output, all_snapshots, snapshot_variables = debug_transformer_flow()

        print("\nDebugging completed successfully!")
        print("All 43 snapshots are available in the 'all_snapshots' dictionary.")
        print("Use PyCharm's variable inspector to examine tensor dimensions at each layer.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("This might be due to PyCharm debugger limitations with large tensors.")
        print("Try running without debug mode or reduce the tensor sizes for debugging.")