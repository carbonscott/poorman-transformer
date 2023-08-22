import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, embd_size, head_size, dropout = 0.1):
        super().__init__()

        self.embd_size = embd_size
        self.head_size = head_size
        self.dropout   = dropout

        # Self-attention layer to update each node by aggregating features from all other nodes...
        self.proj_q = nn.Linear(self.embd_size, self.head_size)    # What do I (this node) want?
        self.proj_k = nn.Linear(self.embd_size, self.head_size)    # What do I have publicly?
        self.proj_v = nn.Linear(self.embd_size, self.head_size)    # What do I provide to update the entire graph?

        # Store a mask to prevent it from gradient tracking...
        self.mask_initialized = False

        # Use dropout after the scaled dot self-attention...
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        B: Batch, T: Context length, E: Embedding length, H: Head Embedding length
        Arguments:
            x : (B, T, E)

        Return:
            a : (B, T, H)
                Nodes (embd per token) updated ('Jiggled') through weighted sum (aggregation).
        """
        # Linearly project them to a vector space...
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Scaled dot product...
        w = q @ k.permute(0,2,1)    # Q @ K.T
        w /= torch.sqrt(torch.tensor(self.head_size))

        # Masking in the decoder...
        # Store a mask to prevent it from gradient tracking
        if not self.mask_initialized:
            mask = torch.ones_like(w).triu(diagonal=1).bool()
            self.register_buffer('mask', mask)
            self.mask_initialized = True
        w[mask] = float('-inf')

        # Obtain the softmax...
        w = w.softmax(dim = -1)

        # Aggregate information from all nodes...
        a = w @ v

        return a




class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size, head_size, dropout = 0.1):
        super().__init__()

        num_heads = embd_size // head_size
        self.multi_head_att_layer = nn.ModuleList([ SingleHeadAttention(embd_size, head_size) for _ in range(num_heads) ])

        self.proj_linear = nn.Linear(embd_size, embd_size)

        # Use dropout at the end...
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        y = [ layer(x) for layer in self.multi_head_att_layer ]
        y = torch.cat(y, dim = -1)    # ...concatenate the head embedding dimension

        y = self.proj_linear(y)

        y = self.dropout(y)

        return y




class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout = 0.1):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(    embd_size, 4 * embd_size),
            nn.GELU(),
            nn.Linear(4 * embd_size,     embd_size),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.ff_layer(x)




class TransformerBlock(nn.Module):
    def __init__(self, embd_size, num_heads):
        super().__init__()

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.multi_head_att_layer = MultiHeadAttention(embd_size, head_size)

        # Define the feedforward layer to add non-linearity to the model...
        self.ff_layer = FeedForward(embd_size)

        # Define layers to optimize model training...
        self.layer_norm_pre_multi_head  = nn.LayerNorm(embd_size)
        self.layer_norm_pre_feedforward = nn.LayerNorm(embd_size)


    def forward(self, x):
        """
        Arguments:
            x : (B, T, E)

        Returns:
            out : (B, T, E)
        """
        nodes_embd = x

        # ___/ MULTI-HEAD ATTENTION BLOCK \___
        # Go through multi-head attention to update nodes in vector space...
        # ...Pre norm
        nodes_embd_norm = self.layer_norm_pre_multi_head(nodes_embd)

        # ...Attention
        nodes_embd_update = self.multi_head_att_layer(nodes_embd_norm)    # (B, T, E)

        # ...Residual connection (out -> prenorm)
        nodes_embd_update += nodes_embd

        # Learn a better embedding representation by introducing non-linearity...
        # ...Pre norm
        nodes_embd_update_norm = self.layer_norm_pre_feedforward(nodes_embd_update)    # (B, T, E)

        # ...Feed forward
        nodes_embd_better = self.ff_layer(nodes_embd_update_norm)    # (B, T, E)

        # ...Residual connection (out -> prenorm)
        nodes_embd_better += nodes_embd_update

        return nodes_embd_better




class Transformer(nn.Module):
    def __init__(self, token_lib_size, embd_size, num_blocks, num_heads):
        super().__init__()

        # Define the embedding layer to embed each node to a vector space...
        self.embd_layer = nn.Embedding(token_lib_size, embd_size)

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.transformer_block = nn.Sequential(*tuple(
            TransformerBlock(embd_size, num_heads) for _ in range(num_blocks)
        ))

        # Prediction head...
        self.pred_head = nn.Linear(embd_size, token_lib_size)


    def forward(self, x):
        """
        Arguments:
            x : (B, T, N)
        """
        # ___/ EMBED ALL NODES \___
        nodes_embd = self.embd_layer(x)    # (B, T, N) -> (B, T, E)

        # ___/ MULTI-HEAD ATTENTION BLOCK \___
        # Go through multi-head attention to update nodes in vector space...
        nodes_embd_better = self.transformer_block(nodes_embd)    # (B, T, E) -> (B, T, E)

        # ___/ PREDICTION HEAD \___
        out = self.pred_head(nodes_embd_better)

        return out
