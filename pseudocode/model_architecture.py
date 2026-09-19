"""
Pseudocode: IresFM Model Architecture (Swin Transformer-based Power Forecasting)

This file documents the neural network architecture for power forecasting.
The model uses a Swin Transformer backbone with cross-attention and gating mechanisms.

Mathematical Notation
--------------------
B: batch size
T: temporal dimension
H, W: spatial dimensions (height, width)
C: channel/embedding dimension
D: MLP expansion ratio
"""

# ===================================================================
# 1. Patch Embedding Layer
# ===================================================================

class PatchEmbedding:
    """
    Converts raw inputs into patch embeddings.

    Purpose:
        - Embed power history (1D temporal sequence)
        - Embed weather history (2D spatial-temporal grids)
        - Embed weather forecast (2D spatial-temporal grids)

    Key Design:
        - Power: 1D convolution preserves temporal resolution (15-min)
        - Weather past: 1D temporal conv (hourly patches)
        - Weather future: 4-hour temporal stride to reduce computation
        - Spatial: Conv reduces 64 spatial points → spatial_embedding_dim
    """

    def __init__(self, model_args):
        """
        Parameters:
            embedding_dim: C (total embedding dimension)
            input_hours: Tp (past hours, 120 in the base-size FM used in the paper)
            output_hours: Tf (future hours, e.g., 120)
            spatial_embedding_dim: 16 spatial tokens in the base-size FM used in the paper

        Layers:
            conv_power_past: Conv1d(in=2, out=C, kernel=1, stride=1)
            conv_weather_past: Conv2d(in=27, out=C, kernel=(1, s_stride), stride=(1, s_stride))
            conv_weather_future: Conv2d(in=27, out=C, kernel=(4, s_stride), stride=(4, s_stride))

        where s_stride = 64 // spatial_embedding_dim
        """
        pass

    def forward(self, input_weather_past, input_weather_future, input_power_past):
        """
        Inputs:
            input_weather_past: [B, Tp=120, 64, 27]
                - Tp hours of past weather
                - 64 spatial points (8×8 flattened)
                - 27 channels (11 HRES + 8 Pangu + 8 FengWu)

            input_weather_future: [B, Tf=120, 64, 27]
                - Tf hours of future weather forecast
                - Same spatial and channel structure

            input_power_past: [B, 4*Tp=480, 2]
                - 15-minute resolution power history
                - 2 channels: (power, task_label)
                - task_label encodes wind/solar mode

        Outputs:
            patch_weather_future: [B, Tf/4=30, so=16, C]
                - 4-hour temporal stride reduces computation
                - Spatial dimension reduced to so

            patch_weather_past: [B, Tp=120, so=16, C]
                - Hourly patches

            patch_power_past: [B, 4*Tp=480, C]
                - 15-minute resolution preserved
                - Temporal encoding is added later to the projected hourly context

        Implementation:
            # Spatial convolution and temporal alignment
            # Weather future: permute to [B, 27, 120, 64]
            # Apply conv_weather_future → [B, C, 30, 16]
            # Permute back to [B, 30, 16, C]

            # Weather past: permute to [B, 27, 120, 64]
            # Apply conv_weather_past → [B, C, 120, 16]
            # Permute back to [B, 120, 16, C]

            # Power: permute to [B, 2, 480]
            # Apply conv_power_past → [B, C, 480]
            # Permute back to [B, 480, C]

        """
        pass


# ===================================================================
# 2. Swin Transformer Components
# ===================================================================

class Attention2D:
    """
    2D Window-based Multi-Head Self-Attention with relative position bias.

    Mathematical Formula:
        Attention(Q, K, V) = softmax((Q·K^T)/√d + B) · V

        where:
        - Q, K, V: query, key, value matrices
        - d: head dimension (C // num_heads)
        - B: relative position bias (learned)

    Window Mechanism:
        - Input is partitioned into non-overlapping windows
        - Attention computed within each window independently
        - Window size: (5, 4) for temporal × spatial
    """

    def __init__(self, dim, heads, window_size, input_resolution, dropout_rate):
        """
        Parameters:
            dim: C (embedding dimension)
            heads: number of attention heads
            window_size: (Wh, Ww) = (5, 4)
            input_resolution: (H, W) input spatial-temporal shape
            dropout_rate: dropout probability

        Layers:
            linear_qkv: Linear(C, 3*C) - projects to Q, K, V
            linear: Linear(C, C) - output projection
            relative_position_bias_table: Parameter[(2*Wh-1)*(2*Ww-1), heads]

        Relative Position Bias:
            - Learned bias for each relative position within window
            - Indexed by relative_position_index (precomputed)
        """
        pass

    def forward(self, x, mask):
        """
        Inputs:
            x: [B*nW, T, C]
                - B*nW: batch × number of windows
                - T: window_size[0] * window_size[1] = 20
                - C: embedding dimension

            mask: [1, nW, 1, T, T]
                - Attention mask for shifted windows
                - -1000 for masked positions, 0 otherwise

        Outputs:
            x: [B*nW, T, C]
                - Attention-weighted features

        Implementation:
            # 1. Project to Q, K, V
            # qkv = linear_qkv(x)  # [B*nW, T, 3*C]
            # qkv = reshape to [B*nW, T, 3, heads, C//heads]
            # qkv = permute to [3, B*nW, heads, T, C//heads]
            # Q, K, V = qkv[0], qkv[1], qkv[2]

            # 2. Compute attention scores
            # attn = (Q @ K^T) * scale  # [B*nW, heads, T, T]
            # scale = (C // heads) ** -0.5

            # 3. Add relative position bias
            # bias = relative_position_bias_table[relative_position_index]
            # bias = reshape to [T, T, heads] → permute to [heads, T, T]
            # attn = attn + bias

            # 4. Add window mask (for shifted windows)
            # attn = attn + mask

            # 5. Softmax and apply to values
            # attn = softmax(attn, dim=-1)
            # x = attn @ V  # [B*nW, heads, T, C//heads]

            # 6. Reshape and project
            # x = permute to [B*nW, T, heads, C//heads]
            # x = reshape to [B*nW, T, C]
            # x = linear(x)
        """
        pass


class MLP:
    """
    Feed-Forward Network with GELU activation.

    Formula:
        MLP(x) = Linear2(GELU(Linear1(x)))

        where:
        - Linear1: C → C*D (expansion)
        - Linear2: C*D → C (projection)
        - D: expansion ratio (typically 4)
    """

    def __init__(self, dim, D, dropout_rate):
        """
        Parameters:
            dim: C (embedding dimension)
            D: expansion ratio
            dropout_rate: dropout probability

        Layers:
            linear1: Linear(C, C*D)
            linear2: Linear(C*D, C)
        """
        pass

    def forward(self, x):
        """
        Inputs:
            x: [B, H, W, C]

        Outputs:
            x: [B, H, W, C]

        Implementation:
            # x = linear1(x)  # [B, H, W, C*D]
            # x = GELU(x)
            # x = dropout(x)
            # x = linear2(x)  # [B, H, W, C]
            # x = dropout(x)
        """
        pass


class SwinTransformerBlock:
    """
    Swin Transformer Block with window-based attention and shifted windows.

    Architecture:
        x' = x + DropPath(LayerNorm(WindowAttention(x)))
        x'' = x' + DropPath(LayerNorm(MLP(x')))

    Shifted Window Mechanism:
        - Even blocks (i % 2 == 0): regular windows
        - Odd blocks (i % 2 == 1): shifted windows
        - Shift size: (window_size[0]//2, window_size[1]//2)
    """

    def __init__(self, dim, input_resolution, heads, drop_path_ratio, D):
        """
        Parameters:
            dim: C
            input_resolution: (H, W)
            heads: number of attention heads
            drop_path_ratio: stochastic depth probability
            D: MLP expansion ratio

        Components:
            attention: Attention2D
            norm1, norm2: LayerNorm(C)
            feedforward: MLP
            drop_path: DropPath (stochastic depth)
            mask: attention mask for shifted windows
        """
        pass

    def forward(self, x, roll):
        """
        Inputs:
            x: [B, H, W, C]
            roll: bool - whether to apply window shifting

        Outputs:
            x: [B, H, W, C]

        Implementation:
            # 1. Window partitioning
            # if roll:
            #     x = torch.roll(x, shifts=(Wh//2, Ww//2), dims=(1, 2))

            # 2. Partition into windows
            # nWh, nWw = H // Wh, W // Ww
            # x = reshape to [B, nWh, Wh, nWw, Ww, C]
            # x = permute to [B, nWh, nWw, Wh, Ww, C]
            # x = reshape to [B*nWh*nWw, Wh*Ww, C]

            # 3. Window attention
            # x = attention(x, mask if roll else zero_mask)

            # 4. Reverse window partitioning
            # x = reshape to [B, nWh, nWw, Wh, Ww, C]
            # x = permute to [B, nWh, Wh, nWw, Ww, C]
            # x = reshape to [B, H, W, C]

            # 5. Reverse cyclic shift
            # if roll:
            #     x = torch.roll(x, shifts=(-Wh//2, -Ww//2), dims=(1, 2))

            # 6. Residual connections
            # x = skip + drop_path(norm1(x))
            # x = x + drop_path(norm2(feedforward(x)))
        """
        pass


class SwinTransformerLayer:
    """
    Stack of Swin Transformer Blocks with alternating window patterns.
    """

    def __init__(self, depth, dim, input_resolution, heads, drop_path_ratio_list, D):
        """
        Parameters:
            depth: number of blocks in this layer
            dim: C
            input_resolution: (H, W)
            heads: number of attention heads
            drop_path_ratio_list: list of drop path ratios for each block
            D: MLP expansion ratio
        """
        pass

    def forward(self, x):
        """
        Inputs:
            x: [B, H, W, C]

        Outputs:
            x: [B, H, W, C]

        Implementation:
            # for i in range(depth):
            #     if i % 2 == 0:
            #         x = blocks[i](x, roll=False)  # regular windows
            #     else:
            #         x = blocks[i](x, roll=True)   # shifted windows
        """
        pass


# ===================================================================
# 3. Spatial Resolution Adjustment
# ===================================================================

class DownSample:
    """
    Reduces spatial resolution by 2× in both dimensions.
    Increases channel dimension by 2×.

    Operation:
        [B, H, W, C] → [B, H/2, W/2, 2C]

    Method:
        - Reshape to merge 2×2 spatial patches
        - Apply LayerNorm and Linear projection
    """

    def __init__(self, dim):
        """
        Parameters:
            dim: C (input channels)

        Layers:
            norm: LayerNorm(4*C)
            linear: Linear(4*C, 2*C)
        """
        pass

    def forward(self, x):
        """
        Inputs:
            x: [B, H, W, C]

        Outputs:
            x: [B, H/2, W/2, 2C]

        Implementation:
            # Reshape [B, H, W, C] → [B, H/2, 2, W/2, 2, C]
            # Permute to [B, H/2, W/2, 2, 2, C]
            # Reshape to [B, H/2, W/2, 4C]
            # x = norm(x)
            # x = linear(x)  # [B, H/2, W/2, 2C]
        """
        pass


class UpSample:
    """
    Increases spatial resolution by 2× in both dimensions.
    Decreases channel dimension by 2×.

    Operation:
        [B, H, W, 2C] → [B, 2H, 2W, C]
    """

    def __init__(self, input_dim, output_dim):
        """
        Parameters:
            input_dim: 2C
            output_dim: C

        Layers:
            linear1: Linear(2C, 4C)
            linear2: Linear(C, C)
            norm: LayerNorm(C)
        """
        pass

    def forward(self, x):
        """
        Inputs:
            x: [B, H, W, 2C]

        Outputs:
            x: [B, 2H, 2W, C]

        Implementation:
            # x = linear1(x)  # [B, H, W, 4C]
            # Reshape to [B, H, W, 2, 2, C]
            # Permute to [B, H, 2, W, 2, C]
            # Reshape to [B, 2H, 2W, C]
            # x = norm(x)
            # x = linear2(x)
        """
        pass


# ===================================================================
# 4. Output Head
# ===================================================================

class PatchRecovery:
    """
    Converts patch embeddings back to power predictions.

    Operation:
        [B, H, W, 2C] → [B, 480]

        where 480 = 120 hours × 4 (15-minute intervals)
    """

    def __init__(self, model_args, dropout_rate=0.3):
        """
        Parameters:
            embedding_dim: C
            output_hours: Tf (e.g., 120)
            spatial_embedding_dim: so

        Layers:
            conv_air: Conv2d(2C, C, kernel=(5,4), stride=(5,4))
            linear: Linear(C*H'*W', 480)

        where H', W' are dimensions after conv
        """
        pass

    def forward(self, x):
        """
        Inputs:
            x: [B, H, W, 2C]

        Outputs:
            power: [B, 480]

        Implementation:
            # x = permute to [B, 2C, H, W]
            # x = conv_air(x)  # [B, C, H', W']
            # x = GELU(x)
            # x = flatten to [B, C*H'*W']
            # x = dropout(x)
            # x = linear(x)  # [B, 480]
        """
        pass


# ===================================================================
# 5. Autoregressive Power Model (Path 2)
# ===================================================================

class PowerARModel:
    """
    GRU temporal expert for power prediction.
    Uses the fused historical weather and power sequence, including temporal
    positional encoding, shared with the cross-attention memory and gate.

    Architecture:
        Fused past sequence → Bidirectional GRU → MLP → Output
    """

    def __init__(self, C, ar_hidden_dim=64, out_steps=480):
        """
        Parameters:
            C: input embedding dimension (384 in the base-size FM used in the paper)
            ar_hidden_dim: H (set to C/2 = 192 by the base-size FM used in the paper)
            out_steps: 480 (output length)

        Layers:
            gru: GRU(C, H, num_layers=2, bidirectional=True, batch_first=True)
            fc: MLP(4H → 8H → 480)

        Final hidden-state dimension: 2 directions × 2 layers × H = 4H
        """
        pass

    def forward(self, x_p):
        """
        Inputs:
            x_p: [B, Tp=120, C=384]
                - 120 hourly tokens of fused historical weather and power
                - Weather features: 360 channels; power features: 24 channels
                - Temporal positional encoding is added after concatenation

        Outputs:
            output_power: [B, 480]

        Implementation:
            # _, h_n = gru(x_p)    # h_n: [4, B, H]
            # h_n = permute to [B, 4, H]
            # h_n = flatten to [B, 4H]
            # output = fc(h_n)     # [B, 480]
        """
        pass


# ===================================================================
# 6. Gating Network (Path 3)
# ===================================================================

class GatingHead:
    """
    Learns adaptive weights to combine weather-based and AR-based predictions.

    Formula:
        gate = sigmoid(MLP(past_features))
        output = gate * AR_output + (1 - gate) * weather_output
    """

    def __init__(self, input_seq_len, C, hidden_dim=64, out_steps=480):
        """
        Parameters:
            input_seq_len: Tp (120 in the base-size FM used in the paper)
            C: embedding dimension
            hidden_dim: intermediate dimension
            out_steps: 480

        Layers:
            fc1: Linear(Tp*C, hidden_dim)
            fc2: Linear(hidden_dim, 480)
        """
        pass

    def forward(self, x_p):
        """
        Inputs:
            x_p: [B, Tp, C]
                - Fused past weather and power features, with temporal encoding
                - The same sequence used as cross-attention memory and GRU input

        Outputs:
            gate: [B, 480]
                - Sigmoid-activated gating weights ∈ [0, 1]

        Implementation:
            # x = flatten(x_p)  # [B, Tp*C]
            # x = GELU(fc1(x))  # [B, hidden_dim]
            # gate = sigmoid(fc2(x))  # [B, 480]
        """
        pass


# ===================================================================
# 7. Main Model: PowerModel
# ===================================================================

class PowerModel:
    """
    Complete IresFM architecture with two prediction pathways and adaptive
    fusion:

    Pathway 1: Swin Transformer on future weather
    Pathway 2: GRU temporal expert on fused historical weather and power
    Adaptive gate: element-wise fusion of the two pathway outputs

    Architecture Flow:
        1. Patch Embedding
        2. Cross-Attention (optional, fuses past weather+power with future weather)
        3. U-Net style Swin Transformer:
           - Layer1: [B, 30, 16, C]
           - DownSample → [B, 15, 8, 2C]
           - Layer2: [B, 15, 8, 2C]
           - Layer3: [B, 15, 8, 2C]
           - UpSample → [B, 30, 16, C]
           - Layer4: [B, 30, 16, C]
           - Skip connection: concat Layer1 output
        4. Patch Recovery → [B, 480]
        5. (Optional) AR path and gating fusion
    """

    def __init__(self, model_args):
        """
        Parameters:
            embedding_dim: C (384 in the base-size FM)
            depths_list: [2, 4, 4, 2] (number of blocks per layer)
            num_heads_list: [6, 8, 8, 6] (attention heads per layer)
            mlp_up_ratio: D (e.g., 4)
            w_cross_attn: whether to use cross-attention
            w_ts_expert: whether to use AR path and gating
            w_sptm_emb: whether to use temporal positional encoding

        Components:
            input_layer: PatchEmbedding
            power_weather_ca: TransformerDecoder (cross-attention)
            layer1, layer2, layer3, layer4: SwinTransformerLayer
            downsample, upsample: DownSample, UpSample
            output_layer: PatchRecovery
            ar_model: PowerARModel (optional)
            gating_head: GatingHead (optional)

        Loss Function:
            MSE(predicted_power, target_power)
        """
        pass

    def forward(self, input_weather_past, input_weather_future, input_power_past, target_power=None):
        """
        Inputs:
            input_weather_past: [B, 120, 64, 27]
            input_weather_future: [B, 120, 64, 27]
            input_power_past: [B, 480, 2]
            target_power: [B, 480] (optional, for training)

        Outputs:
            If target_power is provided:
                {"loss": scalar, "output": [B, 480]}
            Else:
                [B, 480]

        Implementation:
            # 1. Patch Embedding
            # patch_weather_future, patch_weather_past, patch_power_past = input_layer(...)

            # 2. Cross-Attention (if w_cross_attn)
            # Flatten spatial dimension for cross-attention
            # patch_weather_future_ca = proj_weather_future_ca(flatten(patch_weather_future))
            # patch_weather_past_ca = proj_weather_past_ca(flatten(patch_weather_past))  # [B, 120, 360]
            # patch_power_past_ca = proj_power_past_ca(patch_power_past)  # [B, 480, 6]
            # Group four consecutive 15-min embeddings within each hourly token:
            # patch_power_past_ca = patch_power_past_ca.reshape(B, 120, 24)
            # patch_seq_past = concat([patch_weather_past_ca, patch_power_past_ca], dim=-1)  # [B, 120, 384]
            # Add positional encoding (if w_sptm_emb)
            # x_ca = TransformerDecoder(patch_weather_future_ca, patch_seq_past)
            # Add residual: patch_weather_future = patch_weather_future + x_ca

            # 3. Swin Transformer U-Net
            # x = layer1(patch_weather_future)  # [B, 30, 16, C]
            # skip = x
            # x = downsample(x)                 # [B, 15, 8, 2C]
            # x = layer2(x)                     # [B, 15, 8, 2C]
            # x = layer3(x)                     # [B, 15, 8, 2C]
            # x = upsample(x)                   # [B, 30, 16, C]
            # x = layer4(x)                     # [B, 30, 16, C]
            # x = concat([skip, x], dim=-1)     # [B, 30, 16, 2C]

            # 4. Output Head
            # output_weather = output_layer(x)  # [B, 480]

            # 5. AR Path and Gating (if w_ts_expert)
            # output_ar = ar_model(patch_seq_past)           # [B, 480]
            # gate = gating_head(patch_seq_past)             # [B, 480]
            # final_output = gate * output_ar + (1 - gate) * output_weather

            # 6. Loss (if training)
            # if target_power is not None:
            #     loss = MSE(final_output, target_power)
            #     return {"loss": loss, "output": final_output}
            # else:
            #     return final_output
        """
        pass


# ===================================================================
# 8. Model Size Configurations
# ===================================================================

"""
Model Size Configurations:

Small:
    embedding_dim: 192
    weather_embedding_dim / power_embedding_dim: 180 / 12
    depths_list: [2, 4, 4, 2]
    num_heads_list: [4, 6, 6, 4]
    cross_attention_layers / heads: 8 / 8
    Parameters: 35.75M

Medium:
    embedding_dim: 276
    weather_embedding_dim / power_embedding_dim: 256 / 20
    depths_list: [2, 4, 4, 2]
    num_heads_list: [4, 6, 6, 4]
    cross_attention_layers / heads: 12 / 12
    Parameters: 75.07M

Base:
    embedding_dim: 384
    weather_embedding_dim / power_embedding_dim: 360 / 24
    depths_list: [2, 4, 4, 2]
    num_heads_list: [6, 8, 8, 6]
    cross_attention_layers / heads: 12 / 12
    Parameters: 135.68M

Large:
    embedding_dim: 432
    weather_embedding_dim / power_embedding_dim: 400 / 32
    depths_list: [4, 6, 6, 4]
    num_heads_list: [8, 12, 12, 8]
    cross_attention_layers / heads: 16 / 16
    Parameters: 226.16M
"""


# ===================================================================
# 9. Key Hyperparameters
# ===================================================================

"""
Training Hyperparameters:

Optimizer: AdamW
    Formula: θ_{t+1} = θ_t - η * (m_t / (√v_t + ε) + λ * θ_t)
    where:
        m_t: first moment estimate (momentum)
        v_t: second moment estimate (adaptive learning rate)
        η: learning rate
        λ: weight decay

    Pretraining:
        learning_rate: 1e-5
        weight_decay: 0.0
        batch_size: 128
        epochs: 1

    Site-specific fine-tuning:
        learning_rate: 5e-5
        weight_decay: 0.0
        batch_size: 32
        epochs: 1

Learning Rate Schedule: Cosine Annealing with Warmup
    Formula:
        η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T))

    Warmup phase (first 5-10% of training):
        η_t = η_max * (t / T_warmup)

    Warmup ratio: 0.1

Gradient Clipping:
    max_grad_norm: 1.0
    Prevents exploding gradients

The base-size FM used in the paper is pretrained for one epoch. Site-specific adaptation
also uses one epoch with the backbone frozen and the forecast head trainable.
"""
