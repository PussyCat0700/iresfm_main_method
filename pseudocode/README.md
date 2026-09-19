# Pseudocode Documentation

This directory contains pseudocode documentation for the IresFM power
forecasting system. The pseudocode is designed to be readable and
self-contained, not executable.

## Purpose

The pseudo-code serves multiple purposes:

1. **Documentation**: Clearly explains the model architecture, training process, and inference workflow
2. **Reference**: Provides mathematical formulas, tensor shapes, and design decisions
3. **Adaptation Guide**: Helps researchers understand and adapt the approach for their own use cases
4. **Reproducibility**: Documents key hyperparameters and implementation details

**Important**: This is pseudo-code, not runnable code. It uses simplified syntax and focuses on concepts rather than implementation details.

## File Structure

### Core Workflow Files

1. **`data_generation.py`** - Synthetic data generation
   - Physical power models (wind turbine, PV system)
   - Weather data assembly (HRES + Pangu + FengWu)
   - Temporal interpolation and spatial patch selection
   - See this file for understanding the open-source data format

2. **`model_architecture.py`** - Neural network architecture
   - Swin Transformer backbone
   - Cross-attention mechanism
   - Autoregressive power model
   - Gating network for multi-path fusion
   - Mathematical formulas and tensor shapes

3. **`pretraining.py`** - Pretraining workflow
   - Synthetic data loading
   - Model initialization
   - Training loop and optimization
   - Checkpoint management

4. **`finetuning.py`** - Fine-tuning workflow
   - Real data loading
   - Parameter-efficient fine-tuning (freeze backbone)
   - Station-specific adaptation
   - Transfer learning strategies

5. **`inference.py`** - Inference workflow
   - Model loading
   - Weather forecast processing
   - Sliding window inference
   - Evaluation metrics

6. **`data_loader.py`** - Data loading utilities
   - Dataset classes
   - Data preprocessing
   - Normalization and augmentation
   - Open-source data format

7. **[`experiments.md`](experiments.md)** - Experiment protocols
   - Three surrogate fidelity levels
   - Fig. 3a–f comparison axes, configurations and base settings
   - Repetition, optimization and evaluation definitions

## How to Read the Pseudo-Code

### Notation Conventions

**Tensor Shapes**:
- `[B, T, D]`: Batch size × Time steps × Feature dimension
- `[B, H, W, C]`: Batch size × Height × Width × Channels
- Example: `[16, 120, 64, 27]` = 16 samples, 120 hours, 64 spatial points, 27 channels

**Mathematical Formulas**:
- LaTeX-style notation in docstrings
- Example: `Attention(Q,K,V) = softmax(QK^T/√d)V`

**Code Structure**:
- Function signatures with type hints
- Detailed docstrings explaining purpose, inputs, outputs
- Implementation notes in comments (prefixed with `#`)
- `pass` statements indicate pseudo-code (not executable)

### Example: Reading a Pseudo-Code Function

```python
def ExampleFunction(input_tensor, hyperparameter):
    \"\"\"
    Brief description of what this function does.

    Mathematical formulation:
        output = σ(W·input + b)
        where σ is sigmoid activation

    Args:
        input_tensor: [B, T, D] - Batch of sequences
            B: batch size
            T: sequence length
            D: feature dimension
        hyperparameter: float - Controls some behavior

    Returns:
        output_tensor: [B, T, D'] - Transformed sequences
            D': output feature dimension

    Implementation notes:
        - Uses linear transformation followed by activation
        - Preserves temporal dimension
        - Can be parallelized across batch
    \"\"\"
    # Step 1: Linear transformation
    # W: [D, D'], b: [D']
    # output = input @ W + b  # [B, T, D']

    # Step 2: Activation
    # output = sigmoid(output)  # [B, T, D']

    # Step 3: Return
    # return output
    pass  # Pseudo-code, not executable
```

**What to focus on**:
1. **Docstring**: Understand the purpose and mathematical formulation
2. **Args/Returns**: Note the tensor shapes and their meanings
3. **Implementation notes**: Follow the high-level logic
4. **Comments**: See the operations and intermediate shapes

## Key Concepts

### 1. Data Format

**Power Data**:
- Resolution: 15-minute intervals
- Format: CSV with columns [time, power]
- Public CSV unit: installed-capacity-normalized power
- Model input: additional Z-score scaling from the adaptation period
- Predictions: transformed back to capacity-normalized power

**Weather Data**:
- Resolution: Hourly
- Format: NPY arrays with shape [hours, 64, 27]
- 64 spatial points: 8×8 patch around station (flattened)
- 27 channels: 11 HRES + 8 Pangu + 8 FengWu
- Normalization: Z-score per channel

**Task Label**:
- Wind: 0
- Solar: 1
- Appended as second channel to power input

### 2. Model Architecture

**Input**:
- Weather past: [B, 120, 64, 27] (5 days history)
- Weather future: [B, 120, 64, 27] (5 days forecast)
- Power past: [B, 480, 2] (5 days history, 15-min resolution)

**Output**:
- Power future: [B, 480] (5 days forecast, 15-min resolution)

**Architecture**:
1. **Patch Embedding**: Convert inputs to patch embeddings
2. **Cross-Attention** (optional): Fuse past context with future weather
3. **Swin Transformer**: U-Net style with 4 layers
4. **Patch Recovery**: Convert embeddings back to power predictions
5. **AR Path** (optional): GRU temporal expert using fused historical weather and power features
6. **Gating** (optional): Adaptive fusion of weather and AR paths

### 3. Training Process

**Pretraining**:
- Data: surrogate power from 10,000 globally sampled sites (5,000 wind and
  5,000 solar) over the calendar year 2021
- Duration: one epoch (approximately 2.1 million samples)
- Batch size: 128
- Learning rate: 1e-5 with linear warmup and cosine decay
- Objective: Learn general weather-to-power patterns

**Fine-Tuning**:
- Data: Real station-specific power
- Duration: one epoch per station
- Batch size: 32
- Learning rate: 5e-5 with linear warmup and cosine decay
- Strategy: train only `output_layer`, `ar_model.fc` and `gating_head.fc2`; freeze all remaining parameters
- Objective: Adapt to station-specific characteristics

### 4. Evaluation Metrics

- **RMSE**: Root Mean Square Error (capacity percentage points in the public package)
- **MAE**: Mean Absolute Error (capacity percentage points in the public package)
- **R²**: Coefficient of Determination
- **Corr**: Pearson correlation coefficient

## Adapting for Your Use Case

### 1. Different Data Format

If your data format differs:
- Modify `data_loader.py` to match your format
- Adjust temporal resolution (e.g., hourly instead of 15-min)
- Change spatial representation (e.g., single point instead of 8×8 patch)

### 2. Different Model Size

To use a different model size:
- See `model_architecture.py` for size configurations
- Adjust `embedding_dim`, `depths_list`, `num_heads_list`
- Trade-off: larger models = better performance but more computation

### 3. Different Forecast Horizon

To change forecast horizon:
- Adjust `ndays_before` and `ndays_after` parameters
- Modify input/output lengths accordingly
- May need to retrain from scratch

### 4. Single-Point Forecasting

If you don't have spatial weather data:
- Remove spatial dimension (64 → 1)
- Use 1D convolutions instead of 2D
- Simplify Swin Transformer to 1D attention

### 5. Different Weather Sources

To use different weather data:
- Modify channel count (27 → your number)
- Adjust `data_generation.py` to load your weather format
- Recompute normalization statistics

## Mathematical Background

### Attention Mechanism

```
Attention(Q, K, V) = softmax((Q·K^T)/√d) · V
```

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d: Dimension of keys (for scaling)

### Window-Based Attention

Swin Transformer partitions input into non-overlapping windows and computes attention within each window:
- Window size: (5, 4) for temporal × spatial
- Reduces complexity from O(N²) to O(N·W²) where W is window size
- Shifted windows enable cross-window connections

### Loss Function

```
Loss = MSE(predicted, target) = (1/N) * Σ(predicted - target)²
```

Where N = batch_size × output_length (e.g., 16 × 480)

### Optimizer (AdamW)

```
θ_{t+1} = θ_t - η * (m_t / (√v_t + ε) + λ * θ_t)
```

Where:
- θ: model parameters
- η: learning rate
- m_t: first moment (momentum)
- v_t: second moment (adaptive learning rate)
- λ: weight decay

## Common Questions

**Q: Can I run this code directly?**
A: No, this is pseudo-code for documentation purposes. You need to implement it in your framework of choice (PyTorch, TensorFlow, etc.).

**Q: What framework should I use?**
A: The pseudo-code is framework-agnostic. We recommend PyTorch for its flexibility and ease of use.

**Q: How do I get the actual model weights?**
A: The FM is available in ONNX format from the public release.
The pseudocode documents how an independent implementation can be trained.

**Q: Can I use this for other forecasting tasks?**
A: Yes! The architecture is general and can be adapted for other multivariate time series forecasting tasks.

**Q: What if I don't have synthetic data?**
A: You can skip pretraining and train directly on real data, but performance may be lower. Alternatively, use data augmentation techniques.

**Q: How do I choose hyperparameters?**
A: Start with the values in the pseudo-code, then tune based on validation performance. Key hyperparameters: learning rate, batch size, model size.

## References

For more details on the underlying techniques:

1. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
2. **Attention Mechanism**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
3. **Transfer Learning**: Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014
4. **Wind Power Modeling**: windpowerlib documentation
5. **Solar Power Modeling**: pvlib documentation

## Contact

For questions about the pseudo-code or the approach, please refer to the main README.md in the repository root.
