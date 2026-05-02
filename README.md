# Nora Optimizer (Normalized Orthogonal Row Alignment)

Nora is a memory-efficient, highly scalable optimizer designed specifically for training Large Language Models (LLMs) and deep neural networks. It applies **Orthogonal Row Alignment** to large weight matrices while seamlessly falling back to standard **Adam/AdamW** for 1D vectors (like biases and LayerNorms) and embedding layers.

## 🌟 Why use Nora instead of AdamW (and other matrix optimizers)?

* **The Holy Trinity: Efficiency, Speed, AND Stability:** While other matrix optimizers force trade-offs—Muon achieves efficient preconditioning but is computationally slow, and RMNP achieves speed ($\mathcal{O}(mn)$) but sacrifices stability—Nora uniquely satisfies all three core desiderata. It approximates Muon-like structured preconditioning for efficiency, uses row-wise normalization for extreme speed, and maintains strict scale-invariance for stability.
* **Eliminating Radial Jitters & Uncontrolled Row Growth:** In scale-invariant networks, radial updates (directions parallel to the weights) do not improve the model but instead inject noise and chaotically perturb the effective learning rate. Nora prevents this by explicitly projecting momentum onto the row-wise orthogonal complement of the weights. This row-wise orthogonality ensures the norm of each row grows uniformly and at a controlled, monotonic second-order rate, preventing any single row from uncontrolled norm explosion.
* **Lower VRAM Consumption:** For large matrices, Nora completely drops the heavy second-order momentum (`exp_avg_sq`) used by Adam. This drastically reduces the GPU memory footprint during training.
* **Provable Scalability:** Nora is theoretically proven to be a scalable optimizer under the Maximal Update Parametrization ($\mu$P) framework. It provides established learning rate scaling laws ($\eta_t \propto 1/\sqrt{n}$) that guarantee stable feature learning as model width approaches infinity.
* **Mixed Precision Ready:** Out-of-the-box compatibility with PyTorch's `fp16` and `bf16`. It automatically maintains `fp32` master weights for low-precision parameters to ensure stable updates.

<img width="1024" height="1536" alt="936323f19d90b00c6f1a2f67e55aa0a4" src="Nora.png" />

## ⚠️ How to Replace Adam (The Correct Way)

While Nora inherits from PyTorch's `torch.optim.Optimizer`, **do not** initialize it directly via `Nora(model.parameters())`. Doing so will incorrectly apply matrix-specific math to 1D vectors and embedding layers, which will stall your convergence.

Instead, use the included `get_nora_optimizer` factory function. It acts as an automatic router, scanning your model and applying Nora to 2D+ matrices and Adam to everything else.

### Standard Usage

```python
import torch
import torch.nn as nn
# Import the factory function instead of the raw class
from nora_xiugai import get_nora_optimizer

# 1. Initialize your model
model = MyTransformerModel()

# 2. Replace your AdamW with get_nora_optimizer
# Note: You MUST pass the 'model' itself, not 'model.parameters()'
optimizer = get_nora_optimizer(
    model=model,
    lr_nora=0.005,    # Learning rate for Nora-updated matrices (usually higher)
    lr_adam=0.001,    # Learning rate for Adam-updated vectors/embeddings
    weight_decay=0.1, # Global weight decay
    momentum=0.95,
    beta=0.95
)

# 3. Train as usual! No changes needed in your training loop.
for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
```

## ⚙️ Parameters Explained

When calling `get_nora_optimizer`, you can configure the following hyperparameters:

* **`model`**: Your PyTorch model instance. The function needs this to inspect parameter names and dimensions for proper routing.
* **`lr_nora`** *(float, default: `0.005`)*: The learning rate applied to Nora parameter groups (Linear weights, etc.). Because Nora's mathematical scale differs from Adam, this is typically set slightly higher than a standard Adam learning rate.
* **`lr_adam`** *(float, default: `0.001`)*: The standard learning rate for layers falling back to standard Adam (Biases, LayerNorms, Embeddings, Language Model Heads).
* **`weight_decay`** *(float, default: `0.1`)*: Global weight decay applied to both Nora and Adam parameter groups.
* **`momentum` / `beta`** *(float, default: `0.95`)*: Hyperparameters controlling Nora's dual-momentum smoothing and gradient accumulation.

## 🔍 How the Routing Works Under the Hood

When you call `get_nora_optimizer`, it iterates through `model.named_parameters()`. 
A parameter is assigned to the **Nora group** ONLY IF:
1. It requires a gradient.
2. It has 2 or more dimensions (`param.ndim >= 2`).
3. Its name does NOT contain `"embed"` (Embedding layers).
4. Its name does NOT contain `"lm_head"` (Output projection layers).

All other parameters are safely routed to the **Adam group** within the same optimizer instance.



