# Mistral 7B PyTorch Implementation

This repository contains a clean and efficient PyTorch implementation of the Mistral 7B language model, focusing on clarity and adherence to the original architecture.

## Features

* **Mistral 7B Architecture:** Implements the core components of Mistral 7B, including:
    * Grouped Query Attention (GQA)
    * Rotary Position Embeddings (RoPE)
    * RMS Normalization
    * Feedforward Network with SiLU activation
* **Modular Design:** Code is structured into modular classes for easy understanding and modification.
* **Clear and Well-Documented:** Comprehensive type hints and docstrings are provided for all classes and functions.
* **Correct Implementation:** Ensures accurate implementation of key architectural details, including RMSNorm and RoPE.
* **Efficient Attention:** Optimized attention mechanism for efficient processing.

## Requirements

* Python 3.x
* PyTorch

## Usage

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install PyTorch:**

    ```bash
    # Install PyTorch according to your system and CUDA setup.
    # See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    ```

3.  **Use the Model:**

    ```python
    import torch
    from mistral_7b import Mistral7B, ModelArgs

    # Define model arguments
    args = ModelArgs(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        vocab_size=32000
    )

    # Initialize the model
    model = Mistral7B(args)

    # Example input (batch_size, sequence_length)
    tokens = torch.randint(0, args.vocab_size, (1, 10))

    # Forward pass
    logits, cache = model(tokens)

    print(logits.shape) # Output logits shape: (1, 10, 32000)
    ```

## Model Architecture

The model consists of the following components:

* **Token Embeddings:** Maps input token IDs to dense vectors.
* **Transformer Blocks:** Stacked layers of attention and feedforward networks.
* **Grouped Query Attention (GQA):** Efficient attention mechanism.
* **Rotary Position Embeddings (RoPE):** Adds positional information to the input embeddings.
* **RMS Normalization:** Normalizes the output of each sub-layer.
* **Feedforward Network:** Uses SiLU activation.
* **Output Head:** Linear layer that maps the final hidden states to logits.

## Key Implementation Details

* **Grouped Query Attention (GQA):** The attention mechanism efficiently handles key and value projections.
* **Rotary Position Embeddings (RoPE):** RoPE is used to encode positional information.
* **RMS Normalization:** RMSNorm is used for layer normalization.
* **SiLU Activation:** SiLU activation function is used in the feedforward network.
* **Cache Handling:** The cache is properly implemented to improve inference speed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

[MIT License](LICENSE)
