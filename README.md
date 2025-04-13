# Transformers for Language Translation

This repository implements a custom Transformer-based model for language translation, inspired by the "Attention Is All You Need" paper. The model is designed to translate sentences between two languages, with the flexibility to train on any language pair.

## Overview

- The Transformer model is based on the "Attention Is All You Need" paper.
- Implements positional encoding, multi-head attention, and masked multi-head attention.
- Includes dynamic batching for variable-length sequences.
- Uses WMT 2014 English-German dataset for demonstration.

## Project Structure

- `transformers_module_notebook.ipynb`: Notebook for experimenting with Transformer components.
- `transformer/`: Contains core modules for the Transformer model.
  - `modules.py`: Implementation of Transformer components (e.g., Multi-Head Attention, Feed Forward Network).
  - `tokenizer.py`: Functions for creating and loading tokenizers.
  - `dataset.py`: Dataset classes for dynamic batching and static dataset handling.
- `language_translation_notebook.ipynb`: Notebook for training and evaluating the Transformer model.
- `download_dataset.py`: Script to download and extract datasets from Kaggle.
- `requirements.txt`: Python dependencies for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harishaa10/transformers.git
   cd transformers
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Authenticate Kaggle API (if downloading datasets):
   - Place your `kaggle.json` file in `~/.kaggle/`.

5. Download the dataset:
   ```bash
   python download_dataset.py
   ```
