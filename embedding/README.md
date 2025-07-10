# Local Embedding Module

This module provides local embedding generation with custom dimensions (1536D) for the Research Assistant Chat Bot project.

## Features

- 🏠 **Local Processing**: No API calls, complete privacy
- 🎯 **Custom Dimensions**: Generates exactly 1536-dimensional vectors
- 🚀 **GPU Acceleration**: Automatically uses CUDA if available
- 🔄 **Batch Processing**: Efficient batch processing for large datasets
- 💾 **PCA Caching**: Saves and reuses PCA reducers for consistency
- 📊 **Progress Tracking**: Real-time progress monitoring

## Installation

1. Install dependencies:
```bash
cd embedding
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```bash
DB_PORT=5432
```

## Usage

### Basic Usage

```python
from embedding import LocalEmbedder

# Initialize embedder
embedder = LocalEmbedder()

# Get embeddings for texts
texts = ["Your text here", "Another document"]
embeddings = embedder.get_embeddings(texts)

print(f"Shape: {embeddings.shape}")  # Should be (2, 1536)
```

### Database Processing

```python
from embedding import update_embeddings, test_embeddings

# Test the setup first
if test_embeddings():
    print("✅ Setup is working!")
    
    # Process the entire database
    update_embeddings()
else:
    print("❌ Setup failed!")
```

### Command Line Usage

```bash
# Run from project root
cd embedding
python local_embeddings.py
```

## Configuration

### Model Options

You can change the base model by modifying `MODEL_NAME`:

```python
# High-quality models (recommended)
MODEL_NAME = "BAAI/bge-large-en-v1.5"        # 1024D → 1536D (default)
MODEL_NAME = "intfloat/e5-large-v2"          # 1024D → 1536D

# Faster, smaller models
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768D → 1536D
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384D → 1536D
```

### Dimension Handling

- **Expansion (1024 → 1536)**: Uses zero-padding to increase dimensions
- **Expansion (384 → 1536)**: Uses zero-padding to increase dimensions  
- **Exact Match**: No transformation needed

## Performance

### Expected Performance (500K records)

| Hardware | Processing Time | Memory Usage |
|----------|----------------|--------------|
| **CPU Only** | ~2-4 hours | ~4-8 GB |
| **GPU (8GB)** | ~30-60 minutes | ~6-10 GB |
| **GPU (16GB+)** | ~20-40 minutes | ~8-12 GB |

### Optimization Tips

1. **Batch Size**: Adjust `embedding_batch_size` based on GPU memory
   - 8GB GPU: `batch_size=32`
   - 16GB GPU: `batch_size=64` 
   - 24GB+ GPU: `batch_size=128`

2. **Database Batch Size**: Increase for faster database operations
   - Default: `batch_size=1000`
   - High-memory: `batch_size=2000`

## File Structure

```
embedding/
├── __init__.py              # Package initialization
├── local_embeddings.py      # Main embedding script
├── requirements.txt         # Dependencies
├── README.md               # This file
└── pca_reducer_*.pkl       # Cached PCA reducers (auto-generated)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   embedder.get_embeddings(texts, batch_size=16)
   ```

2. **Model Download Fails**
   ```bash
   # Pre-download models
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
   ```

3. **Database Connection Issues**
   - Check database credentials in DB_CONFIG
   - Ensure PostgreSQL is running
   - Verify network connectivity

### Performance Monitoring

```python
import torch

# Monitor GPU usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"GPU Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
```

## Cost Comparison

| Method | Cost | Time | Control |
|--------|------|------|---------|
| **OpenAI API** | $3-5 | 30-60 min | ❌ External |
| **Local Embedding** | $0 | 30-60 min | ✅ Complete |

Local embedding gives you the same performance with:
- ✅ Zero ongoing costs
- ✅ Complete data privacy  
- ✅ Custom dimensions
- ✅ Offline capability

## Advanced Configuration

### Custom PCA Settings

```python
from sklearn.decomposition import PCA

# Custom PCA with specific variance retention
embedder = LocalEmbedder()
embedder.reducer = PCA(n_components=1536, random_state=42)
```

### Multi-GPU Support

```python
# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder = LocalEmbedder()
embedder.model.to(device)
``` 