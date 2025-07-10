"""
Local Embedding Module for Research Assistant Chat Bot

This module provides local embedding generation capabilities using 
sentence transformers and dimensionality reduction techniques.

Classes:
    LocalEmbedder: Main class for generating local embeddings with custom dimensions

Functions:
    get_text_for_embedding: Combines database row fields into embedding text
    update_embeddings: Main function to process and update database embeddings
    test_embeddings: Test function to verify embedding setup
"""

from .local_embeddings import LocalEmbedder, get_text_for_embedding, update_embeddings, test_embeddings

__version__ = "1.0.0"
__author__ = "Research Assistant Team"

__all__ = [
    "LocalEmbedder",
    "get_text_for_embedding", 
    "update_embeddings",
    "test_embeddings"
] 