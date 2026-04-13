# core/embedder.py
#
# PURPOSE: Loads DINOv2 and converts any fabric image into a 384-dim vector.
# WHY: Raw images can't be compared mathematically. Vectors can.
#      Two visually similar fabrics will produce vectors that are "close"
#      in 384-dimensional space — that closeness is what FAISS measures.

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from typing import Union

from config.settings import MODEL_NAME, EMBEDDING_DIM
from utils.logger import get_logger

logger = get_logger(__name__)


class FabricEmbedder:

    def __init__(self):
        # Loads DINOv2 model and processor onto CPU/GPU, sets to eval mode
        logger.info(f"Loading model: {MODEL_NAME}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # AutoImageProcessor handles resize + normalize automatically using ImageNet stats
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        # eval() disables dropout layers — critical for consistent, reproducible embeddings
        self.model.eval()

        logger.info(f"Model loaded | output dim: {EMBEDDING_DIM} ✓")

    def _load_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        # Accepts file path or PIL Image, always returns clean RGB PIL Image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        # Convert handles RGBA PNGs, grayscale, palette images — all become RGB
        return image.convert("RGB")

    @torch.no_grad()
    def embed(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        # Takes one image, returns a single L2-normalized vector of shape (384,)
        #
        # WHY @torch.no_grad():
        #   During training, PyTorch tracks every operation to compute gradients.
        #   We are only doing inference (no training), so gradients waste memory.
        #   This decorator disables that tracking entirely.
        #
        # WHY CLS token (index 0):
        #   DINOv2 splits the image into patches. The CLS token at position 0
        #   is a special token that aggregates information from ALL patches.
        #   It represents the "global summary" of the image — perfect for
        #   whole-image similarity matching.
        #
        # WHY L2 normalize:
        #   Normalization puts all vectors on a unit sphere (length = 1).
        #   This means dot product between two vectors = cosine similarity.
        #   FAISS IndexFlatIP (Inner Product) then directly gives us
        #   cosine similarity scores, which we convert to match percentage.

        image = self._load_image(image_input)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Shape: (1, num_patches + 1, 384) → take CLS token → (1, 384) → (384,)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embedding = cls_embedding.squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding  # shape: (384,)

    def embed_batch(self, image_paths: list, batch_size: int = 16) -> np.ndarray:
        # Embeds a list of images in batches, returns array of shape (N, 384)
        # WHY batching: Processing 16 images at once is faster than 16 separate
        # forward passes due to how PyTorch parallelizes matrix operations on CPU.
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images = [self._load_image(p) for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # L2 normalize each row independently
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            cls_embeddings = cls_embeddings / np.maximum(norms, 1e-10)

            all_embeddings.append(cls_embeddings)
            logger.debug(f"Batch {i // batch_size + 1} done ({len(batch_paths)} images)")

        return np.vstack(all_embeddings)  # shape: (N, 384)


if __name__ == "__main__":
    embedder = FabricEmbedder()
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    vec = embedder.embed(dummy)
    print(f"Shape : {vec.shape}")        # must be (384,)
    print(f"Norm  : {np.linalg.norm(vec):.4f}")  # must be ~1.0