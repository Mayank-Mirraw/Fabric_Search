
An internal automation tool designed for Mirraw to match designer fabric requests with vendor catalogs using state-of-the-art Computer Vision.

##  The Technology
- **Brain:** Meta's **DINOv2** (Vision Transformer). Unlike traditional CNNs, DINOv2 understands global textures and intricate patterns, making it perfect for high-end ethnic fabrics.
- **Memory:** **FAISS** (Facebook AI Similarity Search) for high-speed vector retrieval.
- **Backend:** FastAPI for seamless integration with future Mirraw workflows.

##  Project Structure
- `app/`: API entry points.
- `core/`: The engine (Embedder, Indexer, Searcher).
- `data/`: Local storage for catalog images and search indexes (Git-ignored).
- `utils/`: PDF importers and image processing helpers.

##  Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Place vendor images in `data/catalog/`.
4. Run `python app/app.py` to start the service.

---
*Built with ❤️ at Mirraw*
