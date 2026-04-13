# core/searcher.py
#
# PURPOSE: Takes a query image, finds the closest matching fabrics in FAISS,
#          fetches vendor details from SQLite, returns ranked results with
#          match percentage.
#
# WHY separate from indexer.py:
#   Searcher only READS — it never writes to FAISS or SQLite.
#   This separation means the UI can safely call search() dozens of times
#   without any risk of corrupting the index.
#
# MATCH PERCENTAGE:
#   Since vectors are L2-normalized, FAISS inner product score is cosine similarity.
#   Range is 0.0 to 1.0. Multiply by 100 → clean percentage for the UI.

import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union

from config.settings import FAISS_INDEX_PATH, TOP_K
from core.embedder import FabricEmbedder
from core.database import get_fabric_by_faiss_id, update_fabric_details
from utils.logger import get_logger

logger = get_logger(__name__)


def load_index() -> faiss.IndexFlatIP:
    # Loads FAISS index from disk — raises clear error if index doesn't exist yet
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_INDEX_PATH}. "
            f"Run indexer.py first to build the index."
        )
    logger.info("Loading FAISS index from disk")
    return faiss.read_index(str(FAISS_INDEX_PATH))


class FabricSearcher:

    def __init__(self):
        # Loads FAISS index and DINOv2 embedder once — reused for every search call
        #
        # WHY load once in __init__ and not inside search():
        #   Loading the model and index takes 2-3 seconds each time.
        #   In Streamlit, search() can be called on every user interaction.
        #   Loading once at startup means every search is instant after that.
        self.index = load_index()
        self.embedder = FabricEmbedder()
        logger.info(f"Searcher ready — {self.index.ntotal} vectors in index")

    def search(self, query_image: Union[str, Path, Image.Image], top_k: int = TOP_K) -> list:
        # Takes a query image, returns top_k matches as a list of dicts with match %
        #
        # Returns list of dicts, each containing:
        #   vendor_name, contact, image_path, fabric_code, price, match_percentage
        #   ranked from highest to lowest match

        if self.index.ntotal == 0:
            logger.warning("Index is empty — no vendors indexed yet")
            return []

        # Embed the query image into a 384-dim vector
        query_vector = self.embedder.embed(query_image)

        # FAISS expects shape (1, 384) — add batch dimension
        query_vector = np.array([query_vector], dtype=np.float32)

        # Search FAISS — returns distances and faiss_ids of top_k closest vectors
        # WHY min(top_k, ntotal): if index has fewer vectors than top_k,
        # FAISS will crash. This prevents that edge case.
        k = min(top_k, self.index.ntotal)
        scores, faiss_ids = self.index.search(query_vector, k)

        # scores shape: (1, k) — unwrap the batch dimension
        scores = scores[0]
        faiss_ids = faiss_ids[0]

        results = []
        for score, faiss_id in zip(scores, faiss_ids):

            # FAISS returns -1 as faiss_id when it can't find enough matches
            if faiss_id == -1:
                continue

            # Fetch full vendor + fabric details from SQLite using faiss_id
            fabric = get_fabric_by_faiss_id(int(faiss_id))

            if fabric is None:
                # This should never happen if sync is maintained, but log it if it does
                logger.warning(f"faiss_id={faiss_id} found in index but missing in SQLite")
                continue

            # Convert cosine similarity (0.0–1.0) to match percentage
            #MIN_SCORE = 0.45
            #clamped = max(MIN_SCORE, min(1.0, float(score)))
            #match_percentage = round(((clamped - MIN_SCORE) / (1.0 - MIN_SCORE)) * 100, 2)

            raw_score = float(score)
            safe_score = max(0.0, raw_score)
            match_percentage = round(safe_score * 100, 1)


            results.append({
                "vendor_name"      : fabric["vendor_name"],
                "contact"          : fabric["contact"],
                "image_path"       : fabric["image_path"],
                "fabric_code"      : fabric["fabric_code"],
                "price"            : fabric["price"],
                "faiss_id"         : faiss_id,
                "fabric_id"        : fabric["fabric_id"],
                "match_percentage" : match_percentage
            })

        # Sort by match percentage descending — highest match first
        results.sort(key=lambda x: x["match_percentage"], reverse=True)
        logger.info(f"Search complete — top match: {results[0]['match_percentage']}% "
                    f"({results[0]['vendor_name']})" if results else "No results found")
        return results

    def update_fabric(self, fabric_id: int, price: float = None, fabric_code: str = None):
        # Updates price or fabric code for a fabric after an order call with vendor
        # Old price is automatically saved to price_history in SQLite
        update_fabric_details(fabric_id, price=price, fabric_code=fabric_code)
        logger.info(f"Updated fabric_id={fabric_id} — price={price}, code={fabric_code}")


if __name__ == "__main__":
    # Quick test — searches using a random dummy image
    import numpy as np
    from PIL import Image

    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    searcher = FabricSearcher()
    results = searcher.search(dummy, top_k=3)

    for r in results:
        print(f"{r['match_percentage']}% — {r['vendor_name']} — {r['image_path']}")