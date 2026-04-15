# core/indexer.py

import faiss
import numpy as np
from pathlib import Path

from config.settings import (
    CATALOG_DIR, FAISS_INDEX_PATH, EMBEDDING_DIM, SUPPORTED_EXTENSIONS
)
from core.embedder import FabricEmbedder
from core.database import (
    initialize_database, add_vendor, add_fabric
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_or_create_index() -> faiss.IndexIDMap:
    if FAISS_INDEX_PATH.exists():
        logger.info("Loading existing FAISS IndexIDMap from disk")
        return faiss.read_index(str(FAISS_INDEX_PATH))

    logger.info("Creating fresh IndexIDMap(IndexFlatIP)")
    base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(base_index)


def save_index(index: faiss.IndexIDMap):
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index saved → {FAISS_INDEX_PATH}")


def index_vendor_images(
    vendor_name: str,
    contact: str,
    image_paths: list,
    fabric_code: str = None,
    price: float = None
):
    initialize_database()

    embedder = FabricEmbedder()
    index = load_or_create_index()

    vendor_id = add_vendor(vendor_name, contact)
    logger.info(f"Vendor '{vendor_name}' → vendor_id={vendor_id}")

    import shutil
    vendor_dir = Path(CATALOG_DIR) / vendor_name
    vendor_dir.mkdir(parents=True, exist_ok=True)

    permanent_paths = []
    for img_path in image_paths:
        img_path = Path(img_path)
        dest = vendor_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        permanent_paths.append(dest)

    image_paths = permanent_paths

    new_vectors = []
    new_ids = []  # Tracking the exact database IDs
    indexed_count = 0
    skipped_count = 0

    for image_path in image_paths:
        image_path = Path(image_path)

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported file: {image_path.name}")
            skipped_count += 1
            continue

        try:
            # 1. EMBED FIRST
            vector = embedder.embed(image_path)

            # 2. DATABASE SECOND
            fabric_id = add_fabric(
                vendor_id=vendor_id,
                image_path=str(image_path.resolve()),
                fabric_code=fabric_code,
                price=price
            )

            if fabric_id is None:
                logger.warning(f"Already indexed, skipping: {image_path.name}")
                skipped_count += 1
                continue

            # 3. QUEUE FOR FAISS
            new_vectors.append(vector)
            new_ids.append(fabric_id)

            logger.info(f"Embedded: {image_path.name} → fabric_id={fabric_id}")
            indexed_count += 1

        except Exception as e:
            logger.error(f"Failed to embed {image_path.name}: {e}")
            skipped_count += 1
            continue

    if new_vectors:
        vectors_array = np.array(new_vectors, dtype=np.float32)
        ids_array = np.array(new_ids, dtype=np.int64)
        index.add_with_ids(vectors_array, ids_array)
        save_index(index)
        logger.info(f"Added {indexed_count} vectors to FAISS index with explicit IDs")
    else:
        logger.warning("No new vectors to add")

    logger.info(f"Done — indexed: {indexed_count}, skipped: {skipped_count}")
    return indexed_count, skipped_count


def index_catalog_bulk():
    initialize_database()

    catalog_path = Path(CATALOG_DIR)
    if not catalog_path.exists():
        logger.error(f"Catalog folder not found: {catalog_path}")
        return

    vendor_folders = [f for f in catalog_path.iterdir() if f.is_dir()]
    logger.info(f"Found {len(vendor_folders)} vendor folders")

    total_indexed = 0
    total_skipped = 0

    for vendor_folder in vendor_folders:
        vendor_name = vendor_folder.name
        image_paths = [
            p for p in vendor_folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_paths:
            continue

        logger.info(f"Indexing vendor: {vendor_name} ({len(image_paths)} images)")
        indexed, skipped = index_vendor_images(
            vendor_name=vendor_name,
            contact=None,
            image_paths=image_paths
        )
        total_indexed += indexed
        total_skipped += skipped

    logger.info(f"Bulk indexing complete — total indexed: {total_indexed}, skipped: {total_skipped}")

if __name__ == "__main__":
    index_catalog_bulk()