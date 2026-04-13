# core/indexer.py
#
# PURPOSE: Scans the catalog folder, embeds all images, saves vectors to FAISS
#          and metadata to SQLite. This file is run ONCE to build the index,
#          then only when new vendors/images are added.
#
# WHY separate from app.py:
#   Indexing is a write operation (slow, heavy).
#   Searching is a read operation (fast, lightweight).
#   Mixing both in one file means every time the UI loads, it risks
#   accidentally re-indexing. Keeping them separate is safer and cleaner.
#
# SYNC CONTRACT:
#   SQLite is written FIRST → auto-generates fabric_id.
#   That same id is used as the FAISS vector position.
#   This is the only rule that keeps both systems in sync.

import faiss
import numpy as np
from pathlib import Path

from config.settings import (
    CATALOG_DIR, FAISS_INDEX_PATH, EMBEDDING_DIM, SUPPORTED_EXTENSIONS
)
from core.embedder import FabricEmbedder
from core.database import (
    initialize_database, add_vendor, add_fabric, get_next_faiss_id
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_or_create_index() -> faiss.IndexFlatIP:
    # Loads existing FAISS index from disk, or creates a fresh one if none exists
    #
    # WHY IndexFlatIP (Inner Product):
    #   Since all our vectors are L2-normalized (length=1), inner product
    #   between two vectors equals their cosine similarity.
    #   Score of 1.0 = identical, 0.0 = completely different.
    #   Multiply by 100 → match percentage. Simple and accurate.
    if FAISS_INDEX_PATH.exists():
        logger.info("Loading existing FAISS index from disk")
        return faiss.read_index(str(FAISS_INDEX_PATH))

    logger.info("No existing index found — creating fresh IndexFlatIP")
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def save_index(index: faiss.IndexFlatIP):
    # Saves the FAISS index to disk so it persists between app restarts
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
    # Indexes all images for one vendor — writes to SQLite then FAISS
    # Called from app.py when a new vendor is added via the UI
    initialize_database()

    embedder = FabricEmbedder()
    index = load_or_create_index()

    # Add vendor to SQLite, get vendor_id back
    # Add vendor to SQLite, get vendor_id back
    vendor_id = add_vendor(vendor_name, contact)
    logger.info(f"Vendor '{vendor_name}' → vendor_id={vendor_id}")

    # Permanently copy images to catalog folder before indexing
    # WHY: Streamlit uploads are temp files that get deleted after the request.
    # If we index temp paths, images will show as missing on next app load.
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

    image_paths = permanent_paths  # all paths are now permanent on disk
    logger.info(f"Vendor '{vendor_name}' → vendor_id={vendor_id}")

    new_vectors = []
    indexed_count = 0
    skipped_count = 0

    for image_path in image_paths:
        image_path = Path(image_path)

        # Skip unsupported file types like .DS_Store, .txt etc
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported file: {image_path.name}")
            skipped_count += 1
            continue

        try:
            # Get what the next FAISS position should be
            # WHY here and not once before the loop:
            #   Each successful insert increments the count in SQLite.
            #   Fetching inside the loop ensures the id is always accurate
            #   even if a previous image in the same batch was skipped.
            faiss_id = get_next_faiss_id()

            # Write to SQLite first — this is the sync contract
            fabric_id = add_fabric(
                vendor_id=vendor_id,
                image_path=str(image_path.resolve()),
                faiss_id=faiss_id,
                fabric_code=fabric_code,
                price=price
            )

            if fabric_id is None:
                # add_fabric returns None if image_path already exists (UNIQUE constraint)
                logger.warning(f"Already indexed, skipping: {image_path.name}")
                skipped_count += 1
                continue

            # Embed the image and collect the vector
            vector = embedder.embed(image_path)
            new_vectors.append(vector)

            logger.info(f"Embedded: {image_path.name} → faiss_id={faiss_id}")
            indexed_count += 1

        except Exception as e:
            # Don't crash the whole batch if one image is corrupt
            logger.error(f"Failed to embed {image_path.name}: {e}")
            skipped_count += 1
            continue

    if new_vectors:
        # Stack all new vectors into a 2D array and add to FAISS in one shot
        # WHY one shot: FAISS add() is faster when called once with a batch
        #               than called N times with one vector each
        vectors_array = np.array(new_vectors, dtype=np.float32)
        index.add(vectors_array)
        save_index(index)
        logger.info(f"Added {indexed_count} vectors to FAISS index")
    else:
        logger.warning("No new vectors to add")

    logger.info(f"Done — indexed: {indexed_count}, skipped: {skipped_count}")
    return indexed_count, skipped_count


def index_catalog_bulk():
    # Scans entire catalog/ folder and indexes everything — used for first-time setup
    # Folder structure expected: catalog/vendor_name/image1.jpg, image2.jpg ...
    #
    # WHY this folder convention:
    #   Vendor name is derived from the folder name automatically.
    #   No need to manually type vendor names when bulk-importing 400+ images.
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
        vendor_name = vendor_folder.name  # folder name = vendor name
        image_paths = [
            p for p in vendor_folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_paths:
            logger.warning(f"No images found in {vendor_folder.name}, skipping")
            continue

        logger.info(f"Indexing vendor: {vendor_name} ({len(image_paths)} images)")
        indexed, skipped = index_vendor_images(
            vendor_name=vendor_name,
            contact=None,   # contact unknown during bulk import, can be added later
            image_paths=image_paths
        )
        total_indexed += indexed
        total_skipped += skipped

    logger.info(f"Bulk indexing complete — total indexed: {total_indexed}, skipped: {total_skipped}")


if __name__ == "__main__":
    index_catalog_bulk()