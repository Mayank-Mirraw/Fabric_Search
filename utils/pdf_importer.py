# utils/pdf_importer.py

import fitz  # pymupdf
import re
import io
import uuid
from pathlib import Path
from PIL import Image
from collections import defaultdict
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

def extract_pdf_robust(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    all_pairs = []

    SKIP_WORDS = {
        "VENDOR", "IMAGE", "DATE", "Month", "January", "February", "March",
        "April", "May", "June", "July", "August", "September", "October",
        "November", "December", "FABRIC", "CODE", "PRICE", "WIDTH", "WORK",
        "ASF2359", "Technique", "COLOUR", "Status", "BASE"
    }

    VENDOR_X      = (130, 210)
    FABRIC_CODE_X = (220, 260)
    PRICE_X       = (340, 410)

    for page_num, page in enumerate(doc):
        page_images = []
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if rects:
                y_pos = rects[0].y0
                base_image = doc.extract_image(xref)
                try:
                    pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    page_images.append({"y": y_pos, "img": pil_image})
                except Exception as e:
                    logger.warning(f"Failed to load image on page {page_num}: {e}")

        words = page.get_text("words")
        
        vendor_words = [(w[1], w[0], w[4]) for w in words if VENDOR_X[0] <= w[0] <= VENDOR_X[1] and w[4] not in SKIP_WORDS]
        fabric_words = [(w[1], w[4]) for w in words if FABRIC_CODE_X[0] <= w[0] <= FABRIC_CODE_X[1] and w[4] not in SKIP_WORDS and re.match(r'^[A-Z]{2,5}\d{3,6}$', w[4])]
        price_words  = [(w[1], w[4]) for w in words if PRICE_X[0] <= w[0] <= PRICE_X[1] and w[4] not in SKIP_WORDS]

        def extract_price(raw: str) -> float:
            try: return float(raw)
            except ValueError: pass
            match = re.search(r'(\d+(?:\.\d+)?)$', raw)
            return float(match.group(1)) if match else None

        page_rows = []
        for fy, fcode in fabric_words:
            v_name = None
            if vendor_words:
                nearby_vendors = [w for w in vendor_words if abs(w[0] - fy) <= 15]
                if nearby_vendors:
                    nearby_vendors.sort(key=lambda w: w[1])
                    v_name = " ".join(w[2] for w in nearby_vendors).strip()

            p_val = None
            if price_words:
                closest_price = min(price_words, key=lambda p: abs(p[0] - fy))
                if abs(closest_price[0] - fy) <= 15:
                    p_val = extract_price(closest_price[1])

            if v_name and fcode:
                page_rows.append({
                    "y": fy,
                    "vendor_name": v_name,
                    "fabric_code": fcode,
                    "price": p_val
                })

        for img_dict in page_images:
            if not page_rows:
                continue
            closest_row = min(page_rows, key=lambda r: abs(r["y"] - img_dict["y"]))
            if abs(closest_row["y"] - img_dict["y"]) < 150: 
                all_pairs.append((img_dict["img"], closest_row))

    doc.close()
    logger.info(f"Extracted and paired {len(all_pairs)} items robustly.")
    return all_pairs


def remove_background(pil_image: Image.Image) -> Image.Image:
    from rembg import remove
    output = remove(pil_image)
    background = Image.new("RGB", output.size, (255, 255, 255))
    background.paste(output, mask=output.split()[3])
    return background


def import_pdf(pdf_path: str, remove_bg: bool = False):
    pdf_path = Path(pdf_path)
    logger.info(f"Starting Robust PDF import: {pdf_path.name}")

    pairs = extract_pdf_robust(str(pdf_path))

    if not pairs:
        logger.error("No image-row pairs found. Check PDF format.")
        return 0, 0

    vendor_groups = defaultdict(list)
    for pil_image, row in pairs:
        vendor_groups[(row["vendor_name"] or "Unknown Vendor").strip().title()].append((pil_image, row))

    total_indexed = 0
    total_skipped = 0

    from config.settings import CATALOG_DIR
    from core.database import initialize_database, add_vendor, add_fabric
    from core.embedder import FabricEmbedder
    from core.indexer import load_or_create_index, save_index
    import faiss

    initialize_database()
    embedder = FabricEmbedder()
    index = load_or_create_index()

    for vendor_name, items in vendor_groups.items():
        vendor_dir = Path(CATALOG_DIR) / vendor_name
        vendor_dir.mkdir(parents=True, exist_ok=True)
        vendor_id = add_vendor(vendor_name, contact=None)

        vendor_indexed = 0
        vendor_skipped = 0
        new_vectors = []
        new_ids = [] # Tracking precise IDs

        for pil_image, row in items:
            if remove_bg:
                pil_image = remove_background(pil_image)

            # Ensure valid string for fabric code
            safe_code = row['fabric_code'] if row['fabric_code'] else f"UNKNOWN_{uuid.uuid4().hex[:8].upper()}"
            filename = f"{safe_code}.jpg"
            dest = vendor_dir / filename

            if not dest.exists():
                pil_image.save(str(dest), "JPEG", quality=95)

            try:
                # 1. EMBED FIRST
                vector = embedder.embed(dest)
                
                # 2. DATABASE SECOND
                fabric_id = add_fabric(
                    vendor_id=vendor_id,
                    image_path=str(dest.resolve()),
                    fabric_code=row["fabric_code"],
                    price=row["price"]
                )

                if fabric_id is None:
                    vendor_skipped += 1
                    continue

                # 3. QUEUE FOR FAISS
                new_vectors.append(vector)
                new_ids.append(fabric_id)
                vendor_indexed += 1

            except Exception as e:
                logger.error(f"Failed on {filename}: {e}")
                vendor_skipped += 1

        if new_vectors:
            vectors_array = np.array(new_vectors, dtype=np.float32)
            ids_array = np.array(new_ids, dtype=np.int64)
            index.add_with_ids(vectors_array, ids_array)
            save_index(index)

        total_indexed += vendor_indexed
        total_skipped += vendor_skipped

    logger.info(f"PDF import complete — total indexed: {total_indexed}, skipped: {total_skipped}")
    return total_indexed, total_skipped

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m utils.pdf_importer <path_to_pdf> [--remove-bg]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    remove_bg = "--remove-bg" in sys.argv

    indexed, skipped = import_pdf(pdf_file, remove_bg=remove_bg)
    print(f"\n✅ Done — indexed: {indexed}, skipped: {skipped}")