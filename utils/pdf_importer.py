# utils/pdf_importer.py
#
# PURPOSE: Extracts fabric images + metadata from vendor PDF catalogs and
#          indexes them into FAISS + SQLite automatically.
#
# WHY pymupdf (fitz):
#   Best library for extracting embedded images from PDFs at full original quality.
#   Also extracts text reliably from table cells.
#
# ASSUMPTION:
#   PDF follows fixed column order: Date, Month, Vendor, Image, Fabric Code,
#   Vendor Code, Work Width, Price, Fabric Base, Technique, Colour, Status,
#   Designer Comment, Number of Times Used.
#   One image per row, no empty image cells.

import fitz  # pymupdf
import re
from pathlib import Path
from PIL import Image
import io

from core.indexer import index_vendor_images
from utils.logger import get_logger

logger = get_logger(__name__)

# Column indices based on your PDF template
COL_VENDOR      = 3   # "Shama Creations"
COL_FABRIC_CODE = 4   # "ASF490"
COL_PRICE       = 7   # "600"


def extract_rows_from_pdf(pdf_path: str) -> list:
    # Extracts all text rows from the PDF table, skipping the header row
    doc = fitz.open(pdf_path)
    rows = []

    for page in doc:
        blocks = page.get_text("blocks")
        # Each block is (x0, y0, x1, y1, text, block_no, block_type)
        # Sort top to bottom by y0 coordinate
        blocks = sorted(blocks, key=lambda b: b[1])

        for block in blocks:
            text = block[4].strip()
            if not text or text == "":
                continue
            rows.append(text)

    doc.close()
    return rows


def extract_images_from_pdf(pdf_path: str) -> list:
    # Extracts all embedded images from PDF in top-to-bottom order
    # Returns list of PIL Image objects
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        # get_images() returns list of (xref, smask, w, h, bpc, cs, alt_cs, name, filter, referencer)
        image_list = page.get_images(full=True)

        # Sort images by their vertical position on page
        image_positions = []
        for img in image_list:
            xref = img[0]
            # Get the bounding box of where this image appears on the page
            rects = page.get_image_rects(xref)
            if rects:
                y_position = rects[0].y0
                image_positions.append((y_position, xref))

        # Sort by vertical position so order matches table rows
        image_positions.sort(key=lambda x: x[0])

        for _, xref in image_positions:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(pil_image)

    doc.close()
    logger.info(f"Extracted {len(images)} images from {Path(pdf_path).name}")
    return images

def parse_table_data(pdf_path: str) -> list:
    # Parses PDF using x-coordinate column detection
    # Handles vendor names split across two x positions and composite prices like "Top-950"
    doc = fitz.open(pdf_path)
    parsed_rows = []

    # Skip these words found in header or noise
    SKIP_WORDS = {
        "VENDOR", "IMAGE", "DATE", "Month", "January", "February", "March",
        "April", "May", "June", "July", "August", "September", "October",
        "November", "December", "FABRIC", "CODE", "PRICE", "WIDTH", "WORK",
        "ASF2359", "Technique", "COLOUR", "Status", "BASE"
    }

    for page in doc:
        words = page.get_text("words")

        # Vendor name spans x=130 to x=200 (covers both "Aarav" x=143 and "Tex" x=160)
        VENDOR_X      = (130, 210)
        FABRIC_CODE_X = (220, 260)
        PRICE_X       = (340, 410)

        vendor_words = [
            (w[1], w[0], w[4]) for w in words
            if VENDOR_X[0] <= w[0] <= VENDOR_X[1]
            and w[4] not in SKIP_WORDS
        ]
        fabric_words = [
            (w[1], w[4]) for w in words
            if FABRIC_CODE_X[0] <= w[0] <= FABRIC_CODE_X[1]
            and w[4] not in SKIP_WORDS
            and re.match(r'^[A-Z]{2,5}\d{3,6}$', w[4])  # strict fabric code pattern
        ]
        price_words = [
            (w[1], w[4]) for w in words
            if PRICE_X[0] <= w[0] <= PRICE_X[1]
            and w[4] not in SKIP_WORDS
        ]

        # Group vendor words by row — merge words within 15px vertically
        # AND within same row horizontally (both "Aarav" and "Tex" same y)
        vendor_rows = {}
        for y, x, text in sorted(vendor_words):
            matched_key = None
            for key in vendor_rows:
                if abs(key - y) <= 15:
                    matched_key = key
                    break
            if matched_key is not None:
                # Append in x order so "Aarav Tex" not "Tex Aarav"
                vendor_rows[matched_key].append((x, text))
            else:
                vendor_rows[y] = [(x, text)]

        # Sort words within each vendor row by x to get correct word order
        vendor_list = []
        for y in sorted(vendor_rows.keys()):
            words_in_row = sorted(vendor_rows[y], key=lambda w: w[0])
            full_name = " ".join(w[1] for w in words_in_row).strip()
            vendor_list.append((y, full_name))

        fabric_list = sorted(fabric_words)
        price_list  = sorted(price_words)

        # Extract numeric price from values like "Top-950", "Dupatta-380", "500"
        def extract_price(raw: str) -> float:
            # Try plain number first
            try:
                return float(raw)
            except ValueError:
                pass
            # Try extracting number after dash e.g. "Top-950" → 950
            match = re.search(r'(\d+(?:\.\d+)?)$', raw)
            return float(match.group(1)) if match else None

        for i in range(min(len(vendor_list), len(fabric_list))):
            vendor_name = vendor_list[i][1]
            fabric_code = fabric_list[i][1]

            # Find closest price row by y position
            price = None
            if price_list:
                closest_price = min(price_list, key=lambda p: abs(p[0] - fabric_list[i][0]))
                price = extract_price(closest_price[1])

            if vendor_name and fabric_code:
                parsed_rows.append({
                    "vendor_name": vendor_name,
                    "fabric_code": fabric_code,
                    "price"      : price,
                })

    doc.close()
    logger.info(f"Parsed {len(parsed_rows)} rows from {Path(pdf_path).name}")
    return parsed_rows



def remove_background(pil_image: Image.Image) -> Image.Image:
    # Removes background from fabric image using rembg
    # Returns RGB PIL image with background replaced by white
    from rembg import remove
    output = remove(pil_image)
    # rembg returns RGBA — paste on white background for clean RGB
    background = Image.new("RGB", output.size, (255, 255, 255))
    background.paste(output, mask=output.split()[3])
    return background


def import_pdf(pdf_path: str, remove_bg: bool = False):
    # Main entry point — extracts images + metadata from PDF and indexes everything
    # remove_bg=True enables background removal before indexing (slower but cleaner)
    pdf_path = Path(pdf_path)
    logger.info(f"Starting PDF import: {pdf_path.name}")

    # Step 1 — extract images and table data in parallel
    images   = extract_images_from_pdf(str(pdf_path))
    rows     = parse_table_data(str(pdf_path))

    logger.info(f"Images found: {len(images)} | Data rows found: {len(rows)}")

    # Step 2 — zip images with their metadata row
    # Both lists are sorted top-to-bottom so index 0 matches index 0
    if len(images) != len(rows):
        logger.warning(
            f"Image count ({len(images)}) != row count ({len(rows)}). "
            f"Will process up to min of both."
        )

    pairs = list(zip(images, rows))

    if not pairs:
        logger.error("No image-row pairs found. Check PDF format.")
        return 0, 0

    # Step 3 — group by vendor so we call index_vendor_images once per vendor
    from collections import defaultdict
    vendor_groups = defaultdict(list)
    for pil_image, row in pairs:
        vendor_groups[row["vendor_name"]].append((pil_image, row))

    total_indexed = 0
    total_skipped = 0

    for vendor_name, items in vendor_groups.items():
        # Save images temporarily to disk so index_vendor_images can process them
        import tempfile, shutil
        from config.settings import CATALOG_DIR

        vendor_dir = Path(CATALOG_DIR) / vendor_name
        vendor_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        fabric_codes = []
        prices = []

        for i, (pil_image, row) in enumerate(items):
            if remove_bg:
                logger.info(f"Removing background for {row['fabric_code']}")
                pil_image = remove_background(pil_image)

            # Save image permanently to catalog folder
            filename = f"{row['fabric_code']}.jpg"
            dest = vendor_dir / filename

            if not dest.exists():
                pil_image.save(str(dest), "JPEG", quality=95)

            saved_paths.append(dest)
            fabric_codes.append(row["fabric_code"])
            prices.append(row["price"])

        # Index each image individually so fabric_code + price are stored per image
        vendor_indexed = 0
        vendor_skipped = 0

        from core.indexer import index_vendor_images
        from core.database import initialize_database, add_vendor, add_fabric, get_next_faiss_id
        import faiss
        import numpy as np
        from core.embedder import FabricEmbedder
        from config.settings import FAISS_INDEX_PATH, EMBEDDING_DIM, SUPPORTED_EXTENSIONS
        from core.indexer import load_or_create_index, save_index

        initialize_database()
        embedder = FabricEmbedder()
        index = load_or_create_index()
        vendor_id = add_vendor(vendor_name, contact=None)

        new_vectors = []

        for path, fabric_code, price in zip(saved_paths, fabric_codes, prices):
            try:
                faiss_id = get_next_faiss_id() + len(new_vectors)
                fabric_id = add_fabric(
                    vendor_id=vendor_id,
                    image_path=str(path.resolve()),
                    faiss_id=faiss_id,
                    fabric_code=fabric_code,
                    price=price
                )

                if fabric_id is None:
                    logger.warning(f"Already indexed: {path.name}")
                    vendor_skipped += 1
                    continue

                vector = embedder.embed(path)
                new_vectors.append(vector)
                vendor_indexed += 1
                logger.info(f"Embedded {path.name} → faiss_id={faiss_id}")

            except Exception as e:
                logger.error(f"Failed on {path.name}: {e}")
                vendor_skipped += 1

        if new_vectors:
            index.add(np.array(new_vectors, dtype=np.float32))
            save_index(index)

        logger.info(f"Vendor '{vendor_name}' → indexed: {vendor_indexed}, skipped: {vendor_skipped}")
        total_indexed += vendor_indexed
        total_skipped += vendor_skipped

    logger.info(f"PDF import complete — total indexed: {total_indexed}, skipped: {total_skipped}")
    return total_indexed, total_skipped


# ── Run directly from terminal ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m utils.pdf_importer <path_to_pdf> [--remove-bg]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    remove_bg = "--remove-bg" in sys.argv

    indexed, skipped = import_pdf(pdf_file, remove_bg=remove_bg)
    print(f"\n✅ Done — indexed: {indexed}, skipped: {skipped}")