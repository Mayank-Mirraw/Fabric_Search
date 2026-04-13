# config/settings.py
#
# PURPOSE: Central configuration file for the entire Fabric Search system.
# WHY: In enterprise projects, hardcoding values (paths, model names, numbers)
#      directly inside logic files is a maintenance nightmare. If you hardcode
#      "facebook/dinov2-small" in 3 files and need to change it, you have to
#      hunt down every occurrence. Here, you change it ONCE and everything updates.
#
# RULE: No other file should contain magic strings or magic numbers.
#       If a value might ever change, it belongs here.

from pathlib import Path

# ── Why pathlib.Path instead of plain strings? ─────────────────────────────
# Path objects are OS-aware. On Windows, paths use backslashes (\).
# On Linux/Mac, they use forward slashes (/).
# Path handles this automatically so your code works on any machine.
# __file__ = the absolute path of THIS file (settings.py)
# .resolve() = converts it to a full absolute path, no ambiguity
# .parent.parent = go up two levels (from config/ → to fabric-search/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Directory Paths ────────────────────────────────────────────────────────
# WHY: All data lives under BASE_DIR so the project is self-contained.
# If you move the project folder, nothing breaks because paths are relative
# to BASE_DIR, not hardcoded to "/home/yourname/fabric-search/..."

CATALOG_DIR = BASE_DIR / "data" / "catalog"   # Vendor fabric images live here
INDEX_DIR   = BASE_DIR / "data" / "index"     # FAISS index + SQLite DB live here

# ── File Paths ─────────────────────────────────────────────────────────────
# WHY: Naming these here means indexer.py and searcher.py both refer to the
# SAME file. No risk of one writing "fabric.index" and other reading "fabrics.index"

FAISS_INDEX_PATH = INDEX_DIR / "fabric.index"   # The FAISS binary index file
DATABASE_PATH    = INDEX_DIR / "mirraw.db"       # SQLite database (replaces metadata.csv)

# ── Model Configuration ────────────────────────────────────────────────────
# WHY dinov2-small for now:
# - We are on CPU. Small model = faster indexing, less RAM usage.
# - 384 dimensions is still very powerful for texture/pattern matching.
# - When we move to a GPU server in production, we change ONE line here
#   (dinov2-small → dinov2-large) and EMBEDDING_DIM (384 → 1024).
#   Nothing else in the codebase needs to change.

MODEL_NAME    = "facebook/dinov2-small"
EMBEDDING_DIM = 384   # dinov2-small outputs 384-dimensional vectors
                      # dinov2-base  = 768 dims
                      # dinov2-large = 1024 dims

# ── Image Preprocessing Constants ─────────────────────────────────────────
# WHY 224: DINOv2 was trained on 224x224 images. Feeding it a different size
# doesn't break it, but 224 is the "native" size it understands best.
IMAGE_SIZE = 224

# WHY these specific mean/std values:
# DINOv2 was trained on ImageNet, which normalized all images using these
# exact values. If you don't apply the same normalization at inference time,
# the model "sees" a different distribution than it was trained on → worse results.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ── Search Configuration ───────────────────────────────────────────────────
# WHY TOP_K = 10: Return top 10 closest matches by default.
# The UI will display these ranked by match percentage.
# Designers can scan 10 options quickly; more than that is overwhelming.
TOP_K = 10

# ── Supported Image Formats ────────────────────────────────────────────────
# WHY a set (not a list): Sets have O(1) lookup. Checking "is .jpg in this set"
# is instant regardless of how many extensions are listed.
# We check this in indexer.py to skip non-image files like .DS_Store or .txt
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}