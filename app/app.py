# app/app.py
#
# PURPOSE: Streamlit UI — the only interface designers and managers interact with.
# WHY Streamlit: Pure Python, no frontend knowledge needed, fast to build internal tools.
#
# THREE TABS:
#   1. Search Fabrics     — upload query image, see ranked vendor matches
#   2. Add Vendor         — add new vendor or upload images to existing vendor
#   3. Database Overview  — stats dashboard + update price/fabric code after order calls

import os
# 1. Silence transformers at the environment level before it even loads
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fixes a common warning on Mac/Linux

import warnings
# 2. Catch standard Python warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 3. Silence Hugging Face's custom internal logger
import transformers
transformers.logging.set_verbosity_error()

import streamlit as st
# ... the rest of your imports and code go here ...

import sys
from pathlib import Path

# WHY this sys.path insert:
# Streamlit runs app.py from the app/ folder. Without this, imports like
# "from core.searcher import..." fail because Python can't find the core/ folder.
# This line adds the project root to Python's search path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image, ImageOps
import tempfile
import os

from core.searcher import FabricSearcher
from core.indexer import index_vendor_images
from core.database import (
    initialize_database, get_all_vendors,
    get_database_stats, update_fabric_details
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mirraw Fabric Search",
    page_icon="🧵",
    layout="wide"
)

# ── Initialize database on every startup ──────────────────────────────────
# Safe to call repeatedly — CREATE TABLE IF NOT EXISTS won't overwrite data
initialize_database()


# ── Cache the searcher so it loads only once per session ──────────────────
# WHY @st.cache_resource:
#   Without this, Streamlit re-runs the entire script on every user interaction.
#   That means reloading DINOv2 + FAISS index every time a button is clicked.
#   cache_resource loads it once and reuses it for the entire session.
@st.cache_resource
def get_searcher():
    try:
        return FabricSearcher()
    except FileNotFoundError:
        return None


# ══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Search Fabrics", "➕ Add Vendor", "📊 Database Overview"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH FABRICS
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Search Fabrics")
    st.caption("Upload a reference fabric image to find the closest vendor matches.")

    uploaded_file = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png", "webp"],
        key="search_upload"
    )

    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")

        col_preview, col_spacer = st.columns([1, 3])
        with col_preview:
            st.image(query_image, caption="Your Query Image", use_container_width=True)

        # 1. ONLY do the math when the button is clicked, and save to session_state
        if st.button("🔍 Find Matches", type="primary"):
            searcher = get_searcher()

            if searcher is None:
                st.error("No index found. Please add vendors first in the 'Add Vendor' tab.")
            else:
                with st.spinner("Searching catalog..."):
                    # Save results to memory!
                    st.session_state['search_results'] = searcher.search(query_image, top_k=top_k)

        # 2. Draw the UI OUTSIDE the button click, reading from memory
        if 'search_results' in st.session_state and st.session_state['search_results']:
            results = st.session_state['search_results']
            
            st.success(f"Top {len(results)} matches found")
            st.divider()

            # Display results as cards in a grid — 3 columns per row
            cols = st.columns(3)

            for i, result in enumerate(results):
                with cols[i % 3]:
                    # Show fabric image
                    # Show formatted fabric image
                    img_path = Path(result["image_path"])
                    if img_path.exists():
                        try:
                            # 1. Open the raw image
                            raw_img = Image.open(img_path)
                            
                            # 2. Force it into a perfect 500x500 square
                            # ImageOps.fit crops from the center without distorting the pattern
                            formatted_img = ImageOps.fit(raw_img, (500, 500), centering=(0.5, 0.5))
                            
                            # 3. Display the perfectly square image
                            st.image(formatted_img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    else:
                        st.warning("Image file not found")

                    # Match percentage badge
                    match_pct = result["match_percentage"]

                    if match_pct >= 80:
                        st.success(f"**{match_pct}% Match**")
                    elif match_pct >= 60:
                        st.warning(f"**{match_pct}% Match**")
                    else:
                        st.error(f"**{match_pct}% Match**")

                    # Vendor details
                    st.markdown(f"**Vendor:** {result['vendor_name']}")
                    st.markdown(f"**Contact:** {result['contact'] or 'Not provided'}")
                    st.markdown(f"**Fabric Code:** {result['fabric_code'] or 'Not assigned'}")
                    st.markdown(f"**Price:** ₹{result['price']}" if result['price'] else "**Price:** Not set")

                    # 3. Use st.form to stop typing from triggering a refresh
                    with st.expander("✏️ Update after order call"):
                        with st.form(key=f"form_{result['fabric_id']}"):
                            new_price = st.number_input(
                                "Update Price (₹)",
                                min_value=0.0,
                                value=float(result["price"]) if result["price"] else 0.0,
                                step=10.0
                            )
                            new_code = st.text_input(
                                "Update Fabric Code",
                                value=result["fabric_code"] or ""
                            )
                            
                            # st.form_submit_button replaces standard st.button inside forms
                            submit_btn = st.form_submit_button("Save Updates")
                            
                            if submit_btn:
                                update_fabric_details(
                                    fabric_id=result["fabric_id"],
                                    price=new_price if new_price > 0 else None,
                                    fabric_code=new_code if new_code else None
                                )
                                # Update the session state so the UI reflects the change immediately
                                result['price'] = new_price if new_price > 0 else None
                                result['fabric_code'] = new_code if new_code else None
                                st.success("Updated ✓")
# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ADD VENDOR
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Add Vendor")

    mode = st.radio(
        "What would you like to do?",
        ["Add New Vendor", "Upload Images to Existing Vendor"],
        horizontal=True
    )

    # ── New Vendor ─────────────────────────────────────────────────────────
    if mode == "Add New Vendor":
        st.subheader("New Vendor Details")

        vendor_name = st.text_input("Vendor Name *", placeholder="e.g. Ravi Textiles")
        contact     = st.text_input("Contact Number", placeholder="e.g. 9876543210")
        fabric_code = st.text_input("Fabric Code (optional)")
        price       = st.number_input("Price per metre ₹ (optional)", min_value=0.0, value=0.0)

        uploaded_images = st.file_uploader(
            "Upload fabric images *",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="new_vendor_upload"
        )

        if st.button("➕ Add Vendor & Index Images", type="primary"):
            if not vendor_name:
                st.error("Vendor name is required.")
            elif not uploaded_images:
                st.error("Please upload at least one image.")
            else:
                # Save uploaded files to a temp directory
                # WHY tempfile: Streamlit uploaded files are in-memory objects.
                # DINOv2 needs actual file paths on disk to open with PIL.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    saved_paths = []
                    for uploaded in uploaded_images:
                        tmp_path = Path(tmp_dir) / uploaded.name
                        tmp_path.write_bytes(uploaded.getvalue())
                        saved_paths.append(tmp_path)

                    with st.spinner(f"Indexing {len(saved_paths)} images..."):
                        indexed, skipped = index_vendor_images(
                            vendor_name=vendor_name,
                            contact=contact or None,
                            image_paths=saved_paths,
                            fabric_code=fabric_code or None,
                            price=price if price > 0 else None
                        )
                        # Clear searcher cache so next search loads the updated index
                        st.cache_resource.clear()

                    st.success(f"✓ Indexed {indexed} images | Skipped {skipped}")

    # ── Existing Vendor ────────────────────────────────────────────────────
    else:
        st.subheader("Upload to Existing Vendor")

        vendors = get_all_vendors()

        if not vendors:
            st.warning("No vendors in database yet. Add a new vendor first.")
        else:
            # Build name → id map for the dropdown
            vendor_map = {v["vendor_name"]: v["vendor_id"] for v in vendors}

            # st.selectbox has built-in search filtering — no extra library needed
            selected_name = st.selectbox(
                "Select Vendor",
                options=list(vendor_map.keys()),
                placeholder="Type to search..."
            )

            selected_vendor = next(v for v in vendors if v["vendor_name"] == selected_name)
            st.info(f"Contact: {selected_vendor['contact'] or 'Not on record'}")

            new_images = st.file_uploader(
                "Upload new fabric images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="existing_vendor_upload"
            )

            if st.button("📤 Upload & Index", type="primary"):
                if not new_images:
                    st.error("Please upload at least one image.")
                else:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        saved_paths = []
                        for uploaded in new_images:
                            tmp_path = Path(tmp_dir) / uploaded.name
                            tmp_path.write_bytes(uploaded.getvalue())
                            saved_paths.append(tmp_path)

                        with st.spinner(f"Indexing {len(saved_paths)} images..."):
                            indexed, skipped = index_vendor_images(
                                vendor_name=selected_name,
                                contact=selected_vendor["contact"],
                                image_paths=saved_paths
                            )
                            st.cache_resource.clear()

                    st.success(f"✓ Indexed {indexed} images | Skipped {skipped}")

    st.divider()
    st.subheader("📄 Import from PDF Catalog")

    uploaded_pdf = st.file_uploader(
        "Upload vendor PDF catalog",
        type=["pdf"],
        key="pdf_upload"
    )

    remove_bg = st.checkbox(
        "Remove background from images",
        value=False,
        help="Enables rembg background removal. Slower but improves search accuracy for images with solid backgrounds."
    )

    if st.button("📥 Parse & Index PDF", type="primary"):
        if not uploaded_pdf:
            st.error("Please upload a PDF first.")
        else:
            # Save PDF temporarily to disk — pymupdf needs a real file path
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_pdf_path = tmp.name

            with st.spinner("Parsing PDF and indexing images... this may take a minute."):
                from utils.pdf_importer import import_pdf
                indexed, skipped = import_pdf(tmp_pdf_path, remove_bg=remove_bg)
                st.cache_resource.clear()

            # Clean up temp file
            Path(tmp_pdf_path).unlink(missing_ok=True)

            st.success(f"✅ Done — Indexed: {indexed} | Skipped: {skipped}")
 
# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATABASE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Database Overview")

    if st.button("🔄 Refresh Stats"):
        st.rerun()

    stats = get_database_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Vendors", stats["total_vendors"])
    with col2:
        st.metric("Total Fabrics Indexed", stats["total_fabrics"])

    st.divider()

    # Vendor list table
    st.subheader("All Vendors")
    vendors = get_all_vendors()

    if not vendors:
        st.info("No vendors indexed yet.")
    else:
        for vendor in vendors:
            st.markdown(f"**{vendor['vendor_name']}** — {vendor['contact'] or 'No contact'}")