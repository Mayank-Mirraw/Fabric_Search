# core/database.py

# TWO TABLES:
#   vendors → one row per vendor (name, contact, etc.)
#   fabrics → one row per image (links to vendor, stores FAISS ID, price, code)
#
# WHY separate tables:
#   One vendor sells MANY fabrics. Storing vendor name/contact on every fabric
#   row is wasteful and inconsistent. Separate tables + foreign key = clean.
#
# FAISS ↔ SQLite SYNC RULE (critical):
#   We always write to SQLite FIRST, get the auto-generated fabric_id back,
#   then use that SAME id as the FAISS vector index. This is the contract
#   that keeps both systems in perfect sync forever.

import sqlite3
from pathlib import Path
from config.settings import DATABASE_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def get_connection():
    # Returns a database connection.
    # check_same_thread=False is needed for Streamlit which uses multiple threads.
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Rows behave like dicts: row["vendor_name"] instead of row[0]
    return conn


def initialize_database():
    # Creates tables if they don't already exist.
    # Safe to call every time the app starts — won't overwrite existing data.
    conn = get_connection()
    cursor = conn.cursor()

    # vendors table — one row per vendor
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vendors (
            vendor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT    NOT NULL UNIQUE,   -- UNIQUE prevents duplicate vendor entries
            contact     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # fabrics table — one row per fabric image
    # faiss_id is the critical sync column: it stores the position of this
    # fabric's vector inside the FAISS index. Must never be NULL.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fabrics (
            fabric_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_id    INTEGER NOT NULL,
            image_path   TEXT    NOT NULL UNIQUE,  -- prevents same image indexed twice
            fabric_code  TEXT,                     -- optional, NULL is fine
            price        REAL,                     -- optional, can be updated later
            faiss_id     INTEGER NOT NULL UNIQUE,  -- sync key with FAISS
            indexed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
        )
    """)

    # price_history table — every time price is updated, old value is preserved here
    # WHY: Mirraw negotiates prices with vendors over time. This lets designers
    # see "we paid 670 last time, vendor is now quoting 750" during order calls.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            fabric_id   INTEGER NOT NULL,
            old_price   REAL,
            new_price   REAL,
            changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (fabric_id) REFERENCES fabrics(fabric_id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized ✓")


def add_vendor(vendor_name: str, contact: str = None) -> int:
    # Inserts a new vendor. Returns vendor_id.
    # If vendor already exists, returns their existing id instead of crashing.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vendors (vendor_name, contact)
        VALUES (?, ?)
        ON CONFLICT(vendor_name) DO UPDATE SET contact=excluded.contact
    """, (vendor_name, contact))
    conn.commit()

    cursor.execute("SELECT vendor_id FROM vendors WHERE vendor_name = ?", (vendor_name,))
    vendor_id = cursor.fetchone()["vendor_id"]
    conn.close()
    return vendor_id


def add_fabric(vendor_id: int, image_path: str, faiss_id: int,
               fabric_code: str = None, price: float = None) -> int:
    # Inserts a new fabric row. Returns fabric_id.
    # faiss_id must be passed in — this is the sync contract with FAISS.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO fabrics (vendor_id, image_path, faiss_id, fabric_code, price)
        VALUES (?, ?, ?, ?, ?)
    """, (vendor_id, str(image_path), faiss_id, fabric_code, price))
    conn.commit()

    cursor.execute("SELECT fabric_id FROM fabrics WHERE image_path = ?", (str(image_path),))
    row = cursor.fetchone()
    conn.close()
    return row["fabric_id"] if row else None


def update_fabric_details(fabric_id: int, price: float = None, fabric_code: str = None):
    # Updates price and/or fabric code for an existing fabric.
    # If price changes, the old price is saved to price_history before updating.
    conn = get_connection()
    cursor = conn.cursor()

    if price is not None:
        # Fetch current price first so we can log it to history
        cursor.execute("SELECT price FROM fabrics WHERE fabric_id = ?", (fabric_id,))
        row = cursor.fetchone()
        old_price = row["price"] if row else None

        cursor.execute("""
            INSERT INTO price_history (fabric_id, old_price, new_price)
            VALUES (?, ?, ?)
        """, (fabric_id, old_price, price))

        cursor.execute("UPDATE fabrics SET price = ? WHERE fabric_id = ?", (price, fabric_id))

    if fabric_code is not None:
        cursor.execute("UPDATE fabrics SET fabric_code = ? WHERE fabric_id = ?",
                       (fabric_code, fabric_id))

    conn.commit()
    conn.close()


def get_fabric_by_faiss_id(faiss_id: int) -> dict:
    # Called by searcher.py after FAISS returns a vector match.
    # Takes the faiss_id, returns all vendor + fabric details for the UI.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.fabric_id, f.image_path, f.fabric_code, f.price, f.faiss_id,
               v.vendor_name, v.contact
        FROM fabrics f
        JOIN vendors v ON f.vendor_id = v.vendor_id
        WHERE f.faiss_id = ?
    """, (faiss_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_vendors() -> list:
    # Returns all vendors for the dropdown in the Add Vendor UI tab.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT vendor_id, vendor_name, contact FROM vendors ORDER BY vendor_name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_database_stats() -> dict:
    # Returns counts for the Database Overview dashboard tab.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM vendors")
    vendors = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as total FROM fabrics")
    fabrics = cursor.fetchone()["total"]
    conn.close()
    return {"total_vendors": vendors, "total_fabrics": fabrics}


def get_next_faiss_id() -> int:
    # Returns what the next FAISS index position should be.
    # FAISS vectors are 0-indexed. If 50 vectors exist, next id = 50.
    # This is called by indexer.py before adding a new vector.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM fabrics")
    total = cursor.fetchone()["total"]
    conn.close()
    return total