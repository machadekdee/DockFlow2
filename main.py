from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import uuid
import textwrap
import cv2
import numpy as np

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import pandas as pd
import streamlit as st

# =========================================================
# 1. APP CONFIGURATION & STYLING
# =========================================================

st.set_page_config(
    page_title="DOCKFLOW | Smart Warehouse Command Center",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup Thai font for Matplotlib to prevent tofu [?] boxes
thai_font_file = Path(__file__).parent / "Sarabun-Regular.ttf"
if thai_font_file.exists():
    try:
        fm.fontManager.addfont(str(thai_font_file))
        selected_thai_font = "Sarabun"
    except Exception:
        selected_thai_font = "sans-serif"
else:
    thai_font_candidates = [
        "Sarabun",
        "Loma",
        "Garuda",
        "Tahoma",
        "Leelawadee UI",
        "Leelawadee",
        "Angsana New",
        "Cordia New",
        "TH Sarabun New",
        "Noto Sans Thai",
    ]
    installed_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_thai_font = next((f for f in thai_font_candidates if f in installed_fonts), "sans-serif")

mpl.rcParams["font.family"] = selected_thai_font if selected_thai_font != "sans-serif" else "sans-serif"
mpl.rcParams["font.sans-serif"] = [selected_thai_font, "Sarabun", "Loma", "Garuda", "Tahoma", "Leelawadee UI", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

DB_FILE = Path(__file__).with_name("dockflow_complete.db")

WAREHOUSES = ["WH-A", "WH-B"]

DOCKS = {
    "A-IN": {"warehouse": "WH-A", "kind": "IN", "title": "A1 · ขาเข้ารับสินค้า"},
    "A-OUT": {"warehouse": "WH-A", "kind": "OUT", "title": "A2 · ขาออกจ่ายสินค้า"},
    "B-IN": {"warehouse": "WH-B", "kind": "IN", "title": "B1 · ขาเข้ารับสินค้า"},
    "B-OUT": {"warehouse": "WH-B", "kind": "OUT", "title": "B2 · ขาออกจ่ายสินค้า"},
}

PALLET_STATUS = {
    "ON_INBOUND": "อยู่บนรถขาเข้า",
    "RECEIVING": "พื้นที่พักรับ",
    "AVAILABLE": "พร้อมใช้ (ในคลัง)",
    "RESERVED": "จองแล้ว / รอหยิบ",
    "PICKED": "พื้นที่พักจ่าย",
    "LOADED": "โหลดขึ้นรถแล้ว",
    "SHIPPED": "ส่งออกแล้ว",
}

TRIP_STATUS = {
    "WAITING": "รอเข้าจอด",
    "DOCKED": "กำลังปฏิบัติงาน",
    "DEPARTED": "ออกจากลานแล้ว",
    "CANCELLED": "ยกเลิกแล้ว",
}

CATEGORY_COLOR = {
    "อิเล็กทรอนิกส์": "#3B82F6",
    "เครื่องใช้ไฟฟ้า": "#8B5CF6",
    "เครื่องดื่ม": "#06B6D4",
    "ของใช้ทั่วไป": "#10B981",
}

PRIORITY_ICON = {"High": "🔴", "Normal": "🔵", "Low": "⚪"}
PRIORITY_ORDER = {"High": 1, "Normal": 2, "Low": 3}


# =========================================================
# 2. DATABASE & TRANSACTION ENGINE
# =========================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def open_db():
    conn = sqlite3.connect(str(DB_FILE), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def atomic():
    conn = open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def rows(sql, params=(), conn=None):
    if conn is not None:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    local_conn = open_db()
    try:
        return [dict(r) for r in local_conn.execute(sql, params).fetchall()]
    finally:
        local_conn.close()


def scalar(sql, params=(), conn=None):
    if conn is not None:
        res = conn.execute(sql, params).fetchone()
        return res[0] if res else 0
    local_conn = open_db()
    try:
        res = local_conn.execute(sql, params).fetchone()
        return res[0] if res else 0
    finally:
        local_conn.close()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def log_event(conn, action, reference, detail):
    conn.execute(
        """
        INSERT INTO events(created_at, action, reference, detail)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp(), action, reference, detail),
    )


def fetch_trip(conn, trip_id):
    res = conn.execute(
        """
        SELECT s.*, t.name AS truck_name, t.slots, t.max_weight
        FROM trips s JOIN trucks t ON t.id = s.truck_id
        WHERE s.id = ?
        """,
        (trip_id,),
    ).fetchone()
    ensure(res is not None, "ไม่พบเที่ยวรถนี้")
    return dict(res)


def fetch_pallet(conn, code):
    res = conn.execute(
        "SELECT * FROM pallets WHERE code = ?", (code.strip().upper(),)
    ).fetchone()
    ensure(res is not None, f"ไม่พบรหัสพาเลต: {code}")
    return dict(res)


# =========================================================
# 3. SCHEMA INITIALIZATION & SEED DATA
# =========================================================

def initialize_database():
    conn = open_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_weight REAL NOT NULL CHECK(unit_weight > 0)
            );

            CREATE TABLE IF NOT EXISTS trucks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slots INTEGER NOT NULL CHECK(slots > 0),
                max_weight REAL NOT NULL CHECK(max_weight > 0),
                plate TEXT,
                carrier TEXT,
                driver_name TEXT,
                driver_phone TEXT,
                truck_type TEXT
            );

            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                warehouse TEXT NOT NULL,
                capacity INTEGER NOT NULL CHECK(capacity > 0)
            );

            CREATE TABLE IF NOT EXISTS trips (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('IN', 'OUT')),
                truck_id TEXT NOT NULL REFERENCES trucks(id),
                warehouse TEXT NOT NULL,
                partner TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('High', 'Normal', 'Low')),
                status TEXT NOT NULL CHECK(status IN ('WAITING', 'DOCKED', 'DEPARTED', 'CANCELLED')),
                dock TEXT,
                created_at TEXT NOT NULL,
                departed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS pallets (
                code TEXT PRIMARY KEY,
                sku TEXT NOT NULL REFERENCES products(sku),
                qty INTEGER NOT NULL CHECK(qty > 0),
                weight REAL NOT NULL CHECK(weight > 0),
                warehouse TEXT NOT NULL,
                location TEXT REFERENCES locations(id),
                status TEXT NOT NULL CHECK(
                    status IN ('ON_INBOUND', 'RECEIVING', 'AVAILABLE', 'RESERVED', 'PICKED', 'LOADED', 'SHIPPED')
                ),
                inbound_id TEXT REFERENCES trips(id),
                outbound_id TEXT REFERENCES trips(id),
                inbound_slot INTEGER,
                load_slot INTEGER,
                lot_number TEXT,
                mfg_date TEXT,
                exp_date TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                action TEXT NOT NULL,
                reference TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            """
        )

        # Database Migration: Check and add new columns if upgrading existing db
        existing_truck_cols = [c[1] for c in conn.execute("PRAGMA table_info(trucks)").fetchall()]
        for col_name, col_type in [
            ("plate", "TEXT"),
            ("carrier", "TEXT"),
            ("driver_name", "TEXT"),
            ("driver_phone", "TEXT"),
            ("truck_type", "TEXT")
        ]:
            if col_name not in existing_truck_cols:
                conn.execute(f"ALTER TABLE trucks ADD COLUMN {col_name} {col_type}")

        existing_pallet_cols = [c[1] for c in conn.execute("PRAGMA table_info(pallets)").fetchall()]
        for col_name, col_type in [
            ("lot_number", "TEXT"),
            ("mfg_date", "TEXT"),
            ("exp_date", "TEXT")
        ]:
            if col_name not in existing_pallet_cols:
                conn.execute(f"ALTER TABLE pallets ADD COLUMN {col_name} {col_type}")

        # Ensure existing demo trucks have plate/carrier and realistic logistics capacities populated
        fleet_upgrades = [
            ("T01", 8, 8000.0, "6 ล้อตู้แห้ง", "70-1122 กทม.", "SCG Logistics", "สมชาย ใจซื่อ", "081-234-5678"),
            ("T02", 8, 8000.0, "6 ล้อตู้แห้ง", "70-3344 กทม.", "Flash Express", "สมเกียรติ มั่นคง", "089-876-5432"),
            ("T03", 8, 8500.0, "6 ล้อตู้ทึบ", "71-5566 ชลบุรี", "Linfox Thailand", "อนุชา รวดเร็ว", "086-555-1234"),
            ("T04", 8, 8000.0, "6 ล้อตู้แห้ง", "70-7788 กทม.", "DOCKFLOW Fleet", "วิชัย ส่งมอบ", "082-444-9988"),
            ("T05", 8, 8500.0, "6 ล้อตู้แห้ง", "72-9900 นนทบุรี", "Kerry Logistics", "ธีระ ขยันยิ่ง", "085-111-2233"),
            ("T06", 4, 3500.0, "4 ล้อตู้ทึบ", "1ฒผ-4455 กทม.", "DOCKFLOW Fleet", "บุญรอด ขับปลอดภัย", "083-777-6655"),
            ("T07", 12, 15000.0, "10 ล้อตู้ทึบ", "73-1234 สระบุรี", "SCG Logistics", "ประสิทธิ์ ชำนาญทาง", "087-999-0011"),
            ("T08", 8, 8000.0, "6 ล้อตู้แห้ง", "70-6677 ปทุมธานี", "รถร่วมบริการ", "มานะ ตรงเวลา", "084-222-3344"),
        ]
        for tid, sl, mw, tt, pl, cr, dn, dp in fleet_upgrades:
            conn.execute(
                """
                UPDATE trucks 
                SET slots = ?, max_weight = ?, truck_type = ?, plate = COALESCE(NULLIF(plate, ''), ?),
                    carrier = COALESCE(NULLIF(carrier, ''), ?), driver_name = COALESCE(NULLIF(driver_name, ''), ?),
                    driver_phone = COALESCE(NULLIF(driver_phone, ''), ?)
                WHERE id = ? AND (max_weight = 3000 OR max_weight = 1800 OR max_weight = 4500 OR plate IS NULL OR plate = '')
                """,
                (sl, mw, tt, pl, cr, dn, dp, tid)
            )

        # Ensure existing demo pallets have lot/exp populated if null
        today = datetime.now()
        for row in conn.execute("SELECT code FROM pallets WHERE exp_date IS NULL OR exp_date = ''").fetchall():
            c_code = row[0]
            lot_code = f"LOT-2026-{abs(hash(c_code)) % 5 + 1:02}"
            mfg_d = (today - timedelta(days=60)).strftime("%Y-%m-%d")
            exp_d = (today + timedelta(days=35 if '01' in c_code else 120)).strftime("%Y-%m-%d")
            conn.execute("UPDATE pallets SET lot_number = ?, mfg_date = ?, exp_date = ? WHERE code = ?", (lot_code, mfg_d, exp_d, c_code))

        conn.commit()
    finally:
        conn.close()

    if scalar("SELECT COUNT(*) FROM products") == 0:
        reset_demo_data()


def reset_demo_data():
    with atomic() as conn:
        for t in ["events", "pallets", "trips", "locations", "products", "trucks"]:
            conn.execute(f"DELETE FROM {t}")

        products = [
            ("SKU-001", "จอมอนิเตอร์ 27นิ้ว", "อิเล็กทรอนิกส์", 10),
            ("SKU-002", "เครื่องชงกาแฟอัตโนมัติ", "เครื่องใช้ไฟฟ้า", 15),
            ("SKU-003", "น้ำดื่มคริสตัลแพ็ก", "เครื่องดื่ม", 12),
            ("SKU-004", "กล่องใส่ของเอนกประสงค์", "ของใช้ทั่วไป", 5),
            ("SKU-005", "ลำโพงบลูทูธพกพา", "อิเล็กทรอนิกส์", 4),
            ("SKU-006", "เครื่องฟอกอากาศ", "เครื่องใช้ไฟฟ้า", 14),
        ]
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

        trucks_data = [
            ("T01", "TRUCK 01", 8, 8000, "70-1122 กทม.", "SCG Logistics", "สมชาย ใจซื่อ", "081-234-5678", "6 ล้อตู้แห้ง"),
            ("T02", "TRUCK 02", 8, 8000, "70-3344 กทม.", "Flash Express", "สมเกียรติ มั่นคง", "089-876-5432", "6 ล้อตู้แห้ง"),
            ("T03", "TRUCK 03", 8, 8500, "71-5566 ชลบุรี", "Linfox Thailand", "อนุชา รวดเร็ว", "086-555-1234", "6 ล้อตู้ทึบ"),
            ("T04", "TRUCK 04", 8, 8000, "70-7788 กทม.", "DOCKFLOW Fleet", "วิชัย ส่งมอบ", "082-444-9988", "6 ล้อตู้แห้ง"),
            ("T05", "TRUCK 05", 8, 8500, "72-9900 นนทบุรี", "Kerry Logistics", "ธีระ ขยันยิ่ง", "085-111-2233", "6 ล้อตู้แห้ง"),
            ("T06", "TRUCK 06", 4, 3500, "1ฒผ-4455 กทม.", "DOCKFLOW Fleet", "บุญรอด ขับปลอดภัย", "083-777-6655", "4 ล้อตู้ทึบ"),
            ("T07", "TRUCK 07", 12, 15000, "73-1234 สระบุรี", "SCG Logistics", "ประสิทธิ์ ชำนาญทาง", "087-999-0011", "10 ล้อตู้ทึบ"),
            ("T08", "TRUCK 08", 8, 8000, "70-6677 ปทุมธานี", "รถร่วมบริการ", "มานะ ตรงเวลา", "084-222-3344", "6 ล้อตู้แห้ง"),
        ]
        conn.executemany("INSERT INTO trucks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", trucks_data)

        conn.executemany(
            "INSERT INTO locations VALUES (?, ?, ?)",
            [(f"{wh}-R{i:02}", wh, 4) for wh in WAREHOUSES for i in range(1, 7)],
        )

        ts = timestamp()
        trips = [
            ("IN-001", "IN", "T01", "WH-A", "NORTH SUPPLIER", "High", "DOCKED", "A-IN"),
            ("IN-002", "IN", "T02", "WH-B", "CITY FACTORY", "Normal", "DOCKED", "B-IN"),
            ("IN-003", "IN", "T03", "WH-A", "GREEN AGRO", "Low", "WAITING", None),
            ("OUT-001", "OUT", "T04", "WH-A", "สาขาสยามพารากอน", "High", "DOCKED", "A-OUT"),
            ("OUT-002", "OUT", "T05", "WH-B", "สาขารังสิต", "Normal", "WAITING", None),
        ]
        for tr in trips:
            conn.execute(
                """
                INSERT INTO trips(id, kind, truck_id, warehouse, partner, priority, status, dock, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tr + (ts,),
            )

        # Inventory with Lot, MFG, EXP
        today = datetime.now()
        for i in range(1, 25):
            p = products[(i - 1) % len(products)]
            wh = "WH-A" if i <= 12 else "WH-B"
            loc = f"{wh}-R{((i - 1) % 6) + 1:02}"
            lot_no = f"LOT-2026-{(i % 5) + 1:02}"
            mfg_d = (today - timedelta(days=90 + i * 4)).strftime("%Y-%m-%d")
            exp_offset = 25 if i in (1, 2, 13) else (50 if i in (3, 4, 14) else (110 + i * 8))
            exp_d = (today + timedelta(days=exp_offset)).strftime("%Y-%m-%d")
            conn.execute(
                """
                INSERT INTO pallets(code, sku, qty, weight, warehouse, location, status, lot_number, mfg_date, exp_date)
                VALUES (?, ?, ?, ?, ?, ?, 'AVAILABLE', ?, ?, ?)
                """,
                (f"PAL-{i:03}", p[0], 20, 20 * p[3], wh, loc, lot_no, mfg_d, exp_d),
            )

        # OUT-001 setup
        for i in range(1, 5):
            st_val = "LOADED" if i <= 2 else ("PICKED" if i == 3 else "RESERVED")
            conn.execute(
                """
                UPDATE pallets SET status = ?, outbound_id = 'OUT-001', load_slot = ?, location = CASE WHEN ? = 'RESERVED' THEN location ELSE NULL END
                WHERE code = ?
                """,
                (st_val, i if st_val == "LOADED" else None, st_val, f"PAL-{i:03}"),
            )

        for i in range(13, 16):
            conn.execute(
                "UPDATE pallets SET status = 'RESERVED', outbound_id = 'OUT-002' WHERE code = ?",
                (f"PAL-{i:03}",),
            )

        # Inbound setup
        batches = [("IN-001", "WH-A", "RCV-A", 6), ("IN-002", "WH-B", "RCV-B", 6)]
        for trip_id, wh, pfx, count in batches:
            for i in range(1, count + 1):
                p = products[(i - 1) % len(products)]
                stat = "RECEIVING" if trip_id == "IN-001" and i == 1 else "ON_INBOUND"
                lot_no = f"LOT-2026-IN{i:02}"
                mfg_d = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                exp_d = (today + timedelta(days=180 + i * 15)).strftime("%Y-%m-%d")
                conn.execute(
                    """
                    INSERT INTO pallets(code, sku, qty, weight, warehouse, status, inbound_id, inbound_slot, lot_number, mfg_date, exp_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (f"{pfx}-{i:02}", p[0], 20, 20 * p[3], wh, stat, trip_id, i, lot_no, mfg_d, exp_d),
                )

        log_event(conn, "SYSTEM", "INIT", "รีเซ็ตและสร้างข้อมูลจำลองระบบเรียบร้อย")


# ABC Velocity Classification & Slotting Matrix
ABC_CLASS = {
    "SKU-001": {"class": "A", "label": "Fast-Moving (หมุนเวียนเร็ว)", "color": "#059669", "badge_bg": "#ECFDF5", "opt_racks": ["R01", "R02"]},
    "SKU-003": {"class": "A", "label": "Fast-Moving (หมุนเวียนเร็ว)", "color": "#059669", "badge_bg": "#ECFDF5", "opt_racks": ["R01", "R02"]},
    "SKU-002": {"class": "B", "label": "Medium-Moving (หมุนเวียนปานกลาง)", "color": "#2563EB", "badge_bg": "#EFF6FF", "opt_racks": ["R03", "R04"]},
    "SKU-006": {"class": "B", "label": "Medium-Moving (หมุนเวียนปานกลาง)", "color": "#2563EB", "badge_bg": "#EFF6FF", "opt_racks": ["R03", "R04"]},
    "SKU-004": {"class": "C", "label": "Slow-Moving (หมุนเวียนช้า)", "color": "#64748B", "badge_bg": "#F8FAFC", "opt_racks": ["R05", "R06"]},
    "SKU-005": {"class": "C", "label": "Slow-Moving (หมุนเวียนช้า)", "color": "#64748B", "badge_bg": "#F8FAFC", "opt_racks": ["R05", "R06"]},
}


def get_days_until_expiry(exp_date_str):
    """คำนวณจำนวนวันที่เหลือก่อนถึงวันหมดอายุ"""
    if not exp_date_str:
        return 999
    try:
        exp_dt = datetime.strptime(str(exp_date_str).strip(), "%Y-%m-%d").date()
        today = datetime.now().date()
        return (exp_dt - today).days
    except Exception:
        return 999


def create_or_update_truck(truck_id, name, slots, max_weight, plate, carrier, driver_name, driver_phone, truck_type, is_edit=False):
    """ฟังก์ชันเพิ่มหรือแก้ไขข้อมูลยานพาหนะในระบบ"""
    with atomic() as conn:
        if is_edit:
            conn.execute(
                """
                UPDATE trucks 
                SET name = ?, slots = ?, max_weight = ?, plate = ?, carrier = ?, driver_name = ?, driver_phone = ?, truck_type = ?
                WHERE id = ?
                """,
                (name, slots, max_weight, plate, carrier, driver_name, driver_phone, truck_type, truck_id)
            )
            log_event(conn, "SYSTEM", truck_id, f"แก้ไขข้อมูลรถ {truck_id} ({name} · ทะเบียน {plate} · {carrier})")
        else:
            exist = conn.execute("SELECT 1 FROM trucks WHERE id = ?", (truck_id,)).fetchone()
            ensure(exist is None, f"รหัสรถ [{truck_id}] มีอยู่ในระบบแล้ว กรุณาใช้รหัสอื่น")
            conn.execute(
                """
                INSERT INTO trucks (id, name, slots, max_weight, plate, carrier, driver_name, driver_phone, truck_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (truck_id, name, slots, max_weight, plate, carrier, driver_name, driver_phone, truck_type)
            )
            log_event(conn, "SYSTEM", truck_id, f"ลงทะเบียนรถคันใหม่ {truck_id} ({name} · ทะเบียน {plate} · {carrier})")


def delete_truck_by_id(truck_id):
    """ฟังก์ชันลบรถออกจากระบบ (เฉพาะคันที่ไม่มีเที่ยวรถค้างอยู่)"""
    with atomic() as conn:
        active = conn.execute("SELECT id FROM trips WHERE truck_id = ? AND status NOT IN ('DEPARTED', 'CANCELLED')", (truck_id,)).fetchone()
        ensure(active is None, f"ไม่สามารถลบรถ {truck_id} ได้ เนื่องจากกำลังปฏิบัติงานในเที่ยวรถ {active['id']}")
        conn.execute("DELETE FROM trucks WHERE id = ?", (truck_id,))
        log_event(conn, "SYSTEM", truck_id, f"ลบรถ {truck_id} ออกจากระบบเรียบร้อย")


# =========================================================
# 4. MATPLOTLIB VISUAL ENGINE
# =========================================================

def draw_truck_graphic(trip_id, title_label="TRUCK VIEW"):
    """วาดภาพรถบรรทุกและพาเลตบนรถแบบมองจากด้านบนด้วย Matplotlib รองรับภาษาไทย"""
    trip = rows("SELECT s.*, t.name, t.slots, t.max_weight FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.id = ?", (trip_id,))
    if not trip:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.text(0.5, 0.5, "ไม่พบข้อมูลรถ", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    t_data = trip[0]
    inbound = t_data["kind"] == "IN"
    rel = "inbound_id" if inbound else "outbound_id"
    target_status = "ON_INBOUND" if inbound else "LOADED"
    slot_key = "inbound_slot" if inbound else "load_slot"

    pallets = rows(
        f"""
        SELECT p.*, pr.category FROM pallets p 
        JOIN products pr ON pr.sku = p.sku 
        WHERE p.{rel} = ? AND p.status = ?
        """,
        (trip_id, target_status),
    )
    slot_map = {item[slot_key]: item for item in pallets if item.get(slot_key)}

    fig, ax = plt.subplots(figsize=(5, 7), dpi=140)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Header title
    ax.text(3, 9.4, f"{t_data['name']} ({t_data['id']})", ha="center", va="center", fontsize=12, fontweight="bold", color="#1E293B")
    ax.text(3, 9.0, f"ประเภทงาน: {'รับสินค้าเข้า' if inbound else 'ส่งสินค้าออก'} | คู่ค้า: {t_data['partner']}", ha="center", va="center", fontsize=9, color="#64748B")

    # Truck Body Container
    ax.add_patch(Rectangle((1.0, 1.8), 4.0, 6.5, facecolor="#E2E8F0", edgecolor="#64748B", linewidth=2, linestyle="--"))
    ax.text(3, 8.0, "-- ประตูท้ายรถ (REAR GATE) --", ha="center", va="center", fontsize=8, color="#475569", fontweight="bold")

    # Cabin (Head)
    ax.add_patch(FancyBboxPatch((1.2, 0.4), 3.6, 1.2, boxstyle="round,pad=0.1", facecolor="#4F46E5", edgecolor="#3730A3"))
    ax.text(3, 1.0, "หัวรถบรรทุก (CABIN)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(3, 0.65, f"พิกัดรับน้ำหนัก: {t_data.get('max_weight', 8000):,.0f} kg", ha="center", va="center", fontsize=7.5, color="#E0E7FF", fontweight="bold")

    # Wheels
    for wx, wy in [(0.8, 1.0), (4.8, 1.0), (0.8, 6.5), (4.8, 6.5)]:
        ax.add_patch(Rectangle((wx, wy), 0.4, 1.0, facecolor="#334155", edgecolor="#0F172A"))

    # Slots and Pallets grid inside truck (Dynamic scaling for any truck size: 4, 6, 8, 12+ slots)
    slots_total = max(1, t_data.get("slots", 8))
    row_count = max(1, (slots_total + 1) // 2)
    total_avail_h = 5.7
    slot_h = min(0.95, (total_avail_h - (row_count - 1) * 0.12) / row_count)
    row_pitch = slot_h + 0.12
    start_sy = 7.7 - slot_h
    font_main = 8.5 if row_count <= 4 else (7.5 if row_count <= 6 else 6.5)
    font_sub = 7.0 if row_count <= 4 else (6.0 if row_count <= 6 else 5.5)

    for r in range(row_count):
        for c in range(2):
            slot_idx = r * 2 + c + 1
            if slot_idx > slots_total:
                continue

            sx = 1.35 + c * 1.75
            sy = start_sy - (r * row_pitch)

            item = slot_map.get(slot_idx)
            if item:
                color = CATEGORY_COLOR.get(item["category"], "#0F766E")
                ax.add_patch(FancyBboxPatch((sx, sy), 1.5, slot_h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#0F172A", linewidth=1.5))
                ax.text(sx + 0.75, sy + slot_h * 0.60, item["code"], ha="center", va="center", fontsize=font_main, color="white", fontweight="bold")
                ax.text(sx + 0.75, sy + slot_h * 0.25, f"{item['qty']} หน่วย", ha="center", va="center", fontsize=font_sub, color="#E2E8F0")
            else:
                ax.add_patch(FancyBboxPatch((sx, sy), 1.5, slot_h, boxstyle="round,pad=0.04", facecolor="#FFFFFF", edgecolor="#CBD5E1", linestyle=":"))
                ax.text(sx + 0.75, sy + slot_h * 0.50, f"ช่อง {slot_idx:02}", ha="center", va="center", fontsize=font_main, color="#94A3B8")

    fig.tight_layout(pad=0.5)
    return fig


def draw_yard_overview_html():
    """สร้างภาพรวมลานจอดรถทั้ง 4 ช่อง ด้วย Pure HTML/CSS (คมชัดทุกความละเอียด ไม่มีภาพเบลอ)"""
    docked_trips = {
        r["dock"]: r
        for r in rows(
            """
            SELECT s.*, t.name AS truck_name, t.slots, t.max_weight 
            FROM trips s JOIN trucks t ON t.id = s.truck_id 
            WHERE s.status = 'DOCKED'
            """
        )
    }

    for d_id, trip in docked_trips.items():
        if trip["kind"] == "IN":
            rem = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (trip["id"],))
            recv = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'RECEIVING'", (trip["id"],))
            total = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ?", (trip["id"],))
            trip["stat_line"] = f"บนรถ: {rem} | พักรับ: {recv} (รวม {total} พาเลต)"
            trip["progress"] = (total - rem) / max(total, 1)
            trip["stage_text"] = "กำลังขนถ่ายสินค้าลง" if rem > 0 else "ขนลงครบแล้ว (พร้อมออก)"
        else:
            loaded = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (trip["id"],))
            total = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ?", (trip["id"],))
            trip["stat_line"] = f"โหลดแล้ว {loaded}/{total} พาเลต"
            trip["progress"] = loaded / max(total, 1)
            trip["stage_text"] = "กำลังโหลดสินค้าขึ้นรถ" if loaded < total else "โหลดครบแล้ว (พร้อมออก)"

    docks_layout = [
        ("A-IN",  "A1", "ขาเข้ารับสินค้า", "WH-A", "IN"),
        ("A-OUT", "A2", "ขาออกจ่ายสินค้า", "WH-A", "OUT"),
        ("B-IN",  "B1", "ขาเข้ารับสินค้า", "WH-B", "IN"),
        ("B-OUT", "B2", "ขาออกจ่ายสินค้า", "WH-B", "OUT"),
    ]

    cards_html = ""
    for d_id, code, kind_name, wh, kind in docks_layout:
        trip = docked_trips.get(d_id)
        if trip:
            p_name = trip["partner"]
            if len(p_name) > 20:
                p_name = p_name[:18] + "..."
            pct = int(trip["progress"] * 100)
            bar_color = "#10B981" if trip["progress"] >= 1.0 else "#4F46E5"
            stage_color = "#059669" if trip["progress"] >= 1.0 else "#4338CA"

            cards_html += f"""
            <div style="background:#FFFFFF; border:2.5px solid #10B981; border-radius:14px; padding:14px 16px; box-shadow:0 4px 18px rgba(16,185,129,0.12); display:flex; flex-direction:column; justify-content:space-between; min-height:360px;">
                <div>
                    <div style="background:#1E293B; color:#FFF; border-radius:9px; padding:9px 14px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-size:16px; font-weight:900; letter-spacing:1px;">DOCK {code}</span>
                        <span style="font-size:12px; font-weight:700; color:#94A3B8;">{wh} · {kind}</span>
                    </div>
                    <div style="text-align:center; font-size:13px; font-weight:700; color:#64748B; margin-bottom:8px;">{kind_name} ({wh})</div>
                    <div style="background:#D1FAE5; border:1.5px solid #059669; color:#065F46; border-radius:22px; padding:6px 14px; font-size:12.5px; font-weight:800; text-align:center; margin-bottom:14px;">
                        ● ปฏิบัติงาน (DOCKED)
                    </div>
                </div>
                <div style="background:#EEF2FF; border:2px solid #C7D2FE; border-radius:12px; padding:14px 16px; text-align:center; flex:1; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:19px; font-weight:900; color:#312E81; margin-bottom:4px;">{trip['truck_name']}</div>
                    <div style="font-size:13px; font-weight:700; color:#4338CA; margin-bottom:8px;">รหัสเที่ยว: {trip['id']} ({trip['kind']})</div>
                    <div style="font-size:12.5px; color:#475569; margin-bottom:8px;">คู่ค้า: <strong>{p_name}</strong></div>
                    <div style="font-size:13px; font-weight:800; color:#1E293B; margin-bottom:12px;">{trip['stat_line']}</div>
                    <div style="background:#E2E8F0; border-radius:12px; height:20px; overflow:hidden; position:relative; margin-bottom:10px; border:1px solid #CBD5E1;">
                        <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:10px; transition:width 0.3s ease;"></div>
                        <span style="position:absolute; width:100%; text-align:center; top:0; left:0; font-size:11px; font-weight:900; line-height:20px; color:{'#FFF' if pct > 35 else '#1E293B'};">{pct}%</span>
                    </div>
                    <div style="font-size:13px; font-weight:900; color:{stage_color};">{trip['stage_text']}</div>
                </div>
            </div>
            """
        else:
            cards_html += f"""
            <div style="background:#F8FAFC; border:2.5px dashed #CBD5E1; border-radius:14px; padding:14px 16px; display:flex; flex-direction:column; justify-content:space-between; min-height:360px;">
                <div>
                    <div style="background:#475569; color:#FFF; border-radius:9px; padding:9px 14px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-size:16px; font-weight:900; letter-spacing:1px;">DOCK {code}</span>
                        <span style="font-size:12px; font-weight:700; color:#CBD5E1;">{wh} · {kind}</span>
                    </div>
                    <div style="text-align:center; font-size:13px; font-weight:700; color:#64748B; margin-bottom:8px;">{kind_name} ({wh})</div>
                    <div style="background:#F1F5F9; border:1.5px solid #CBD5E1; color:#64748B; border-radius:22px; padding:6px 14px; font-size:12.5px; font-weight:800; text-align:center; margin-bottom:14px;">
                        ○ ช่องจอดว่าง (VACANT)
                    </div>
                </div>
                <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <div style="width:82px; height:82px; border:3px solid #94A3B8; border-radius:50%; background:#FFF; display:flex; align-items:center; justify-content:center; margin-bottom:18px; box-shadow:0 3px 8px rgba(0,0,0,0.05);">
                        <span style="font-size:36px; font-weight:900; color:#475569; line-height:1;">P</span>
                    </div>
                    <div style="font-size:15px; font-weight:800; color:#475569; margin-bottom:5px;">พร้อมรับรถเข้าเทียบ</div>
                    <div style="font-size:12.5px; color:#94A3B8;">รองรับ: รถ{kind_name}</div>
                </div>
            </div>
            """

    full_html = textwrap.dedent(f"""
    <div style="background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:16px; padding:24px 28px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; box-shadow:0 3px 12px rgba(0,0,0,0.04); margin-bottom:14px;">
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:21px; font-weight:900; color:#0F172A; letter-spacing:0.5px; margin-bottom:5px;">ศูนย์ควบคุมลานจอดรถเทียบท่า (DOCK YARD LIVE COMMAND CENTER)</div>
            <div style="font-size:13px; color:#64748B; font-weight:500;">ระบบติดตามสถานะช่องเทียบแบบเรียลไทม์ · คลังสินค้า WH-A และ WH-B</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:18px;">
            {cards_html}
        </div>
        <div style="border-top:2.5px dashed #CBD5E1; padding-top:12px; text-align:center;">
            <span style="font-size:12px; font-weight:900; color:#94A3B8; letter-spacing:1.5px;">═ เลนถนนทางวิ่งลานจอดรถบรรทุก (TRUCK YARD ACCESS LANE) ═</span>
        </div>
    </div>
    """)
    return full_html


# =========================================================
# 5. BUSINESS LOGIC & EXECUTION ENGINE
# =========================================================

def execute_action(action, ref, val=None):
    """เครื่องยนต์ประมวลผลคำสั่งระบบ WMS พร้อมระบบความปลอดภัยและการบันทึก Log"""
    with atomic() as conn:
        if action == "park":
            trip = fetch_trip(conn, ref)
            ensure(trip["status"] == "WAITING", "รถคันนี้ไม่ได้อยู่ในคิวรอ")
            ensure(val in DOCKS, "ช่องจอดไม่ถูกต้อง")
            dock_cfg = DOCKS[val]
            ensure(dock_cfg["kind"] == trip["kind"], "ประเภทช่องจอดไม่ตรงกับงานรถ")
            ensure(dock_cfg["warehouse"] == trip["warehouse"], "ช่องจอดอยู่คนละคลังสินค้า")

            occupied = conn.execute("SELECT 1 FROM trips WHERE status = 'DOCKED' AND dock = ?", (val,)).fetchone()
            ensure(occupied is None, "ช่องจอดนี้มีรถจอดปฏิบัติงานอยู่แล้ว")

            conn.execute("UPDATE trips SET status = 'DOCKED', dock = ? WHERE id = ?", (val, ref))
            log_event(conn, "PARK", ref, f"นำรถ {trip['truck_name']} เข้าจอดที่ช่อง {val}")
            return

        if action == "cancel_trip":
            trip = fetch_trip(conn, ref)
            ensure(trip["status"] == "WAITING", "สามารถยกเลิกได้เฉพาะเที่ยวรถที่รอเข้าจอดเท่านั้น")
            if trip["kind"] == "OUT":
                conn.execute("UPDATE pallets SET status = 'AVAILABLE', outbound_id = NULL WHERE outbound_id = ?", (ref,))
            else:
                conn.execute("DELETE FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (ref,))
            conn.execute("DELETE FROM trips WHERE id = ?", (ref,))
            log_event(conn, "CANCEL", ref, f"ยกเลิกเที่ยวรถ {ref} และคืนสถานะสินค้าเรียบร้อย")
            return

        if action == "depart":
            trip = fetch_trip(conn, ref)
            ensure(trip["status"] == "DOCKED", "รถไม่ได้จอดอยู่ในช่องจอด")

            if trip["kind"] == "IN":
                rem = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (ref,), conn=conn)
                ensure(rem == 0, f"ยังมีสินค้าค้างอยู่บนรถ {rem} รายการ กรุณาขนลงให้หมดก่อน")
            else:
                cargos = rows("SELECT status FROM pallets WHERE outbound_id = ?", (ref,), conn=conn)
                ensure(len(cargos) > 0, "ใบจ่ายนี้ไม่มีสินค้า")
                ensure(all(c.get("status") == "LOADED" for c in cargos), "ต้องโหลดสินค้าขึ้นรถให้ครบทุกรายการก่อนรถออก")
                conn.execute("UPDATE pallets SET status = 'SHIPPED' WHERE outbound_id = ?", (ref,))

            conn.execute("UPDATE trips SET status = 'DEPARTED', dock = NULL, departed_at = ? WHERE id = ?", (timestamp(), ref))
            log_event(conn, "DEPART", ref, f"รถ {trip['truck_name']} ออกจากคลังสำเร็จ")
            return

        # Pallet-oriented operations
        if action == "receive_all":
            trip = fetch_trip(conn, ref)
            ensure(trip["kind"] == "IN" and trip["status"] == "DOCKED", "ต้องนำรถขาเข้าเข้าจอดก่อนขนลง")
            pallets_to_recv = rows("SELECT code FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (ref,), conn=conn)
            ensure(pallets_to_recv, "ไม่มีสินค้าค้างอยู่บนรถ")
            conn.execute("UPDATE pallets SET status = 'RECEIVING' WHERE inbound_id = ? AND status = 'ON_INBOUND'", (ref,))
            log_event(conn, "RECEIVE", ref, f"ขนสินค้าลงจากรถ {ref} ทั้งหมด ({len(pallets_to_recv)} พาเลต) สู่พื้นที่พักรับ")
            return

        if action == "putaway_all":
            warehouse = ref
            stage_items = rows("SELECT code FROM pallets WHERE warehouse = ? AND status = 'RECEIVING' ORDER BY code ASC", (warehouse,), conn=conn)
            ensure(stage_items, "ไม่มีสินค้าค้างในพื้นที่พักรับ")
            locs = rows(
                """
                SELECT l.id, l.capacity, COUNT(p.code) as used 
                FROM locations l LEFT JOIN pallets p ON p.location = l.id AND p.status IN ('AVAILABLE', 'RESERVED')
                WHERE l.warehouse = ? GROUP BY l.id, l.capacity HAVING used < capacity ORDER BY l.id ASC
                """,
                (warehouse,),
                conn=conn,
            )
            ensure(locs, "ช่องจัดเก็บในคลังนี้เต็มทั้งหมดแล้ว")
            assigned_count = 0
            cur_loc_idx = 0
            loc_caps = {l["id"]: l["capacity"] - l["used"] for l in locs}
            avail_loc_ids = [l["id"] for l in locs]

            for item in stage_items:
                while cur_loc_idx < len(avail_loc_ids) and loc_caps[avail_loc_ids[cur_loc_idx]] <= 0:
                    cur_loc_idx += 1
                if cur_loc_idx >= len(avail_loc_ids):
                    break
                target_loc = avail_loc_ids[cur_loc_idx]
                conn.execute("UPDATE pallets SET status = 'AVAILABLE', location = ? WHERE code = ?", (target_loc, item["code"]))
                loc_caps[target_loc] -= 1
                assigned_count += 1

            ensure(assigned_count > 0, "ไม่สามารถจัดเก็บได้ พื้นที่จัดเก็บไม่เพียงพอ")
            log_event(conn, "PUTAWAY", warehouse, f"จัดเก็บสินค้าอัตโนมัติสำเร็จจำนวน {assigned_count} พาเลต")
            return

        if action == "pick_all":
            trip = fetch_trip(conn, ref)
            ensure(trip["kind"] == "OUT", "คำสั่งนี้ใช้ได้เฉพาะรถขาออก")
            items = rows("SELECT code FROM pallets WHERE outbound_id = ? AND status = 'RESERVED'", (ref,), conn=conn)
            ensure(items, "ไม่มีสินค้าในสถานะรอหยิบ")
            conn.execute("UPDATE pallets SET status = 'PICKED', location = NULL WHERE outbound_id = ? AND status = 'RESERVED'", (ref,))
            log_event(conn, "PICK", ref, f"หยิบสินค้าทั้งหมด ({len(items)} พาเลต) เข้าสู่พื้นที่พักจ่าย")
            return

        if action == "load_all":
            trip = fetch_trip(conn, ref)
            ensure(trip["kind"] == "OUT" and trip["status"] == "DOCKED", "ต้องนำรถขาออกเข้าจอดก่อนจึงจะโหลดสินค้าได้")
            items = rows("SELECT code, weight FROM pallets WHERE outbound_id = ? AND status = 'PICKED'", (ref,), conn=conn)
            ensure(items, "ไม่มีสินค้าในพื้นที่พักจ่ายที่พร้อมโหลด")
            loaded = rows("SELECT load_slot, weight FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (ref,), conn=conn)
            occupied = {it["load_slot"] for it in loaded}
            free_slots = [s for s in range(1, trip["slots"] + 1) if s not in occupied]
            cur_weight = sum(it["weight"] for it in loaded)

            loaded_cnt = 0
            for idx, item in enumerate(items):
                if idx >= len(free_slots):
                    break
                if cur_weight + item["weight"] > trip["max_weight"]:
                    break
                slot = free_slots[idx]
                conn.execute("UPDATE pallets SET status = 'LOADED', load_slot = ? WHERE code = ?", (slot, item["code"]))
                cur_weight += item["weight"]
                loaded_cnt += 1

            ensure(loaded_cnt > 0, "ไม่สามารถโหลดได้ (ช่องบรรทุกเต็มหรือน้ำหนักเกินพิกัดรถ)")
            log_event(conn, "LOAD", ref, f"โหลดสินค้าขึ้นรถ {trip['truck_name']} จำนวน {loaded_cnt} พาเลต")
            return

        # Single pallet actions
        pallet = fetch_pallet(conn, ref)
        code = pallet["code"]

        if action == "receive":
            trip = fetch_trip(conn, val)
            ensure(trip["kind"] == "IN" and trip["status"] == "DOCKED", "ต้องนำรถขาเข้าเข้าจอดก่อนขนลง")
            ensure(pallet["inbound_id"] == val, "พาเลตนี้ไม่ได้อยู่ในเที่ยวรถนี้")
            ensure(pallet["status"] == "ON_INBOUND", "พาเลตนี้ถูกขนลงไปแล้ว")

            conn.execute("UPDATE pallets SET status = 'RECEIVING' WHERE code = ?", (code,))
            log_event(conn, "RECEIVE", code, f"ขนลงจาก {val} สู่พื้นที่พักรับคลัง {pallet['warehouse']}")
            return

        if action == "putaway":
            ensure(pallet["status"] == "RECEIVING", "สินค้าไม่ได้อยู่ในพื้นที่พักรับ")
            loc = conn.execute("SELECT * FROM locations WHERE id = ?", (val,)).fetchone()
            ensure(loc is not None, "ไม่พบตำแหน่งจัดเก็บนี้")
            ensure(loc["warehouse"] == pallet["warehouse"], "ตำแหน่งจัดเก็บอยู่คนละคลังสินค้า")

            used = scalar("SELECT COUNT(*) FROM pallets WHERE location = ? AND status IN ('AVAILABLE', 'RESERVED')", (val,), conn=conn)
            ensure(used < loc["capacity"], "ตำแหน่งจัดเก็บนี้เต็มแล้ว")

            conn.execute("UPDATE pallets SET status = 'AVAILABLE', location = ? WHERE code = ?", (val, code))
            log_event(conn, "PUTAWAY", code, f"จัดเก็บเข้าชั้นวาง {val} สำเร็จ")
            return

        if action == "relocate":
            loc = conn.execute("SELECT * FROM locations WHERE id = ?", (val,)).fetchone()
            ensure(loc is not None, "ไม่พบตำแหน่งจัดเก็บปลายทาง")
            ensure(loc["warehouse"] == pallet["warehouse"], "ตำแหน่งจัดเก็บปลายทางอยู่คนละคลังสินค้า")
            used = scalar("SELECT COUNT(*) FROM pallets WHERE location = ? AND status IN ('AVAILABLE', 'RESERVED')", (val,), conn=conn)
            ensure(used < loc["capacity"], f"ตำแหน่งชั้น {val} เต็มแล้ว (จุได้สูงสุด {loc['capacity']} พาเลต)")
            old_loc = pallet.get("location") or "ไม่ระบุ"
            conn.execute("UPDATE pallets SET location = ? WHERE code = ?", (val, code))
            log_event(conn, "PUTAWAY", code, f"ย้ายพาเลต {code} จากชั้น {old_loc} ➔ ไปยัง {val} ตามหลัก ABC Slotting")
            return

        trip = fetch_trip(conn, val)
        ensure(trip["kind"] == "OUT" and trip["status"] != "DEPARTED", "ใบจ่ายสินค้าไม่ถูกต้อง")
        ensure(pallet["outbound_id"] == val, "พาเลตนี้ไม่ได้ผูกกับใบจ่ายของรถคันนี้")

        if action == "pick":
            ensure(pallet["status"] == "RESERVED", "พาเลตไม่ได้อยู่ในสถานะรอหยิบ")
            old_loc = pallet["location"]
            conn.execute("UPDATE pallets SET status = 'PICKED', location = NULL WHERE code = ?", (code,))
            log_event(conn, "PICK", code, f"หยิบสินค้าจาก {old_loc} ไปพื้นที่พักจ่าย")
            return

        ensure(trip["status"] == "DOCKED", "ต้องนำรถขาออกเข้าจอดก่อนจึงจะโหลดสินค้าได้")

        if action == "unload":
            ensure(pallet["status"] == "LOADED", "พาเลตไม่ได้อยู่บนรถ")
            conn.execute("UPDATE pallets SET status = 'PICKED', load_slot = NULL WHERE code = ?", (code,))
            log_event(conn, "UNLOAD", code, f"นำพาเลต {code} ลงจากรถกลับสู่พื้นที่พักจ่าย (ยังจองใบเดิม)")
            return

        ensure(pallet["status"] == "PICKED", "ต้องหยิบสินค้ามาก่อนโหลดขึ้นรถ")
        loaded_items = rows("SELECT load_slot, weight FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (val,), conn=conn)
        occupied = {it["load_slot"] for it in loaded_items}
        free_slots = [s for s in range(1, trip["slots"] + 1) if s not in occupied]

        ensure(free_slots, "พื้นที่ช่องบรรทุกบนรถเต็มแล้ว")
        current_weight = sum(it["weight"] for it in loaded_items)
        ensure(current_weight + pallet["weight"] <= trip["max_weight"], "น้ำหนักรวมเกินพิกัดบรรทุกของรถ")

        slot = free_slots[0]
        conn.execute("UPDATE pallets SET status = 'LOADED', load_slot = ? WHERE code = ?", (slot, code))
        log_event(conn, "LOAD", code, f"โหลดขึ้นรถ {trip['truck_name']} ช่องที่ {slot}")



def decode_barcode_image(image_bytes):
    """ถอดรหัสบาร์โค้ดและคิวอาร์โค้ดจากรูปภาพด้วย OpenCV จริง"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # 1. Barcode detector
        detector = cv2.barcode_BarcodeDetector()
        ok, decoded_info, _, _ = detector.detectAndDecode(img)
        if ok and decoded_info:
            for c in decoded_info:
                if c and c.strip():
                    return c.strip()
        # 2. QR code detector fallback
        qr = cv2.QRCodeDetector()
        data, _, _ = qr.detectAndDecode(img)
        if data and data.strip():
            return data.strip()
    except Exception as e:
        pass
    return None


def execute_barcode_scan(raw_code, action_mode="AUTO"):
    """
    ประมวลผลการสแกนบาร์โค้ดจริง และบันทึกลงฐานข้อมูล SQLite แบบเรียลไทม์
    รองรับทั้งยิงสแกนเนอร์ USB, พิมพ์รหัส, และถ่ายภาพด้วยกล้อง
    """
    code = raw_code.strip().upper()
    if not code:
        raise ValueError("กรุณาระบุหรือสแกนรหัสบาร์โค้ด")

    with atomic() as conn:
        pallet = conn.execute("SELECT * FROM pallets WHERE code = ?", (code,)).fetchone()
        ensure(pallet is not None, f"ไม่พบบาร์โค้ดพาเลต [{code}] ในระบบคลังสินค้า")
        p = dict(pallet)
        prev_status = p["status"]
        result_msg = ""
        action_name = ""

        if action_mode == "LOOKUP":
            return {
                "code": code,
                "pallet": p,
                "prev_status": prev_status,
                "new_status": prev_status,
                "action": "LOOKUP",
                "message": f"สแกนตรวจสอบข้อมูลสำเร็จ: {code} ({PALLET_STATUS.get(prev_status, prev_status)})"
            }

        # AUTO MODE or ACTION SPECIFIC
        if action_mode in ("AUTO", "RECEIVE") and p["status"] == "ON_INBOUND":
            trip = conn.execute("SELECT * FROM trips WHERE id = ?", (p["inbound_id"],)).fetchone()
            ensure(trip is not None and trip["status"] == "DOCKED", f"รถขาเข้า {p['inbound_id']} ยังไม่เข้าจอดเทียบท่า ไม่สามารถขนถ่ายลงได้")
            conn.execute("UPDATE pallets SET status = 'RECEIVING' WHERE code = ?", (code,))
            action_name = "SCAN_RECEIVE"
            result_msg = f"สแกนขนถ่ายลงสำเร็จ: {code} ➔ เข้าสู่พื้นที่พักรับ (คลัง {p['warehouse']})"

        elif action_mode in ("AUTO", "PUTAWAY") and p["status"] == "RECEIVING":
            # Auto-find best rack location with capacity
            loc = conn.execute(
                """SELECT l.id, l.capacity, COUNT(p2.code) AS used
                FROM locations l LEFT JOIN pallets p2 ON p2.location = l.id AND p2.status IN ('AVAILABLE', 'RESERVED')
                WHERE l.warehouse = ? GROUP BY l.id HAVING used < capacity ORDER BY used ASC, l.id ASC LIMIT 1""",
                (p["warehouse"],)
            ).fetchone()
            ensure(loc is not None, f"ช่องจัดเก็บในคลัง {p['warehouse']} เต็มทั้งหมดแล้ว")
            target_loc = loc["id"]
            conn.execute("UPDATE pallets SET status = 'AVAILABLE', location = ? WHERE code = ?", (target_loc, code))
            action_name = "SCAN_PUTAWAY"
            result_msg = f"สแกนจัดเก็บสำเร็จ: {code} ➔ จัดเก็บบนชั้นวาง [{target_loc}] (พร้อมใช้งาน)"

        elif action_mode in ("AUTO", "PICK") and p["status"] == "RESERVED":
            old_loc = p["location"]
            conn.execute("UPDATE pallets SET status = 'PICKED', location = NULL WHERE code = ?", (code,))
            action_name = "SCAN_PICK"
            result_msg = f"สแกนหยิบสำเร็จ: {code} (จากชั้น {old_loc}) ➔ เข้าสู่พื้นที่พักจ่ายสินค้า"

        elif action_mode in ("AUTO", "LOAD") and p["status"] == "PICKED":
            trip = conn.execute("SELECT s.*, t.name AS truck_name, t.slots, t.max_weight FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.id = ?", (p["outbound_id"],)).fetchone()
            ensure(trip is not None and trip["status"] == "DOCKED", f"รถขาออก {p['outbound_id']} ยังไม่เทียบท่า กรุณานำรถเข้าจอดก่อนโหลด")
            loaded = conn.execute("SELECT load_slot, weight FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (p["outbound_id"],)).fetchall()
            occupied = {it["load_slot"] for it in loaded}
            free_slots = [s for s in range(1, trip["slots"] + 1) if s not in occupied]
            ensure(bool(free_slots), "ช่องบรรทุกบนรถคันนี้เต็มแล้ว")
            cur_wt = sum(it["weight"] for it in loaded)
            ensure(cur_wt + p["weight"] <= trip["max_weight"], "น้ำหนักรวมจะเกินพิกัดบรรทุกของรถ")
            slot = free_slots[0]
            conn.execute("UPDATE pallets SET status = 'LOADED', load_slot = ? WHERE code = ?", (slot, code))
            action_name = "SCAN_LOAD"
            result_msg = f"สแกนโหลดขึ้นรถสำเร็จ: {code} ➔ บรรจุบนรถ {trip['truck_name']} (ช่องที่ {slot})"

        elif p["status"] == "AVAILABLE":
            action_name = "SCAN_AUDIT"
            result_msg = f"บาร์โค้ด {code} อยู่ในสถานะพร้อมใช้บนชั้นวาง [{p['location']}] (คลัง {p['warehouse']})"

        elif p["status"] == "LOADED":
            action_name = "SCAN_AUDIT"
            result_msg = f"บาร์โค้ด {code} บรรจุอยู่บนรถขาออก [{p['outbound_id']}] ช่องที่ {p['load_slot']}"

        elif p["status"] == "SHIPPED":
            action_name = "SCAN_AUDIT"
            result_msg = f"บาร์โค้ด {code} ถูกส่งมอบออกจากคลังแล้ว (เที่ยว {p['outbound_id']})"

        else:
            raise ValueError(f"พาเลต {code} อยู่ในสถานะ {PALLET_STATUS.get(p['status'], p['status'])} ไม่ตรงกับเงื่อนไขคำสั่ง {action_mode}")

        log_event(conn, action_name, code, result_msg)
        updated_p = conn.execute("SELECT * FROM pallets WHERE code = ?", (code,)).fetchone()

        return {
            "code": code,
            "pallet": dict(updated_p),
            "prev_status": prev_status,
            "new_status": updated_p["status"],
            "action": action_name,
            "message": result_msg
        }


def create_new_trip(kind, truck_id, warehouse, partner, priority, payload):
    partner_clean = partner.strip() if partner else ""
    ensure(bool(partner_clean), "กรุณากรอกชื่อคู่ค้าหรือสาขาปลายทาง")
    trip_id = make_id(kind)

    with atomic() as conn:
        truck = conn.execute("SELECT * FROM trucks WHERE id = ?", (truck_id,)).fetchone()
        ensure(truck is not None, "ไม่พบรถบรรทุก")
        busy = conn.execute("SELECT 1 FROM trips WHERE truck_id = ? AND status NOT IN ('DEPARTED', 'CANCELLED')", (truck_id,)).fetchone()
        ensure(busy is None, "รถคันนี้กำลังปฏิบัติงานเที่ยวอื่นอยู่")

        if kind == "IN":
            sku, qty, count = payload
            prod = conn.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
            ensure(prod is not None, "ไม่พบรหัสสินค้า")
            total_weight = qty * count * prod["unit_weight"]
        else:
            codes = list(dict.fromkeys(payload))
            count = len(codes)
            ensure(count > 0, "กรุณาเลือกพาเลตอย่างน้อย 1 รายการ")
            total_weight = 0
            for c in codes:
                pal = fetch_pallet(conn, c)
                ensure(pal["status"] == "AVAILABLE", f"พาเลต {c} ไม่พร้อมใช้งานหรือถูกจองไปแล้ว")
                ensure(pal["warehouse"] == warehouse, f"พาเลต {c} อยู่คนละคลังสินค้า")
                total_weight += pal["weight"]

        ensure(count <= truck["slots"], f"จำนวน {count} พาเลต เกินช่องบรรทุกของรถ ({truck['slots']} ช่อง)")
        ensure(total_weight <= truck["max_weight"], f"น้ำหนักรวม {total_weight:,.0f} kg เกินพิกัดรถ ({truck['max_weight']:,.0f} kg)")

        conn.execute(
            """
            INSERT INTO trips(id, kind, truck_id, warehouse, partner, priority, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'WAITING', ?)
            """,
            (trip_id, kind, truck_id, warehouse, partner_clean, priority, timestamp()),
        )

        if kind == "IN":
            today_str = datetime.now().strftime("%Y-%m-%d")
            exp_str = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")
            lot_str = f"LOT-{datetime.now().strftime('%Y%m')}-{trip_id[-3:]}"
            for s in range(1, count + 1):
                conn.execute(
                    """
                    INSERT INTO pallets(code, sku, qty, weight, warehouse, status, inbound_id, inbound_slot, lot_number, mfg_date, exp_date)
                    VALUES (?, ?, ?, ?, ?, 'ON_INBOUND', ?, ?, ?, ?, ?)
                    """,
                    (make_id("RCV"), sku, qty, qty * prod["unit_weight"], warehouse, trip_id, s, lot_str, today_str, exp_str),
                )
        else:
            for c in codes:
                conn.execute("UPDATE pallets SET status = 'RESERVED', outbound_id = ? WHERE code = ?", (trip_id, c))

        log_event(conn, "CREATE", trip_id, f"สร้างเที่ยวรถ {kind} ({trip_id}) สำเร็จ · {count} พาเลต · รวม {total_weight:,.0f} kg")
    return trip_id


# =========================================================
# 6. STREAMLIT UI COMPONENTS & PAGES
# =========================================================

def show_pallet_table(items):
    """ฟังก์ชันแสดงผลรายการพาเลตแบบมาตรฐาน รองรับ fallback ปลอดภัย พร้อมข้อมูลล็อต วันหมดอายุ และ ABC Class"""
    if not items:
        st.info("ไม่มีข้อมูลรายการพาเลต")
        return
    display = []
    for it in items:
        pos = it.get("location") or (
            "บนรถขาเข้า" if it["status"] == "ON_INBOUND" else
            "พื้นที่พักรับ" if it["status"] == "RECEIVING" else
            "พื้นที่พักจ่าย" if it["status"] == "PICKED" else
            f"บนรถออก (ช่อง {it.get('load_slot', '-')})" if it["status"] == "LOADED" else
            "ส่งออกแล้ว" if it["status"] == "SHIPPED" else "—"
        )
        abc_c = ABC_CLASS.get(it.get("sku"), {}).get("class", "-")
        days_left = get_days_until_expiry(it.get("exp_date"))
        if it.get("exp_date"):
            if days_left < 30:
                exp_disp = f"⚠️ {it['exp_date']} ({days_left} วัน)"
            elif days_left < 60:
                exp_disp = f"⏳ {it['exp_date']} ({days_left} วัน)"
            else:
                exp_disp = f"{it['exp_date']} ({days_left} วัน)"
        else:
            exp_disp = "-"

        display.append({
            "รหัสพาเลต": it["code"],
            "SKU": it["sku"],
            "ชื่อสินค้า": it.get("product_name", it.get("name", it["sku"])),
            "หมวดหมู่": it.get("category", "-"),
            "กลุ่ม ABC": f"Class {abc_c}",
            "ล็อตสินค้า": it.get("lot_number") or "-",
            "วันหมดอายุ": exp_disp,
            "จำนวน": it["qty"],
            "น้ำหนัก (kg)": f"{it['weight']:,.1f}",
            "คลัง": it["warehouse"],
            "ตำแหน่ง": pos,
            "สถานะ": PALLET_STATUS.get(it["status"], it["status"]),
        })
    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)



def generate_barcode_svg(code):
    """สร้าง SVG barcode จำลองจากรหัสพาเลตสไตล์ Logistics Standard"""
    bars = []
    x = 12
    # Standard start bars
    bars.append('<rect x="6" y="6" width="3" height="52" fill="#0F172A"/>')
    bars.append('<rect x="11" y="6" width="1" height="52" fill="#0F172A"/>')
    x = 16
    for i, ch in enumerate(code):
        v = ord(ch)
        for bit in range(5):
            w = 3 if ((v >> bit) & 1) else 1
            h = 48 if ((i + bit) % 3 == 0) else 44
            bars.append(f'<rect x="{x}" y="6" width="{w}" height="{h}" fill="#0F172A"/>')
            x += w + 2
        x += 3
    # Stop bars
    bars.append(f'<rect x="{x}" y="6" width="2" height="52" fill="#0F172A"/>')
    bars.append(f'<rect x="{x+4}" y="6" width="3" height="52" fill="#0F172A"/>')
    total_w = x + 16
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} 74" width="100%" height="74" style="background:#FFFFFF;border-radius:4px;">
    {''.join(bars)}
    <text x="{total_w//2}" y="67" text-anchor="middle" font-family="'Courier New', monospace" font-size="11" font-weight="bold" fill="#0F172A">*{code}*</text>
    </svg>"""
    return svg


@st.dialog("ใบส่งมอบสินค้า (Bill of Lading / Delivery Note)", width="large")
def delivery_note_dialog(trip_id):
    """แสดงใบส่งมอบสินค้าและใบกำกับการขนส่งมาตรฐาน Logistics สากล แบบทางการปราศจากอีโมจิ พร้อมระบบพิมพ์ PDF/กระดาษสมบูรณ์ 100%"""
    trip_data = rows(
        """SELECT s.*, t.name AS truck_name, t.slots, t.max_weight 
        FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.id = ?""",
        (trip_id,),
    )
    if not trip_data:
        st.error("ไม่พบข้อมูลเที่ยวรถ")
        return
    trip = trip_data[0]
    is_inbound = (trip["kind"] == "IN")
    doc_type_th = "ใบรับมอบสินค้าเข้าคลัง (GOODS RECEIPT NOTE)" if is_inbound else "ใบส่งมอบสินค้าออกคลัง (BILL OF LADING / DELIVERY ORDER)"
    doc_type_en = "INBOUND SHIPMENT RECEIPT" if is_inbound else "OUTBOUND BILL OF LADING"
    
    # Cargo Manifest Items
    rel_key = "inbound_id" if is_inbound else "outbound_id"
    items = rows(
        f"""SELECT p.code, p.sku, pr.name AS prod_name, pr.category, p.qty, p.weight, p.status, p.load_slot, p.inbound_slot
        FROM pallets p JOIN products pr ON pr.sku = p.sku
        WHERE p.{rel_key} = ? ORDER BY p.code""",
        (trip_id,),
    )
    
    tot_pallets = len(items)
    tot_units = sum(it["qty"] for it in items)
    tot_wt = sum(it["weight"] for it in items)
    max_truck_wt = trip.get("max_weight") or 8000.0
    wt_pct = int((tot_wt / max(max_truck_wt, 1)) * 100)
    
    barcode_svg = generate_barcode_svg(trip["id"])

    # Build Cargo Table Rows in Pure HTML (No emojis)
    table_rows_html = ""
    if items:
        for idx, it in enumerate(items, 1):
            slot_num = it.get("inbound_slot") if is_inbound else it.get("load_slot")
            slot_text = f"ช่อง {slot_num}" if slot_num else "-"
            bg_row = "#FFFFFF" if idx % 2 != 0 else "#F8FAFC"
            table_rows_html += f"""<tr style="background-color: {bg_row}; border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px; text-align: center; font-size: 12px; color: #64748B;">{idx}</td>
                <td style="padding: 8px 10px; font-weight: 700; font-family: monospace; font-size: 13px; color: #0F172A;">{it['code']}</td>
                <td style="padding: 8px 10px; font-size: 12px; color: #2563EB; font-weight: 600;">{it['sku']}</td>
                <td style="padding: 8px 10px; font-size: 13px; color: #1E293B;">{it['prod_name']}</td>
                <td style="padding: 8px 10px; font-size: 12px; color: #475569;">{it['category']}</td>
                <td style="padding: 8px; text-align: center; font-size: 12px; color: #475569;">{slot_text}</td>
                <td style="padding: 8px 10px; text-align: right; font-size: 13px; font-weight: 600; color: #0F172A;">{it['qty']:,}</td>
                <td style="padding: 8px 10px; text-align: right; font-size: 13px; font-weight: 700; color: #0F172A;">{it['weight']:,.1f}</td>
            </tr>"""
    else:
        table_rows_html = """<tr><td colspan="8" style="padding: 24px; text-align: center; color: #94A3B8; font-size: 13px;">ไม่มีรายการสินค้าที่บันทึกในเที่ยวรถนี้</td></tr>"""

    # Top Control Bar (Clean, formal, no emojis)
    c_btn1, c_btn2 = st.columns([1, 1.2])
    with c_btn1:
        st.markdown(
            """<button onclick="window.print()" style="background: #1E3A8A; color: white; border: none; padding: 9px 20px; border-radius: 6px; font-size: 13px; font-weight: bold; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 4px rgba(30,58,138,0.2);">พิมพ์เอกสาร / บันทึก PDF (Print Document)</button>""",
            unsafe_allow_html=True
        )
    with c_btn2:
        st.caption("คำแนะนำ: กดปุ่มสั่งพิมพ์ หรือกด Ctrl + P (Windows) / Cmd + P (Mac) เพื่อพิมพ์หรือบันทึกเป็น PDF")

    # Document Printable Sheet (Zero Emojis, 100% Formal Corporate Style)
    doc_html = textwrap.dedent(f"""
<div id="delivery-doc-sheet" style="background: #FFFFFF; color: #0F172A; border: 1px solid #CBD5E1; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); padding: 26px 30px; margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, sans-serif; line-height: 1.4;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0F172A; padding-bottom: 12px; margin-bottom: 14px;">
        <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="font-size: 20px; font-weight: 900; letter-spacing: 1px; color: #1E3A8A;">DOCKFLOW LOGISTICS NETWORK</span>
            </div>
            <div style="font-size: 11px; color: #475569; font-weight: 600;">ศูนย์กระจายสินค้าและคลังสินค้า (Bangkok Central Distribution Hub)</div>
            <div style="font-size: 10px; color: #64748B;">789 ถนนบางนา-ตราด แขวงบางนา เขตบางนา กรุงเทพฯ 10260 · โทร: 02-999-8888 · เลขประจำตัวผู้เสียภาษี: 0105566009988</div>
        </div>
        <div style="text-align: right; min-width: 220px;">
            <div style="font-size: 14px; font-weight: 900; color: #0F172A;">{doc_type_th}</div>
            <div style="font-size: 10.5px; font-weight: 700; color: #64748B; letter-spacing: 0.5px; margin-bottom: 6px;">{doc_type_en}</div>
            <div style="font-size: 12px; font-weight: 700; color: #1E3A8A; background: #EEF2FF; border: 1px solid #C7D2FE; padding: 3px 8px; border-radius: 4px; display: inline-block;">เลขที่เอกสาร: {trip['id']}</div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; font-size: 12px;">
        <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; background: #F8FAFC;">
            <div style="font-weight: 800; color: #1E3A8A; margin-bottom: 6px; font-size: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">1. ข้อมูลต้นทาง / คลังสินค้า (Origin Terminal)</div>
            <div style="display: grid; grid-template-columns: 95px 1fr; gap: 4px;">
                <span style="color: #64748B;">คลังสินค้า:</span><strong>{trip['warehouse']} (ศูนย์ใหญ่)</strong>
                <span style="color: #64748B;">ช่องเทียบท่า:</span><strong>{trip['dock'] or 'ช่องจอดทั่วไป'} ({DOCKS.get(trip['dock'], {}).get('title', 'N/A').replace('·', '-')})</strong>
                <span style="color: #64748B;">ประเภทงาน:</span><strong style="color: {'#059669' if is_inbound else '#2563EB'};">{'ขาเข้ารับสินค้า (Inbound)' if is_inbound else 'ขาออกจ่ายสินค้า (Outbound)'}</strong>
            </div>
        </div>

        <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; background: #F8FAFC;">
            <div style="font-weight: 800; color: #1E3A8A; margin-bottom: 6px; font-size: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">2. ข้อมูลยานพาหนะและขนส่ง (Carrier Details)</div>
            <div style="display: grid; grid-template-columns: 95px 1fr; gap: 4px;">
                <span style="color: #64748B;">รถบรรทุก:</span><strong>{trip['truck_name']} (รหัส {trip['truck_id']})</strong>
                <span style="color: #64748B;">พิกัดบรรทุก:</span><span>ความจุ {trip['slots']} ช่อง · รับน้ำหนักได้ {max_truck_wt:,.0f} kg</span>
                <span style="color: #64748B;">ความเร่งด่วน:</span><strong>{trip['priority']}</strong>
            </div>
        </div>

        <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; background: #F8FAFC;">
            <div style="font-weight: 800; color: #1E3A8A; margin-bottom: 6px; font-size: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">3. ข้อมูลคู่ค้า / ปลายทาง (Consignee / Destination)</div>
            <div style="display: grid; grid-template-columns: 95px 1fr; gap: 4px;">
                <span style="color: #64748B;">ชื่อคู่ค้า:</span><strong style="font-size: 13px;">{trip['partner']}</strong>
                <span style="color: #64748B;">สถานที่ส่งมอบ:</span><span>จุดรับ-ส่งสินค้าตามเอกสารสั่งงาน WMS</span>
            </div>
        </div>

        <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; background: #F8FAFC;">
            <div style="font-weight: 800; color: #1E3A8A; margin-bottom: 6px; font-size: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">4. บันทึกเวลาและการปฏิบัติงาน (Timestamps)</div>
            <div style="display: grid; grid-template-columns: 95px 1fr; gap: 4px;">
                <span style="color: #64748B;">เวลาสร้างเอกสาร:</span><span>{trip['created_at']}</span>
                <span style="color: #64748B;">เวลาปล่อยรถออก:</span><strong style="color: {'#059669' if trip.get('departed_at') else '#D97706'};">{trip.get('departed_at') or 'อยู่ระหว่างปฏิบัติงานเทียบท่า'}</strong>
                <span style="color: #64748B;">สถานะเที่ยวรถ:</span><span style="font-weight: 600;">{TRIP_STATUS.get(trip['status'], trip['status'])}</span>
            </div>
        </div>
    </div>

    <div style="margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 800; color: #0F172A; font-size: 13px;">รายการสินค้าตามบัญชีบรรทุก (CARGO MANIFEST)</span>
            <span style="font-size: 11px; color: #64748B;">รวมทั้งสิ้น {tot_pallets} พาเลต</span>
        </div>
        <table style="width: 100%; border-collapse: collapse; border: 1px solid #CBD5E1; font-size: 12px;">
            <thead>
                <tr style="background-color: #1E293B; color: #FFFFFF;">
                    <th style="padding: 8px; text-align: center; width: 45px; border-right: 1px solid #334155;">ลำดับ</th>
                    <th style="padding: 8px 10px; text-align: left; width: 110px; border-right: 1px solid #334155;">รหัสพาเลต</th>
                    <th style="padding: 8px 10px; text-align: left; width: 90px; border-right: 1px solid #334155;">รหัส SKU</th>
                    <th style="padding: 8px 10px; text-align: left; border-right: 1px solid #334155;">รายการสินค้า</th>
                    <th style="padding: 8px 10px; text-align: left; width: 110px; border-right: 1px solid #334155;">หมวดหมู่</th>
                    <th style="padding: 8px; text-align: center; width: 75px; border-right: 1px solid #334155;">ช่องบนรถ</th>
                    <th style="padding: 8px 10px; text-align: right; width: 75px; border-right: 1px solid #334155;">จำนวน</th>
                    <th style="padding: 8px 10px; text-align: right; width: 105px;">น้ำหนัก (kg)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
            <tfoot>
                <tr style="background-color: #F1F5F9; border-top: 2px solid #0F172A; font-weight: 800;">
                    <td colspan="6" style="padding: 9px 12px; text-align: right; font-size: 13px; color: #0F172A;">สรุปยอดรวมสุทธิ (GRAND TOTAL):</td>
                    <td style="padding: 9px 10px; text-align: right; font-size: 13px; color: #1E3A8A;">{tot_units:,} ชิ้น</td>
                    <td style="padding: 9px 10px; text-align: right; font-size: 13px; color: #1E3A8A;">{tot_wt:,.1f} kg</td>
                </tr>
            </tfoot>
        </table>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 14px; font-size: 12px;">
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 7px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 10.5px;">ปริมาณสินค้าทั้งหมด</div>
            <strong style="font-size: 15px; color: #0F172A;">{tot_pallets} พาเลต</strong> <span style="color: #64748B; font-size: 11px;">({tot_units:,} หน่วย)</span>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 7px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 10.5px;">น้ำหนักบรรทุกรวม</div>
            <strong style="font-size: 15px; color: #0F172A;">{tot_wt:,.1f} kg</strong> <span style="color: #64748B; font-size: 11px;">(พิกัด: {max_truck_wt:,.0f} kg)</span>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 7px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 10.5px;">อัตราการใช้พิกัดน้ำหนักรถ</div>
            <strong style="font-size: 15px; color: {'#059669' if wt_pct <= 90 else '#D97706'};">{wt_pct}%</strong> <span style="color: #64748B; font-size: 11px;">(เกณฑ์ปลอดภัย)</span>
        </div>
    </div>

    <div style="border-left: 3px solid #64748B; padding: 6px 12px; background: #F8FAFC; margin-bottom: 16px; font-size: 10.5px; color: #475569;">
        <strong>เงื่อนไขและข้อกำหนดการส่งมอบสินค้า:</strong><br>
        1. สินค้าตามรายการข้างต้นได้รับการตรวจนับ บรรจุ และซีลผนึกตามมาตรฐานความปลอดภัย WMS เรียบร้อย<br>
        2. ผู้รับมอบได้ตรวจสอบสภาพภายนอกของสินค้า พาเลต และเอกสารกำกับแล้วพบว่าอยู่ในสภาพสมบูรณ์ ครบถ้วนตามจำนวนที่ระบุ<br>
        3. เอกสารนี้มีผลสมบูรณ์เมื่อมีลายเซ็นรับรองครบทั้ง 3 ฝ่าย และบันทึกเข้าระบบประวัติของคลังสินค้าแล้ว
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; text-align: center; margin-top: 10px;">
        <div style="border: 1px solid #CBD5E1; border-radius: 6px; padding: 12px 10px; background: #FFFFFF;">
            <div style="font-size: 11.5px; font-weight: 800; color: #1E293B; margin-bottom: 30px;">เจ้าหน้าที่คลังสินค้าผู้จัดส่ง<br><span style="font-size: 10px; color: #64748B; font-weight: normal;">(Warehouse Dispatcher)</span></div>
            <div style="border-bottom: 1px dashed #64748B; width: 75%; margin: 0 auto 5px auto;"></div>
            <div style="font-size: 10.5px; color: #334155;">( .................................................... )</div>
            <div style="font-size: 10px; color: #64748B; margin-top: 2px;">วันที่: ...... / ...... / 2569</div>
        </div>

        <div style="border: 1px solid #CBD5E1; border-radius: 6px; padding: 12px 10px; background: #FFFFFF;">
            <div style="font-size: 11.5px; font-weight: 800; color: #1E293B; margin-bottom: 30px;">พนักงานขับรถ / ผู้ขนส่ง<br><span style="font-size: 10px; color: #64748B; font-weight: normal;">(Carrier / Driver)</span></div>
            <div style="border-bottom: 1px dashed #64748B; width: 75%; margin: 0 auto 5px auto;"></div>
            <div style="font-size: 10.5px; color: #334155;">( .................................................... )</div>
            <div style="font-size: 10px; color: #64748B; margin-top: 2px;">ทะเบียน: {trip['truck_name']}</div>
        </div>

        <div style="border: 1px solid #CBD5E1; border-radius: 6px; padding: 12px 10px; background: #FFFFFF; position: relative; overflow: hidden;">
            <div style="position: absolute; right: 8px; top: 8px; width: 62px; height: 62px; border: 2px dashed #DC2626; border-radius: 50%; opacity: 0.25; display: flex; align-items: center; justify-content: center; transform: rotate(-15deg); color: #DC2626; font-size: 8.5px; font-weight: 900; text-align: center; line-height: 1.1; pointer-events: none;">DOCKFLOW<br>VERIFIED<br>OFFICIAL</div>
            <div style="font-size: 11.5px; font-weight: 800; color: #1E293B; margin-bottom: 30px;">ผู้รับมอบสินค้าปลายทาง<br><span style="font-size: 10px; color: #64748B; font-weight: normal;">(Consignee / Receiver)</span></div>
            <div style="border-bottom: 1px dashed #64748B; width: 75%; margin: 0 auto 5px auto;"></div>
            <div style="font-size: 10.5px; color: #334155;">( .................................................... )</div>
            <div style="font-size: 10px; color: #64748B; margin-top: 2px;">วันที่: ...... / ...... / 2569</div>
        </div>
    </div>

    <div style="margin-top: 18px; padding-top: 10px; border-top: 1px solid #E2E8F0; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 10px; color: #94A3B8;">ระบบบริหารจัดการคลังสินค้า DOCKFLOW WMS System · เอกสารต้นฉบับ (Original Copy)</div>
        <div style="width: 130px;">{barcode_svg}</div>
    </div>
</div>
""")
    st.html(doc_html)


@st.dialog("🏷️ ป้ายบาร์โค้ด & ข้อมูลพาเลตสินค้า (Digital Pallet Tag)")
def pallet_detail_dialog(code):
    res = rows(
        """
        SELECT p.*, pr.name, pr.category, pr.unit_weight 
        FROM pallets p JOIN products pr ON pr.sku = p.sku 
        WHERE p.code = ?
        """,
        (code,),
    )
    if not res:
        st.error(f"ไม่พบข้อมูลพาเลต: {code}")
        return
    item = res[0]
    
    # 1. Digital Pallet Tag (Realistic Logistics Label)
    svg_barcode = generate_barcode_svg(item["code"])
    cat_color = CATEGORY_COLOR.get(item["category"], "#2563EB")
    abc_info = ABC_CLASS.get(item["sku"], {"class": "-", "label": "ทั่วไป", "color": "#64748B", "badge_bg": "#F8FAFC"})
    days_left = get_days_until_expiry(item.get("exp_date"))
    if days_left < 30:
        exp_badge = f"<span style='color:#DC2626; font-weight:700;'>⚠️ ใกล้หมดอายุ ({days_left} วัน)</span>"
    elif days_left < 60:
        exp_badge = f"<span style='color:#D97706; font-weight:700;'>⏳ เฝ้าระวัง ({days_left} วัน)</span>"
    else:
        exp_badge = f"<span style='color:#059669; font-weight:700;'>✅ ปกติ ({days_left} วัน)</span>"
    
    tag_html = f"""
    <div style="border: 2px solid #0F172A; border-radius: 10px; padding: 16px 20px; background: #FFFFFF; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); max-width: 490px; margin: 0 auto 16px auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0F172A; padding-bottom: 8px; margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: 900; color: #0F172A; letter-spacing: 1.5px;">DOCKFLOW WMS · PALLET TAG</div>
            <div style="display: flex; gap: 6px; align-items: center;">
                <span style="font-size: 11px; font-weight: 700; background: {abc_info['color']}; color: #FFFFFF; padding: 2px 7px; border-radius: 4px;">Class {abc_info['class']}</span>
                <span style="font-size: 11px; font-weight: 700; background: {cat_color}; color: #FFFFFF; padding: 2px 8px; border-radius: 4px;">{item['category']}</span>
            </div>
        </div>
        <div style="text-align: center; margin-bottom: 8px;">
            {svg_barcode}
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px; margin-bottom: 10px;">
            <div style="font-size: 16px; font-weight: 800; color: #0F172A;">{item['name']}</div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
                <span style="color: #64748B; font-weight: 600;">SKU: <strong style="color: #2563EB;">{item['sku']}</strong></span>
                <span style="color: #475569;">ล็อต: <strong style="color: #0F172A;">{item.get('lot_number') or '-'}</strong></span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; margin-bottom: 8px;">
            <div style="border-left: 3px solid #2563EB; padding-left: 6px;">
                <span style="color: #64748B;">จำนวนในพาเลต:</span><br>
                <strong style="font-size: 14px; color: #0F172A;">{item['qty']} หน่วย</strong>
            </div>
            <div style="border-left: 3px solid #10B981; padding-left: 6px;">
                <span style="color: #64748B;">น้ำหนักสุทธิ:</span><br>
                <strong style="font-size: 14px; color: #0F172A;">{item['weight']:,.1f} kg</strong>
            </div>
            <div style="border-left: 3px solid #F59E0B; padding-left: 6px;">
                <span style="color: #64748B;">คลัง / ตำแหน่งชั้น:</span><br>
                <strong style="font-size: 13px; color: #0F172A;">{item['warehouse']} · {item['location'] or 'พื้นที่พัก / บนรถ'}</strong>
            </div>
            <div style="border-left: 3px solid #8B5CF6; padding-left: 6px;">
                <span style="color: #64748B;">สถานะปัจจุบัน:</span><br>
                <strong style="font-size: 13px; color: #0F172A;">{PALLET_STATUS.get(item['status'], item['status'])}</strong>
            </div>
            <div style="border-left: 3px solid #06B6D4; padding-left: 6px;">
                <span style="color: #64748B;">วันผลิต (MFG):</span><br>
                <strong style="font-size: 12px; color: #0F172A;">{item.get('mfg_date') or '-'}</strong>
            </div>
            <div style="border-left: 3px solid #EC4899; padding-left: 6px;">
                <span style="color: #64748B;">วันหมดอายุ (EXP):</span><br>
                <strong style="font-size: 12px; color: #0F172A;">{item.get('exp_date') or '-'}</strong> {exp_badge}
            </div>
        </div>
        <div style="font-size: 10px; color: #94A3B8; display: flex; justify-content: space-between; border-top: 1px dashed #CBD5E1; padding-top: 6px;">
            <span>การจัดกลุ่มความเร็ว: <strong>{abc_info['label']}</strong></span>
            <span>อ้างอิง: {item['inbound_id'] or item['outbound_id'] or 'INIT'}</span>
        </div>
    </div>
    """
    st.markdown(tag_html, unsafe_allow_html=True)

    # 2. Reference Details
    with st.expander("🔍 ดูข้อมูลเชื่อมโยงในระบบ (System References)"):
        c1, c2 = st.columns(2)
        c1.markdown(f"**เที่ยวรับเข้า (Inbound):** `{item['inbound_id'] or 'ตั้งต้นระบบ'}`")
        c2.markdown(f"**ใบจ่ายสินค้า (Outbound):** `{item['outbound_id'] or 'ยังไม่ถูกจอง'}`")
        if item.get("inbound_slot"):
            c1.caption(f"ช่องบนรถขาเข้า: ช่องที่ {item['inbound_slot']}")
        if item.get("load_slot"):
            c2.caption(f"ช่องบนรถขาออก: ช่องที่ {item['load_slot']}")


# --- PAGE 1: CONTROL TOWER ---
@st.fragment(run_every="3s")
def control_tower_fragment():
    counts = {r["status"]: r["n"] for r in rows("SELECT status, COUNT(*) AS n FROM pallets GROUP BY status")}
    ready = counts.get("AVAILABLE", 0)
    receiving = counts.get("RECEIVING", 0)
    committed = sum(counts.get(s, 0) for s in ["RESERVED", "PICKED", "LOADED"])
    docked = scalar("SELECT COUNT(*) FROM trips WHERE status = 'DOCKED'")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 พร้อมใช้ในคลัง", f"{ready} พาเลต")
    c2.metric("📥 พื้นที่พักรับ", f"{receiving} พาเลต")
    c3.metric("📤 จอง/เตรียมส่ง", f"{committed} พาเลต")
    c4.metric("🅿️ ช่องจอดใช้งาน", f"{docked} / 4 ช่อง")

    st.caption(f"อัปเดตอัตโนมัติทุก 3 วินาที · เวลาเซิร์ฟเวอร์: {timestamp()}")

    st.subheader("ภาพรวมลานจอดรถ (Live Yard Map)")
    t1, t2, t3 = st.tabs(["🗺️ แผนผังกราฟิกลานจอด (Graphic Overview)", "🎛️ ข้อมูลสรุปช่องจอดสด (Interactive Dock Cards)", "📊 สถิติ & การวิเคราะห์เชิงลึก (Warehouse Analytics)"])

    with t1:
        yard_html = draw_yard_overview_html()
        st.html(yard_html)

    with t2:
        docked_trips = {r["dock"]: r for r in rows("SELECT s.*, t.name, t.slots FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.status = 'DOCKED'")}
        cols = st.columns(4)
        for idx, (dock_id, cfg) in enumerate(DOCKS.items()):
            with cols[idx]:
                with st.container(border=True):
                    trip = docked_trips.get(dock_id)
                    badge_color = "🟢" if trip else "⚪"
                    st.markdown(f"**{badge_color} {cfg['title']}**")
                    if trip:
                        st.markdown(f"🚚 **{trip['name']}** (`{trip['id']}`)")
                        st.caption(f"คู่ค้า: {trip['partner']}")
                        if trip["kind"] == "IN":
                            left_on_truck = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (trip["id"],))
                            st.info(f"เหลือบนรถ: {left_on_truck} พาเลต")
                        else:
                            loaded = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (trip["id"],))
                            total = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ?", (trip["id"],))
                            st.success(f"โหลดแล้ว: {loaded}/{total} พาเลต")
                    else:
                        st.caption("ว่างพร้อมใช้งาน")
    with t3:
        st.subheader("📈 การวิเคราะห์ข้อมูลคลังสินค้าเชิงลึก (Warehouse Analytics & BI)")
        st.caption("ระบบวิเคราะห์สถิติคลังสินค้าแบบเรียลไทม์ · สัดส่วนหมวดหมู่ ความจุคลัง และอัตราการหมุนเวียนสินค้า (Symmetric Analytics Hub)")
        
        # ============================================================
        # ROW 1: 2 SYMMETRIC 50% / 50% CHART CARDS
        # ============================================================
        r1_col1, r1_col2 = st.columns(2)
        
        with r1_col1:
            with st.container(border=True):
                st.markdown("##### 🍩 สัดส่วนสินค้าตามหมวดหมู่ (Category Share)")
                cat_data = rows("""
                    SELECT pr.category, COUNT(p.code) AS cnt, SUM(p.weight) AS total_weight
                    FROM pallets p JOIN products pr ON pr.sku = p.sku
                    WHERE p.status IN ('AVAILABLE', 'RESERVED', 'RECEIVING', 'PICKED', 'LOADED')
                    GROUP BY pr.category ORDER BY cnt DESC
                """)
                if cat_data:
                    labels = [d["category"] for d in cat_data]
                    sizes = [d["cnt"] for d in cat_data]
                    colors = [CATEGORY_COLOR.get(l, "#64748B") for l in labels]
                    
                    fig_cat, ax_cat = plt.subplots(figsize=(5.2, 3.4), dpi=140)
                    fig_cat.patch.set_facecolor("#FFFFFF")
                    ax_cat.set_facecolor("#FFFFFF")
                    
                    wedges, texts, autotexts = ax_cat.pie(
                        sizes, labels=None, colors=colors, autopct="%1.0f%%",
                        startangle=90, pctdistance=0.72,
                        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=2)
                    )
                    for t in autotexts:
                        t.set_fontsize(8.5)
                        t.set_color("white")
                        t.set_fontweight("bold")
                    
                    ax_cat.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.88, 0.5), fontsize=8.5, frameon=False)
                    ax_cat.text(0, 0.11, "สต็อกรวม", ha="center", va="center", fontsize=9.5, color="#64748B", fontweight="bold")
                    ax_cat.text(0, -0.13, f"{sum(sizes)} พาเลต", ha="center", va="center", fontsize=13, color="#0F172A", fontweight="bold")
                    fig_cat.tight_layout()
                    st.pyplot(fig_cat, use_container_width=True)
                    plt.close(fig_cat)
                    
                    # 2x2 Symmetrical Category Badges Underneath
                    c_row1 = st.columns(2)
                    c_row2 = st.columns(2)
                    tot_cat_cnt = max(sum(sizes), 1)
                    for idx, c in enumerate(cat_data[:4]):
                        target_col = c_row1[idx] if idx < 2 else c_row2[idx - 2]
                        with target_col:
                            color_pill = CATEGORY_COLOR.get(c['category'], '#64748B')
                            pct_cat = int((c['cnt'] / tot_cat_cnt) * 100)
                            st.markdown(
                                f"""<div style="border-left: 4px solid {color_pill}; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px; min-height: 52px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                                    <div style="font-size: 11px; color: #64748B; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500;">{c['category']}</div>
                                    <strong style="font-size: 13.5px; color: #0F172A;">{c['cnt']} <span style="font-size: 11px; font-weight: normal; color: #64748B;">พาเลต ({pct_cat}%)</span></strong>
                                </div>""",
                                unsafe_allow_html=True
                            )
                else:
                    st.info("ยังไม่มีข้อมูลสินค้าในคลัง")
        
        with r1_col2:
            with st.container(border=True):
                st.markdown("##### 🏢 อัตราการใช้ความจุคลัง (Warehouse Capacity: WH-A vs WH-B)")
                wh_util = []
                for wh_name in WAREHOUSES:
                    total_cap = scalar("SELECT SUM(capacity) FROM locations WHERE warehouse = ?", (wh_name,))
                    used_cnt = scalar("SELECT COUNT(*) FROM pallets WHERE warehouse = ? AND status IN ('AVAILABLE', 'RESERVED') AND location IS NOT NULL", (wh_name,))
                    wh_util.append({"wh": wh_name, "used": used_cnt, "cap": total_cap or 1})
                
                fig_wh, ax_wh = plt.subplots(figsize=(5.2, 3.4), dpi=140)
                fig_wh.patch.set_facecolor("#FFFFFF")
                ax_wh.set_facecolor("#FFFFFF")
                
                x_pos = range(len(wh_util))
                bar_w = 0.42
                caps = [w["cap"] for w in wh_util]
                useds = [w["used"] for w in wh_util]
                pcts = [int((u / c) * 100) for u, c in zip(useds, caps)]
                
                ax_wh.bar(x_pos, caps, color="#F1F5F9", edgecolor="#CBD5E1", label="ความจุสูงสุด", width=bar_w, linewidth=1.2)
                ax_wh.bar(x_pos, useds, color=["#3B82F6", "#8B5CF6"], edgecolor="#1E293B", label="จัดเก็บจริง", width=bar_w, linewidth=1.2)
                
                for idx, p in enumerate(pcts):
                    ax_wh.text(idx, useds[idx] + 0.8, f"{useds[idx]}/{caps[idx]} ({p}%)", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")
                
                ax_wh.set_xticks(x_pos)
                ax_wh.set_xticklabels([f"คลัง {w['wh']}" for w in wh_util], fontsize=9.5, fontweight="bold")
                ax_wh.set_ylabel("จำนวนพาเลต", fontsize=8.5)
                ax_wh.set_ylim(0, max(caps) + 6)
                ax_wh.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")
                ax_wh.spines["top"].set_visible(False)
                ax_wh.spines["right"].set_visible(False)
                ax_wh.spines["left"].set_color("#CBD5E1")
                ax_wh.spines["bottom"].set_color("#CBD5E1")
                fig_wh.tight_layout()
                st.pyplot(fig_wh, use_container_width=True)
                plt.close(fig_wh)
                
                # 2x2 Symmetrical Warehouse Summary Underneath
                tot_caps = sum(caps)
                tot_used = sum(useds)
                tot_free = max(0, tot_caps - tot_used)
                free_pct = int((tot_free / max(tot_caps, 1)) * 100)

                w_row1 = st.columns(2)
                w_row2 = st.columns(2)

                with w_row1[0]:
                    st.markdown(
                        f"""<div style="border-left: 4px solid #3B82F6; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px; min-height: 52px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">คลัง WH-A (ความจุ {caps[0]} ช่อง)</div>
                            <strong style="font-size: 13.5px; color: #0F172A;">{useds[0]} <span style="font-size: 11px; font-weight: normal; color: #3B82F6;">พาเลต ({pcts[0]}%)</span></strong>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with w_row1[1]:
                    st.markdown(
                        f"""<div style="border-left: 4px solid #8B5CF6; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px; min-height: 52px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">คลัง WH-B (ความจุ {caps[1]} ช่อง)</div>
                            <strong style="font-size: 13.5px; color: #0F172A;">{useds[1]} <span style="font-size: 11px; font-weight: normal; color: #8B5CF6;">พาเลต ({pcts[1]}%)</span></strong>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with w_row2[0]:
                    st.markdown(
                        f"""<div style="border-left: 4px solid #0EA5E9; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px; min-height: 52px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">ความจุรวมคลังสินค้าทั้งหมด</div>
                            <strong style="font-size: 13.5px; color: #0F172A;">{tot_caps} <span style="font-size: 11px; font-weight: normal; color: #64748B;">ช่องจัดเก็บ (2 คลัง)</span></strong>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with w_row2[1]:
                    st.markdown(
                        f"""<div style="border-left: 4px solid #10B981; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px; min-height: 52px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">พื้นที่ว่างพร้อมจัดเก็บ (Free Space)</div>
                            <strong style="font-size: 13.5px; color: #0F172A;">{tot_free} <span style="font-size: 11px; font-weight: normal; color: #10B981;">ช่องว่าง ({free_pct}%)</span></strong>
                        </div>""",
                        unsafe_allow_html=True
                    )

        # ============================================================
        # ROW 2: 2 SYMMETRIC 50% / 50% OPERATIONAL METRIC CARDS
        # ============================================================
        r2_col1, r2_col2 = st.columns(2)
        
        with r2_col1:
            with st.container(border=True):
                st.markdown("##### 🔄 ปริมาณสินค้าหมุนเวียน (Inventory Turnover: IN vs OUT)")
                in_count = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id IS NOT NULL")
                out_count = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id IS NOT NULL")
                in_trips = scalar("SELECT COUNT(*) FROM trips WHERE kind = 'IN'")
                out_trips = scalar("SELECT COUNT(*) FROM trips WHERE kind = 'OUT'")
                shipped_cnt = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'SHIPPED'")
                ratio = (out_count / max(in_count, 1)) * 100
                
                # Symmetrical 2x2 grid metrics
                t1, t2 = st.columns(2)
                t1.metric("📥 รับเข้าสะสม (IN)", f"{in_count} พาเลต", f"{in_trips} เที่ยวรถ")
                t2.metric("📤 จ่ายออกสะสม (OUT)", f"{out_count} พาเลต", f"{out_trips} เที่ยวรถ")
                
                st.divider()
                
                t3_m, t4_m = st.columns(2)
                t3_m.metric("⚡ อัตราหมุนเวียน (Ratio)", f"{ratio:.0f}%", "สัดส่วนจ่าย/รับ")
                t4_m.metric("🚀 ส่งมอบสำเร็จ (Shipped)", f"{shipped_cnt} พาเลต", "จบกระบวนการ")
                
                st.caption(f"เปรียบเทียบสัดส่วน: รับเข้า {in_count} พาเลต ➔ จ่ายออก {out_count} พาเลต")
                st.progress(min(1.0, out_count / max(in_count, 1)))

        with r2_col2:
            with st.container(border=True):
                st.markdown("##### 🧱 สถานะสินค้าในกระบวนการ (Inventory Pipeline Flow)")
                status_counts = {r["status"]: r["n"] for r in rows("SELECT status, COUNT(*) AS n FROM pallets GROUP BY status")}
                tot_all_pallets = max(sum(status_counts.values()), 1)
                
                p_c1, p_c2 = st.columns(2)
                
                with p_c1:
                    st.markdown("**📥 กระบวนการรับเข้า (Inbound Flow)**")
                    in_flow = [
                        ("ON_INBOUND", "🚚 บนรถขาเข้า", "#6366F1"),
                        ("RECEIVING", "📥 พื้นที่พักรับ", "#06B6D4"),
                        ("AVAILABLE", "✅ พร้อมใช้ในคลัง", "#10B981"),
                    ]
                    for sc, sname, scolor in in_flow:
                        cnt = status_counts.get(sc, 0)
                        pct = int((cnt / tot_all_pallets) * 100)
                        st.markdown(
                            f"""<div style="margin-bottom: 6px; padding: 6px 10px; background: #F8FAFC; border-radius: 6px; border-left: 3px solid {scolor}; display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 12px; color: #334155;">{sname}</span>
                                <div><strong style="font-size: 13px; color: #0F172A;">{cnt}</strong> <span style="font-size: 10px; color: #64748B;">({pct}%)</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )

                with p_c2:
                    st.markdown("**📤 กระบวนการจ่ายออก (Outbound Flow)**")
                    out_flow = [
                        ("RESERVED", "🔖 จองแล้ว / รอหยิบ", "#F59E0B"),
                        ("PICKED", "🛒 พื้นที่พักจ่าย", "#8B5CF6"),
                        ("LOADED", "📦 โหลดบนรถแล้ว", "#3B82F6"),
                        ("SHIPPED", "🚀 ส่งมอบสำเร็จ", "#059669"),
                    ]
                    for sc, sname, scolor in out_flow:
                        cnt = status_counts.get(sc, 0)
                        pct = int((cnt / tot_all_pallets) * 100)
                        st.markdown(
                            f"""<div style="margin-bottom: 5px; padding: 5px 10px; background: #F8FAFC; border-radius: 6px; border-left: 3px solid {scolor}; display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 11.5px; color: #334155;">{sname}</span>
                                <div><strong style="font-size: 13px; color: #0F172A;">{cnt}</strong> <span style="font-size: 10px; color: #64748B;">({pct}%)</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )

    st.subheader("กิจกรรมล่าสุดในระบบ (Recent System Activity)")
    st.dataframe(
        pd.DataFrame(
            rows("SELECT created_at AS เวลา, action AS กิจกรรม, reference AS อ้างอิง, detail AS รายละเอียด FROM events ORDER BY id DESC LIMIT 8")
        ),
        use_container_width=True,
        hide_index=True,
    )



def render_smart_workflow_guide():
    """ฟังก์ชันคำแนะนำขั้นตอนการทำงาน (ปิดการแสดงผล)"""
    pass


def page_control_tower():
    st.header("Control Tower")
    st.caption("ภาพรวมสถานะคลังสินค้า ลานเทียบท่า และความคืบหน้าการดำเนินงานแบบเรียลไทม์")
    control_tower_fragment()


# --- PAGE 2: DOCK YARD ---
def page_dock_yard():
    st.header("Dock Yard & Fleet Management")
    st.caption("บริหารจัดการคิวรถเทียบท่า จัดสรรช่องจอด และทะเบียนยานพาหนะขนส่ง (Fleet Registry)")

    tab_dock, tab_fleet = st.tabs([
        "🅿️ จัดการคิว & ช่องเทียบท่า (Dock Yard)",
        "🚚 ทะเบียนยานพาหนะ & กองรถ (Fleet Registry)"
    ])

    with tab_dock:
        # Gate-in / Add Truck to Queue Expander
        with st.expander("➕ นำรถเข้าคิวเทียบท่า (Check-in Truck to Queue / Gate-In)", expanded=False):
            st.caption("เปิดเที่ยวงานขนส่งและนำรถเข้าสู่คิวรอเทียบท่า (รองรับทั้งเที่ยวรับสินค้าเข้า และเที่ยวเบิกจ่ายสินค้าออก)")
            free_trucks = rows("SELECT * FROM trucks WHERE id NOT IN (SELECT truck_id FROM trips WHERE status NOT IN ('DEPARTED', 'CANCELLED')) ORDER BY id ASC")

            if not free_trucks:
                st.warning("⚠️ ไม่มีรถว่างในกองรถขณะนี้ (รถทุกคันกำลังปฏิบัติงาน หรือสามารถเพิ่มรถใหม่ได้ที่แท็บ 'ทะเบียนยานพาหนะ & กองรถ')")
            else:
                t_lookup = {t["id"]: t for t in free_trucks}
                job_kind = st.radio("เลือกประเภทเที่ยวงาน:", ["📥 ขาเข้ารับสินค้า (Inbound Delivery)", "📤 ขาออกส่งสินค้า (Outbound Pickup)"], horizontal=True, key="yard_gatein_kind")
                is_inbound = "ขาเข้า" in job_kind

                with st.form("yard_gatein_form"):
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        sel_tid = st.selectbox(
                            "เลือกรถบรรทุกที่เข้าคิว:*",
                            list(t_lookup),
                            format_func=lambda x: f"{x} · {t_lookup[x]['name']} ({t_lookup[x].get('plate', '-')}) · {t_lookup[x].get('carrier', '-')} (จุ {t_lookup[x]['slots']} ช่อง / {t_lookup[x]['max_weight']:,.0f} kg)"
                        )
                        target_wh = st.selectbox("คลังสินค้า:", WAREHOUSES, key="yard_gatein_wh")
                        gate_prio = st.selectbox("ความเร่งด่วน (Priority):", ["High", "Normal", "Low"], index=1, key="yard_gatein_prio")
                        def_partner = "บจก. ซัพพลายเออร์ไทย" if is_inbound else "ศูนย์กระจายสินค้าสาขา"
                        partner_name = st.text_input("ชื่อคู่ค้า / ผู้ส่ง / สาขาปลายทาง:*", def_partner, key="yard_gatein_partner")

                    t_selected = t_lookup[sel_tid]

                    with f_col2:
                        if is_inbound:
                            all_prods = rows("SELECT * FROM products")
                            p_dict = {p["sku"]: p for p in all_prods}
                            chosen_sku = st.selectbox("สินค้าที่บรรทุกมา:", list(p_dict), format_func=lambda x: f"{x} · {p_dict[x]['name']} ({p_dict[x]['unit_weight']} kg/หน่วย)")
                            units_per_pal = st.number_input("จำนวนหน่วยต่อพาเลต:", min_value=1, max_value=500, value=20, key="yard_in_units")
                            pal_count = st.number_input("จำนวนพาเลต:", min_value=1, max_value=int(t_selected["slots"]), value=min(4, int(t_selected["slots"])), key="yard_in_count")
                            est_wt = units_per_pal * pal_count * p_dict[chosen_sku]["unit_weight"]
                            st.caption(f"น้ำหนักประเมิน: **{est_wt:,.0f} kg** / พิกัดรถ: **{t_selected['max_weight']:,.0f} kg**")
                        else:
                            avail_pals = rows("SELECT p.*, pr.name FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.warehouse = ? AND p.status = 'AVAILABLE' ORDER BY p.code ASC", (target_wh,))
                            if not avail_pals:
                                st.warning(f"ไม่มีพาเลตพร้อมใช้ในคลัง {target_wh}")
                                chosen_codes = []
                            else:
                                pal_dict = {p["code"]: p for p in avail_pals}
                                chosen_codes = st.multiselect(
                                    "เลือกพาเลตที่จะจ่ายออก:",
                                    list(pal_dict),
                                    max_selections=int(t_selected["slots"]),
                                    format_func=lambda x: f"{x} · {pal_dict[x]['name']} ({pal_dict[x]['weight']:,.0f} kg)"
                                )
                                est_wt = sum(pal_dict[c]["weight"] for c in chosen_codes)
                                st.caption(f"เลือก: **{len(chosen_codes)} / {t_selected['slots']} ช่อง** · น้ำหนัก: **{est_wt:,.0f} / {t_selected['max_weight']:,.0f} kg**")

                    btn_gatein = st.form_submit_button("🚚 บันทึกนำรถเข้าคิวลานจอด (Gate-In to Queue)", type="primary", use_container_width=True)
                    if btn_gatein:
                        if not partner_name.strip():
                            st.error("กรุณาระบุชื่อคู่ค้าหรือปลายทาง")
                        elif not is_inbound and not chosen_codes:
                            st.error("กรุณาเลือกพาเลตสินค้าที่จะจ่ายออกอย่างน้อย 1 รายการ")
                        else:
                            try:
                                kind_code = "IN" if is_inbound else "OUT"
                                payload = (chosen_sku, units_per_pal, pal_count) if is_inbound else chosen_codes
                                new_trip_id = create_new_trip(
                                    kind=kind_code,
                                    truck_id=sel_tid,
                                    warehouse=target_wh,
                                    partner=partner_name.strip(),
                                    priority=gate_prio,
                                    payload=payload
                                )
                                st.toast(f"นำรถ {sel_tid} เข้าคิวสำเร็จ รหัสเที่ยว {new_trip_id}", icon="🚚")
                                st.success(f"นำรถ {sel_tid} เข้าคิวเทียบท่าเรียบร้อย (รหัสเที่ยว {new_trip_id})")
                                st.rerun()
                            except Exception as err:
                                st.error(str(err))

        waiting = rows(
            """
            SELECT s.*, t.name AS truck_name, t.slots, t.plate, t.carrier, t.driver_name, t.driver_phone, t.truck_type,
                   (SELECT COUNT(*) FROM pallets p WHERE p.inbound_id = s.id OR p.outbound_id = s.id) AS pallet_count
            FROM trips s JOIN trucks t ON t.id = s.truck_id 
            WHERE s.status = 'WAITING'
            ORDER BY CASE s.priority WHEN 'High' THEN 1 WHEN 'Normal' THEN 2 ELSE 3 END, s.created_at ASC
            """
        )

        with st.container(border=True):
            st.subheader("คิวรถรอเทียบท่า (Queue)")
            if not waiting:
                st.info("ไม่มีรถรอเทียบท่าในขณะนี้ ระบบพร้อมรับงานขนส่งใหม่")
            else:
                q_table = []
                for w in waiting:
                    q_table.append({
                        "ความเร่งด่วน": f"{PRIORITY_ICON[w['priority']]} {w['priority']}",
                        "รหัสเที่ยว": w["id"],
                        "ประเภท": "ขาเข้า (IN)" if w["kind"] == "IN" else "ขาออก (OUT)",
                        "รถบรรทุก": f"{w['truck_name']} ({w.get('plate') or '-'})",
                        "ผู้ให้บริการ/เจ้าของ": w.get("carrier") or "รถร่วมขนส่ง",
                        "คนขับ": f"{w.get('driver_name') or '-'} ({w.get('driver_phone') or '-'})",
                        "คลัง": w["warehouse"],
                        "คู่ค้า/ปลายทาง": w["partner"],
                        "จำนวน": f"{w['pallet_count']} พาเลต",
                        "เวลาเข้าคิว": w["created_at"],
                    })
                st.dataframe(pd.DataFrame(q_table), use_container_width=True, hide_index=True)

                lookup = {t["id"]: t for t in waiting}
                col_t, col_d = st.columns(2)
                with col_t:
                    selected = st.selectbox(
                        "เลือกเที่ยวรถเพื่อดำเนินการ:",
                        list(lookup),
                        format_func=lambda x: f"{PRIORITY_ICON[lookup[x]['priority']]} [{lookup[x]['priority']}] {x} · {lookup[x]['truck_name']} ({lookup[x].get('plate', '-')}) · {lookup[x].get('carrier', '-')}",
                    )
                trip = lookup[selected]

                occupied_docks = {r["dock"] for r in rows("SELECT dock FROM trips WHERE status = 'DOCKED'")}
                free_docks = [d for d, cfg in DOCKS.items() if cfg["kind"] == trip["kind"] and cfg["warehouse"] == trip["warehouse"] and d not in occupied_docks]

                with col_d:
                    if free_docks:
                        dock_choice = st.selectbox("เลือกช่องเทียบท่าว่าง:", free_docks, format_func=lambda x: f"{x} · {DOCKS[x]['title']}")
                    else:
                        st.selectbox("เลือกช่องเทียบท่าว่าง:", ["ไม่มีช่องว่างสำหรับประเภทนี้"], disabled=True)
                        dock_choice = None

                b_cancel, b_park = st.columns(2)
                with b_cancel:
                    if st.button("ยกเลิกเที่ยวรถ (Cancel Trip)", use_container_width=True):
                        try:
                            execute_action("cancel_trip", selected)
                            st.toast(f"ยกเลิกเที่ยวรถ {selected} สำเร็จ", icon="🗑️")
                            st.success(f"ยกเลิกเที่ยวรถ {selected} เรียบร้อยแล้ว")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                with b_park:
                    if free_docks and dock_choice:
                        if st.button("นำรถเข้าเทียบท่า (Park)", type="primary", use_container_width=True):
                            try:
                                execute_action("park", selected, dock_choice)
                                st.toast(f"นำรถเข้าเทียบท่าช่อง {dock_choice} สำเร็จ", icon="✅")
                                st.success(f"นำรถเข้าเทียบท่าช่อง {dock_choice} เรียบร้อยแล้ว")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                    else:
                        st.button("ไม่มีช่องเทียบท่าว่าง", disabled=True, use_container_width=True)

        st.subheader("สถานะรถประจำช่องเทียบท่า (Docked Vehicles)")
        docked_trips = rows(
            """
            SELECT s.id, s.dock, s.kind, s.warehouse, s.partner, 
                   t.name, t.plate, t.carrier, t.driver_name, t.driver_phone, t.truck_type, t.slots, t.max_weight
            FROM trips s JOIN trucks t ON t.id = s.truck_id 
            WHERE s.status = 'DOCKED' ORDER BY s.dock ASC
            """
        )

        if not docked_trips:
            st.info("ไม่มีรถประจำช่องเทียบท่าในขณะนี้")
        else:
            sel_docked = st.selectbox(
                "เลือกรถในช่องเทียบท่าเพื่อตรวจสอบ:",
                [d["id"] for d in docked_trips],
                format_func=lambda x: next(f"ช่อง {d['dock']}: {d['name']} ({d.get('plate', '-')}) · {d.get('carrier', '-')} ({x} · {'ขาเข้า' if d['kind'] == 'IN' else 'ขาออก'})" for d in docked_trips if d["id"] == x),
            )
            c_left, c_right = st.columns([1, 1])
            with c_left:
                fig = draw_truck_graphic(sel_docked)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c_right:
                dt_info = next(d for d in docked_trips if d["id"] == sel_docked)
                with st.container(border=True):
                    st.subheader(f"🚚 {dt_info['name']} (ช่อง {dt_info['dock']})")
                    st.markdown(f"**ป้ายทะเบียน:** `{dt_info.get('plate') or '-'}` | **ประเภทรถ:** `{dt_info.get('truck_type') or '-'}`")
                    st.markdown(f"**ผู้ให้บริการขนส่ง:** `{dt_info.get('carrier') or '-'}`")
                    st.markdown(f"**พนักงานขับรถ:** `{dt_info.get('driver_name') or '-'}` (โทร `{dt_info.get('driver_phone') or '-'}`)")
                    st.markdown(f"**ประเภท:** {'รับสินค้าเข้า (IN)' if dt_info['kind'] == 'IN' else 'จ่ายสินค้าออก (OUT)'}")
                    st.markdown(f"**คลังสินค้า:** `{dt_info['warehouse']}` | **คู่ค้า:** `{dt_info['partner']}`")

                    if dt_info["kind"] == "IN":
                        on_truck = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (sel_docked,))
                        recv = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'RECEIVING'", (sel_docked,))
                        avail = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'AVAILABLE'", (sel_docked,))
                        st.metric("สินค้าค้างบนรถ", f"{on_truck} พาเลต")
                        st.caption(f"พักรับแล้ว: {recv} | จัดเก็บแล้ว: {avail}")
                        if on_truck == 0:
                            st.success("ขนถ่ายสินค้าลงครบถ้วน พร้อมบันทึกปล่อยรถ")
                            if st.button("🚚 บันทึกปล่อยรถออกจากลาน (Depart)", type="primary", use_container_width=True):
                                try:
                                    execute_action("depart", sel_docked)
                                    st.toast("บันทึกปล่อยรถขาเข้าเรียบร้อย", icon="✅")
                                    st.success("บันทึกปล่อยรถออกจากลานเรียบร้อย")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                    else:
                        loaded = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (sel_docked,))
                        total = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ?", (sel_docked,))
                        st.metric("ความคืบหน้าการโหลด", f"{loaded} / {total} พาเลต")
                        if loaded == total and total > 0:
                            st.success("โหลดสินค้าขึ้นรถครบถ้วน พร้อมบันทึกปล่อยรถ")
                            if st.button("🚚 บันทึกปล่อยรถออกจากลาน (Depart)", type="primary", use_container_width=True):
                                try:
                                    execute_action("depart", sel_docked)
                                    st.toast("บันทึกปล่อยรถขาออกเรียบร้อย", icon="🚚")
                                    st.success("บันทึกปล่อยรถออกจากลานเรียบร้อย")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

    with tab_fleet:
        st.subheader("ทะเบียนยานพาหนะและผู้ให้บริการขนส่ง (Fleet Registry)")
        st.caption("ตรวจสอบกองรถบรรทุก ผู้ให้บริการขนส่ง พนักงานขับรถ และสร้าง/แก้ไข/ลบข้อมูลยานพาหนะได้อย่างยืดหยุ่น")

        all_trucks = rows("""
            SELECT t.*, 
                   (SELECT id FROM trips WHERE truck_id = t.id AND status NOT IN ('DEPARTED', 'CANCELLED') LIMIT 1) AS active_trip_id,
                   (SELECT status FROM trips WHERE truck_id = t.id AND status NOT IN ('DEPARTED', 'CANCELLED') LIMIT 1) AS trip_status
            FROM trucks t ORDER BY t.id ASC
        """)

        tot_trucks = len(all_trucks)
        busy_trucks = sum(1 for t in all_trucks if t.get("active_trip_id"))
        idle_trucks = tot_trucks - busy_trucks
        carriers_cnt = len(set(t.get("carrier") for t in all_trucks if t.get("carrier")))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("รถทั้งหมดในระบบ", f"{tot_trucks} คัน")
        k2.metric("กำลังปฏิบัติงาน", f"{busy_trucks} คัน")
        k3.metric("ว่างพร้อมรับงาน", f"{idle_trucks} คัน")
        k4.metric("ผู้ให้บริการขนส่ง", f"{carriers_cnt} บริษัท")

        fleet_rows = []
        for tr in all_trucks:
            status_badge = f"🔴 วิ่งงาน ({tr['active_trip_id']})" if tr.get("active_trip_id") else "🟢 ว่างพร้อมใช้งาน"
            fleet_rows.append({
                "รหัสรถ": tr["id"],
                "ป้ายทะเบียน": tr.get("plate") or "-",
                "ชื่อเรียก/รุ่นรถ": tr["name"],
                "ผู้ให้บริการ/เจ้าของ": tr.get("carrier") or "-",
                "พนักงานขับรถ": tr.get("driver_name") or "-",
                "เบอร์โทรศัพท์": tr.get("driver_phone") or "-",
                "ประเภทรถ": tr.get("truck_type") or "รถบรรทุก",
                "ความจุ (ช่อง)": f"{tr['slots']} ช่อง",
                "พิกัดน้ำหนัก": f"{tr['max_weight']:,.0f} kg",
                "สถานะงาน": status_badge
            })
        st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, hide_index=True)

        st.info("💡 **ขั้นตอนการนำรถเข้าปฏิบัติงาน:** รถที่ลงทะเบียนในหน้านี้จะมีสถานะ **'🟢 ว่างพร้อมใช้งาน'** เมื่อรถเข้ามาถึงคลัง สามารถเปิดเที่ยวงานและนำรถเข้าคิวลานเทียบท่าได้ทันทีผ่านทางลัดด้านล่าง หรือไปที่แท็บ **'🅿️ จัดการคิว & ช่องเทียบท่า'**")

        # Quick Dispatch from Fleet Registry
        idle_trucks_list = [t for t in all_trucks if not t.get("active_trip_id")]
        if idle_trucks_list:
            with st.expander("🚀 ทางลัด: เปิดเที่ยวงานและส่งรถว่างเข้าคิวเทียบท่าทันที (Quick Dispatch to Yard Queue)"):
                st.caption("นำรถที่สถานะว่าง เข้าสู่คิวลานจอด (Dock Yard) ได้โดยตรงจากหน้านี้")
                qd_lookup = {t["id"]: t for t in idle_trucks_list}
                q_col1, q_col2 = st.columns(2)
                with q_col1:
                    sel_q_tid = st.selectbox(
                        "เลือกรถว่างเพื่อเปิดเที่ยวงาน:",
                        list(qd_lookup),
                        format_func=lambda x: f"{x} · {qd_lookup[x]['name']} ({qd_lookup[x].get('plate', '-')}) · {qd_lookup[x].get('carrier', '-')}",
                        key="qd_truck_select"
                    )
                with q_col2:
                    q_job = st.radio("ประเภทงาน:", ["📥 ขาเข้ารับสินค้า (IN)", "📤 ขาออกส่งสินค้า (OUT)"], horizontal=True, key="qd_job_radio")

                sel_q_truck = qd_lookup[sel_q_tid]
                is_q_in = "ขาเข้า" in q_job

                with st.form("quick_dispatch_fleet_form"):
                    qd_f1, qd_f2 = st.columns(2)
                    with qd_f1:
                        qd_wh = st.selectbox("คลังสินค้า:", WAREHOUSES, key="qd_form_wh")
                        qd_prio = st.selectbox("ความเร่งด่วน (Priority):", ["High", "Normal", "Low"], index=1, key="qd_form_prio")
                        qd_partner = st.text_input("ชื่อคู่ค้า / ผู้ส่ง / สาขาปลายทาง:*", "บจก. คู่ค้าซัพพลายเออร์" if is_q_in else "สาขาหลักปลายทาง", key="qd_form_partner")
                    with qd_f2:
                        if is_q_in:
                            prods_all = rows("SELECT * FROM products")
                            qd_pmap = {p["sku"]: p for p in prods_all}
                            qd_sku = st.selectbox("สินค้า:", list(qd_pmap), format_func=lambda x: f"{x} · {qd_pmap[x]['name']}", key="qd_form_sku")
                            qd_u = st.number_input("หน่วย/พาเลต:", 1, 500, 20, key="qd_form_u")
                            qd_c = st.number_input("จำนวนพาเลต:", 1, int(sel_q_truck["slots"]), min(4, int(sel_q_truck["slots"])), key="qd_form_c")
                            st.caption(f"น้ำหนัก: **{qd_u * qd_c * qd_pmap[qd_sku]['unit_weight']:,.0f} kg** / พิกัดรถ: **{sel_q_truck['max_weight']:,.0f} kg**")
                        else:
                            av_p = rows("SELECT p.*, pr.name FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.warehouse = ? AND p.status = 'AVAILABLE' ORDER BY p.code ASC", (qd_wh,))
                            if not av_p:
                                st.warning(f"ไม่มีพาเลตพร้อมใช้ในคลัง {qd_wh}")
                                qd_codes = []
                            else:
                                pal_dict = {p["code"]: p for p in av_p}
                                qd_codes = st.multiselect(
                                    "เลือกพาเลตที่จะจ่ายออก:",
                                    list(pal_dict),
                                    max_selections=int(sel_q_truck["slots"]),
                                    format_func=lambda x: f"{x} · {pal_dict[x]['name']} ({pal_dict[x]['weight']:,.0f} kg)",
                                    key="qd_form_codes"
                                )

                    btn_send_yard = st.form_submit_button("🚀 บันทึกและส่งรถเข้าคิวเทียบท่าทันที", type="primary", use_container_width=True)
                    if btn_send_yard:
                        if not qd_partner.strip():
                            st.error("กรุณาระบุชื่อคู่ค้าหรือปลายทาง")
                        elif not is_q_in and not qd_codes:
                            st.error("กรุณาเลือกพาเลตอย่างน้อย 1 รายการ")
                        else:
                            try:
                                kind_val = "IN" if is_q_in else "OUT"
                                payload_val = (qd_sku, qd_u, qd_c) if is_q_in else qd_codes
                                new_id = create_new_trip(
                                    kind=kind_val,
                                    truck_id=sel_q_tid,
                                    warehouse=qd_wh,
                                    partner=qd_partner.strip(),
                                    priority=qd_prio,
                                    payload=payload_val
                                )
                                st.toast(f"ส่งรถ {sel_q_tid} เข้าคิวลานเทียบท่าสำเร็จ ({new_id})", icon="🚀")
                                st.success(f"ส่งรถ {sel_q_tid} เข้าคิวเทียบท่าเรียบร้อย (รหัสเที่ยว {new_id})")
                                st.rerun()
                            except Exception as ex:
                                st.error(str(ex))

        st.divider()

        col_add, col_edit = st.columns([1, 1])

        with col_add:
            with st.container(border=True):
                st.markdown("#### ➕ ลงทะเบียนรถคันใหม่ (Register New Truck)")
                st.caption("เพิ่มรถบรรทุกใหม่เข้าสู่ระบบกองรถ (Fleet Registry)")
                with st.form("form_add_truck", clear_on_submit=True):
                    c_a1, c_a2 = st.columns(2)
                    with c_a1:
                        new_id = st.text_input("รหัสรถ (Truck ID)*:", placeholder="เช่น T09, T10").strip().upper()
                        new_name = st.text_input("ชื่อเรียก / รุ่นรถ*:", placeholder="เช่น Isuzu Giga 360")
                        new_plate = st.text_input("ป้ายทะเบียน*:", placeholder="เช่น 71-9988 กทม.")
                        new_type = st.selectbox("ประเภทรถ:", ["6 ล้อตู้แห้ง", "6 ล้อตู้ทึบ", "4 ล้อตู้ทึบ", "10 ล้อตู้ทึบ", "รถพ่วงเทรลเลอร์ 18 ล้อ"])
                    with c_a2:
                        new_carrier = st.text_input("บริษัทขนส่ง / เจ้าของรถ*:", placeholder="เช่น SCG Logistics, Fleet ประจำคลัง")
                        new_driver = st.text_input("ชื่อพนักงานขับรถ*:", placeholder="เช่น สมชาย ใจมั่น")
                        new_phone = st.text_input("เบอร์โทรศัพท์ติดต่อ:", placeholder="เช่น 089-123-4567")
                        c_s1, c_s2 = st.columns(2)
                        new_slots = c_s1.number_input("ความจุพาเลต:", min_value=1, max_value=24, value=8)
                        new_max_w = c_s2.number_input("รับน้ำหนักสูงสุด (กก.):", min_value=500.0, max_value=50000.0, value=8000.0, step=500.0)
                    st.caption("💡 **เกณฑ์มาตรฐาน:** 4 ล้อตู้ทึบ (4 ช่อง / 3,500 kg) · 6 ล้อ (8 ช่อง / 8,000-8,500 kg) · 10 ล้อ (12 ช่อง / 15,000 kg) · เทรลเลอร์ 18 ล้อ (22 ช่อง / 28,000 kg)")

                    submit_add = st.form_submit_button("บันทึกลงทะเบียนรถใหม่", type="primary", use_container_width=True)
                    if submit_add:
                        if not new_id or not new_name or not new_plate:
                            st.error("กรุณาระบุรหัสรถ ชื่อเรียก และป้ายทะเบียนให้ครบถ้วน")
                        else:
                            try:
                                create_or_update_truck(
                                    truck_id=new_id,
                                    name=new_name,
                                    slots=int(new_slots),
                                    max_weight=float(new_max_w),
                                    plate=new_plate,
                                    carrier=new_carrier or "รถร่วมบริการ",
                                    driver_name=new_driver or "ไม่ระบุ",
                                    driver_phone=new_phone or "-",
                                    truck_type=new_type,
                                    is_edit=False
                                )
                                st.toast(f"ลงทะเบียนรถ {new_id} ({new_plate}) สำเร็จ", icon="🚚")
                                st.success(f"ลงทะเบียนรถ {new_id} ({new_plate}) เรียบร้อยแล้ว")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

        with col_edit:
            with st.container(border=True):
                st.markdown("#### ✏️ แก้ไขข้อมูลรถ / ลบรถ (Edit / Delete Truck)")
                st.caption("ปรับปรุงข้อมูลรถที่มีอยู่ หรือลบรถที่ไม่มีเที่ยวค้างอยู่ออกจากระบบ")

                truck_options = {t["id"]: t for t in all_trucks}
                edit_tid = st.selectbox(
                    "เลือกรถที่ต้องการแก้ไขหรือลบ:",
                    list(truck_options),
                    format_func=lambda x: f"{x} · {truck_options[x]['name']} ({truck_options[x].get('plate', '-')}) · {truck_options[x].get('carrier', '-')}"
                )
                target_t = truck_options[edit_tid]

                with st.form("form_edit_truck"):
                    e_c1, e_c2 = st.columns(2)
                    with e_c1:
                        edit_name = st.text_input("ชื่อเรียก / รุ่นรถ:", value=target_t["name"])
                        edit_plate = st.text_input("ป้ายทะเบียน:", value=target_t.get("plate") or "")
                        types_list = ["6 ล้อตู้แห้ง", "6 ล้อตู้ทึบ", "4 ล้อตู้ทึบ", "10 ล้อตู้ทึบ", "รถพ่วงเทรลเลอร์ 18 ล้อ"]
                        type_idx = 0
                        if target_t.get("truck_type") in types_list:
                            type_idx = types_list.index(target_t["truck_type"])
                        edit_type = st.selectbox("ประเภทรถ:", types_list, index=type_idx)
                    with e_c2:
                        edit_carrier = st.text_input("บริษัทขนส่ง / เจ้าของรถ:", value=target_t.get("carrier") or "")
                        edit_driver = st.text_input("ชื่อพนักงานขับรถ:", value=target_t.get("driver_name") or "")
                        edit_phone = st.text_input("เบอร์โทรศัพท์ติดต่อ:", value=target_t.get("driver_phone") or "")
                        e_s1, e_s2 = st.columns(2)
                        edit_slots = e_s1.number_input("ความจุพาเลต:", min_value=1, max_value=24, value=int(target_t["slots"]))
                        edit_max_w = e_s2.number_input("รับน้ำหนักสูงสุด (กก.):", min_value=500.0, max_value=50000.0, value=float(target_t["max_weight"]), step=500.0)
                    st.caption(f"💡 สามารถแก้ไขความจุพาเลตและพิกัดน้ำหนักสูงสุดของรถ `{target_t['id']}` ได้ตามจริง")

                    b1, b2 = st.columns(2)
                    with b1:
                        submit_save = st.form_submit_button("💾 บันทึกการแก้ไข", type="primary", use_container_width=True)
                    with b2:
                        st.caption(f"รหัสรถ: `{target_t['id']}` (ไม่สามารถเปลี่ยนรหัสได้)")

                    if submit_save:
                        try:
                            create_or_update_truck(
                                truck_id=target_t["id"],
                                name=edit_name,
                                slots=int(edit_slots),
                                max_weight=float(edit_max_w),
                                plate=edit_plate,
                                carrier=edit_carrier,
                                driver_name=edit_driver,
                                driver_phone=edit_phone,
                                truck_type=edit_type,
                                is_edit=True
                            )
                            st.toast(f"อัปเดตข้อมูลรถ {target_t['id']} สำเร็จ", icon="✅")
                            st.success(f"อัปเดตข้อมูลรถ {target_t['id']} เรียบร้อยแล้ว")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                # Delete button (outside form)
                if target_t.get("active_trip_id"):
                    st.warning(f"⚠️ รถคันนี้กำลังมีเที่ยวรถค้างอยู่ ({target_t['active_trip_id']}) จึงยังไม่สามารถลบได้")
                else:
                    if st.button(f"🗑️ ลบรถ {target_t['id']} ({target_t['name']}) ออกจากระบบ", use_container_width=True):
                        try:
                            delete_truck_by_id(target_t["id"])
                            st.toast(f"ลบรถ {target_t['id']} ออกจากระบบสำเร็จ", icon="🗑️")
                            st.success(f"ลบรถ {target_t['id']} ออกจากระบบเรียบร้อย")
                            st.rerun()
                        except Exception as ex:
                            st.error(str(ex))


# --- PAGE 3: SMART RECEIVING ---
def page_receiving():
    st.header("Receiving & Inbound")
    st.caption("ตรวจสอบสินค้าประจำช่องเทียบท่า ขนถ่ายลงสู่พื้นที่พักรับ และลงทะเบียนเที่ยวรถขาเข้า")

    tab1, tab2 = st.tabs(["ขนถ่ายสินค้าลงจากรถ (Unload)", "ลงทะเบียนเที่ยวรถขาเข้าใหม่ (New Inbound)"])

    with tab1:
        trips = rows("SELECT s.*, t.name FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.kind = 'IN' AND s.status NOT IN ('DEPARTED', 'CANCELLED')")
        if not trips:
            st.info("ไม่มีเที่ยวรถขาเข้าที่กำลังดำเนินการในระบบ")
            return

        lookup = {t["id"]: t for t in trips}
        sel_id = st.selectbox("เลือกเที่ยวรถขาเข้า:", list(lookup), format_func=lambda x: f"{lookup[x]['id']} · {lookup[x]['name']} ({lookup[x]['status']} · {lookup[x]['warehouse']})")
        trip = lookup[sel_id]

        col_left, col_right = st.columns([1, 1])
        with col_left:
            fig = draw_truck_graphic(trip["id"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_right:
            st.subheader("จุดขนถ่ายสินค้า (Unloading Station)")
            pending = rows("SELECT p.*, pr.name AS product_name FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.inbound_id = ? AND p.status = 'ON_INBOUND' ORDER BY p.inbound_slot ASC", (trip["id"],))
            total_pallets = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ?", (trip["id"],))
            unloaded_cnt = total_pallets - len(pending)

            st.progress(unloaded_cnt / max(total_pallets, 1), text=f"ขนถ่ายแล้ว {unloaded_cnt} / {total_pallets} พาเลต")

            if trip["status"] != "DOCKED":
                st.warning("⚠️ รถคันนี้ยังไม่ได้เข้าเทียบท่า กรุณานำรถเข้าเทียบท่าก่อนเริ่มขนถ่าย")
                if trip["status"] == "WAITING":
                    occupied_docks = {r["dock"] for r in rows("SELECT dock FROM trips WHERE status = 'DOCKED'")}
                    free_docks = [d for d, cfg in DOCKS.items() if cfg["kind"] == trip["kind"] and cfg["warehouse"] == trip["warehouse"] and d not in occupied_docks]
                    if free_docks:
                        target_dock = free_docks[0]
                        if st.button(f"นำรถเข้าเทียบท่าช่อง {target_dock}", type="primary", use_container_width=True):
                            try:
                                execute_action("park", trip["id"], target_dock)
                                st.toast(f"นำรถเข้าเทียบท่าช่อง {target_dock} สำเร็จ", icon="✅")
                                st.success(f"นำรถเข้าเทียบท่าช่อง {target_dock} เรียบร้อยแล้ว")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
            elif pending:
                p_lookup = {p["code"]: p for p in pending}
                with st.form("unload_form"):
                    chosen = st.selectbox("เลือกพาเลตบนรถเพื่อขนถ่ายลง:", list(p_lookup), format_func=lambda x: f"{x} (ช่อง {p_lookup[x]['inbound_slot']}) · {p_lookup[x]['product_name']}")
                    if st.form_submit_button("⬇ ขนถ่ายพาเลตนี้ลงพื้นที่พักรับ", type="primary", use_container_width=True):
                        try:
                            execute_action("receive", chosen, trip["id"])
                            st.toast(f"ขนถ่าย {chosen} ลงพื้นที่พักรับเรียบร้อย", icon="✅")
                            st.success(f"ขนถ่าย {chosen} ลงพื้นที่พักรับเรียบร้อย")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                if st.button("ขนถ่ายสินค้าลงทั้งหมด (Unload All)", use_container_width=True):
                    try:
                        execute_action("receive_all", trip["id"])
                        st.toast("ขนถ่ายสินค้าลงจากรถทั้งหมดสำเร็จ", icon="📥")
                        st.success("ขนถ่ายสินค้าลงจากรถทั้งหมดสำเร็จ")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.success("ขนถ่ายสินค้าลงครบทุกรายการเรียบร้อย พร้อมบันทึกปล่อยรถ")
                if st.button("🚚 บันทึกปล่อยรถออกจากลาน (Depart)", type="primary", use_container_width=True):
                    try:
                        execute_action("depart", trip["id"])
                        st.toast("บันทึกปล่อยรถขาเข้าเรียบร้อย", icon="✅")
                        st.success("บันทึกปล่อยรถออกจากลานเรียบร้อย")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.divider()
        st.subheader("รายการพาเลตในเที่ยวรถ (Shipment Manifest)")
        all_items = rows(
            """
            SELECT p.*, pr.name AS product_name, pr.category 
            FROM pallets p JOIN products pr ON pr.sku = p.sku 
            WHERE p.inbound_id = ? ORDER BY p.inbound_slot ASC
            """,
            (trip["id"],),
        )
        show_pallet_table(all_items)

    with tab2:
        trucks = rows("SELECT * FROM trucks WHERE id NOT IN (SELECT truck_id FROM trips WHERE status NOT IN ('DEPARTED', 'CANCELLED'))")
        prods = rows("SELECT * FROM products")
        if not trucks:
            st.warning("ไม่มีรถว่างสำหรับลงทะเบียนเที่ยวใหม่ กรุณารอรถเที่ยวอื่นเสร็จสิ้น")
        else:
            t_map = {t["id"]: t for t in trucks}
            p_map = {p["sku"]: p for p in prods}
            with st.form("new_inbound_form"):
                a, b = st.columns(2)
                with a:
                    t_id = st.selectbox("เลือกรถบรรทุก:", list(t_map), format_func=lambda x: f"{x} · {t_map[x]['name']} ({t_map[x].get('plate', '-')}) · {t_map[x].get('carrier', '-')} (จุ {t_map[x]['slots']} ช่อง · {t_map[x]['max_weight']:,.0f} kg)")
                    wh = st.selectbox("คลังปลายทาง:", WAREHOUSES)
                    partner = st.text_input("ชื่อผู้ส่งสินค้า (Supplier):", "NORTH SUPPLIER CO.")
                with b:
                    sku = st.selectbox("สินค้า:", list(p_map), format_func=lambda x: f"{x} · {p_map[x]['name']} ({p_map[x]['unit_weight']} kg/หน่วย)")
                    qty = st.number_input("จำนวนหน่วยต่อพาเลต:", 1, 500, 20)
                    cnt = st.number_input("จำนวนพาเลต:", 1, t_map[t_id]["slots"], min(4, t_map[t_id]["slots"]))
                    prio = st.selectbox("ความเร่งด่วน (Priority):", ["High", "Normal", "Low"])

                calc_weight = qty * cnt * p_map[sku]["unit_weight"]
                st.caption(f"น้ำหนักรวมคำนวณ: **{calc_weight:,.0f} kg** / พิกัดรถสูงสุด: **{t_map[t_id]['max_weight']:,.0f} kg**")

                if st.form_submit_button("บันทึกเที่ยวรถขาเข้า", type="primary", use_container_width=True):
                    try:
                        execute_action_id = create_new_trip("IN", t_id, wh, partner, prio, (sku, qty, cnt))
                        st.toast(f"บันทึกเที่ยวรถขาเข้า {execute_action_id} สำเร็จ", icon="🚚")
                        st.success(f"บันทึกเที่ยวรถขาเข้า {execute_action_id} เรียบร้อยแล้ว")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# --- PAGE 4: WAREHOUSE MAPPING ---
def page_warehouse():
    st.header("Warehouse Management & Analytics")
    st.caption("ผังตำแหน่งชั้นจัดเก็บ การวิเคราะห์หมุนเวียนสินค้า (ABC Slotting) และระบบเฝ้าระวังวันหมดอายุ (FEFO)")

    wh = st.radio("เลือกคลังสินค้าที่ต้องการบริหารจัดการ:", WAREHOUSES, horizontal=True)

    tab_layout, tab_analytics, tab_inventory = st.tabs([
        f"🗺️ ผังคลัง & จัดเก็บ (Rack Layout · {wh})",
        f"📊 วิเคราะห์คลัง & ABC Slotting & FEFO ({wh})",
        f"📋 ทะเบียนสต็อกสินค้า (Inventory · {wh})"
    ])

    with tab_layout:
        # Locate Pallet in Warehouse Map
        with st.container(border=True):
            qs_col1, qs_col2 = st.columns([4, 1], vertical_alignment="bottom")
            with qs_col1:
                scan_query = st.text_input("ค้นหาตำแหน่งจัดเก็บพาเลตในผังคลัง:", placeholder="ระบุรหัสพาเลต เช่น PAL-001", key="quick_scan_input")
            with qs_col2:
                if st.button("ตรวจสอบตำแหน่ง", use_container_width=True, type="primary", key="btn_quick_scan"):
                    if scan_query.strip():
                        pallet_detail_dialog(scan_query.strip().upper())
                    else:
                        st.toast("กรุณาระบุรหัสพาเลตที่ต้องการค้นหา", icon="⚠️")

        stage_items = rows(
            """
            SELECT p.*, pr.name AS product_name, pr.category 
            FROM pallets p JOIN products pr ON pr.sku = p.sku 
            WHERE p.warehouse = ? AND p.status = 'RECEIVING' ORDER BY p.code ASC
            """,
            (wh,),
        )

        with st.container(border=True):
            st.subheader("จัดเก็บสินค้าเข้าชั้นวาง (Putaway Station)")
            if not stage_items:
                st.info("ไม่มีสินค้าตกค้างในพื้นที่พักรับของคลังนี้ สินค้าทั้งหมดจัดเก็บบนชั้นวางเรียบร้อย")
            else:
                locs = rows(
                    """
                    SELECT l.id, l.capacity, COUNT(p.code) as used 
                    FROM locations l LEFT JOIN pallets p ON p.location = l.id AND p.status IN ('AVAILABLE', 'RESERVED')
                    WHERE l.warehouse = ? GROUP BY l.id, l.capacity HAVING used < capacity ORDER BY used ASC, l.id ASC
                    """,
                    (wh,),
                )
                if not locs:
                    st.error("ช่องจัดเก็บในคลังนี้เต็มทั้งหมดแล้ว กรุณาเบิกจ่ายสินค้าออกก่อน")
                else:
                    p_map = {item["code"]: item for item in stage_items}
                    l_map = {l["id"]: l for l in locs}

                    c_single, c_batch = st.columns([3, 1], vertical_alignment="bottom")
                    with c_single:
                        with st.form("putaway_form"):
                            c1, c2 = st.columns(2)
                            pal_code = c1.selectbox("เลือกพาเลตจากพื้นที่พักรับ:", list(p_map), format_func=lambda x: f"{x} · {p_map[x]['product_name']}")
                            loc_id = c2.selectbox("เลือกตำแหน่งชั้นจัดเก็บ:", list(l_map), format_func=lambda x: f"{x} (ว่าง {l_map[x]['capacity'] - l_map[x]['used']} ช่อง)")
                            if st.form_submit_button("จัดเก็บเข้าชั้นวาง", type="primary", use_container_width=True):
                                try:
                                    execute_action("putaway", pal_code, loc_id)
                                    st.toast(f"จัดเก็บ {pal_code} เข้าชั้นวาง {loc_id} สำเร็จ", icon="✅")
                                    st.success(f"จัดเก็บพาเลต {pal_code} ไปยัง {loc_id} เรียบร้อยแล้ว")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    with c_batch:
                        if st.button("จัดเก็บทั้งหมดอัตโนมัติ (Putaway All)", use_container_width=True):
                            try:
                                execute_action("putaway_all", wh)
                                st.toast(f"จัดเก็บสินค้าในคลัง {wh} ครบทุกพาเลต", icon="📍")
                                st.success(f"จัดเก็บสินค้าในคลัง {wh} ทั้งหมดเรียบร้อยแล้ว")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

        st.subheader(f"ผังตำแหน่งชั้นวางสินค้า · คลัง {wh}")
        locations = rows(
            """
            SELECT l.id, l.capacity, COUNT(p.code) as used 
            FROM locations l LEFT JOIN pallets p ON p.location = l.id AND p.status IN ('AVAILABLE', 'RESERVED')
            WHERE l.warehouse = ? GROUP BY l.id, l.capacity ORDER BY l.id ASC
            """,
            (wh,),
        )

        all_stored_pallets = rows(
            """
            SELECT p.code, p.location, p.sku, pr.name, p.status 
            FROM pallets p JOIN products pr ON p.sku = pr.sku 
            WHERE p.warehouse = ? AND p.location IS NOT NULL
            """,
            (wh,),
        )
        pallets_by_loc = {}
        for p in all_stored_pallets:
            pallets_by_loc.setdefault(p["location"], []).append(p)

        cols = st.columns(3)
        for idx, loc in enumerate(locations):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**ตำแหน่ง: `{loc['id']}`**")
                    st.progress(loc["used"] / loc["capacity"], text=f"บรรจุ {loc['used']} / {loc['capacity']} พาเลต")
                    stored = pallets_by_loc.get(loc["id"], [])
                    for s in stored:
                        tag = " (จอง)" if s["status"] == "RESERVED" else ""
                        abc_tag = ABC_CLASS.get(s["sku"], {}).get("class", "-")
                        btn_txt = f"📦 [{abc_tag}] {s['code']}{tag} · {s['name'][:8]}.."
                        if st.button(btn_txt, key=f"btn_{s['code']}", use_container_width=True):
                            pallet_detail_dialog(s["code"])
                    if not stored:
                        st.caption("— ช่องจัดเก็บว่าง —")

    with tab_analytics:
        st.subheader(f"การวิเคราะห์ประสิทธิภาพคลังสินค้า · {wh}")
        st.caption("วิเคราะห์การจัดตำแหน่งสินค้าตามความเร็วหมุนเวียน (ABC Slotting) และระบบแจ้งเตือนวันหมดอายุ (FEFO)")

        # Summary KPIs for this warehouse
        total_loc_cap = scalar("SELECT SUM(capacity) FROM locations WHERE warehouse = ?", (wh,)) or 1
        stored_cnt = scalar("SELECT COUNT(*) FROM pallets WHERE warehouse = ? AND status IN ('AVAILABLE', 'RESERVED') AND location IS NOT NULL", (wh,))
        stored_wt = scalar("SELECT SUM(weight) FROM pallets WHERE warehouse = ? AND status IN ('AVAILABLE', 'RESERVED')", (wh,)) or 0
        all_wh_pallets = rows("SELECT p.*, pr.name, pr.category FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.warehouse = ?", (wh,))

        near_exp_cnt = sum(1 for p in all_wh_pallets if get_days_until_expiry(p.get("exp_date")) < 30)
        watch_exp_cnt = sum(1 for p in all_wh_pallets if 30 <= get_days_until_expiry(p.get("exp_date")) < 60)

        a_cnt = sum(1 for p in all_wh_pallets if ABC_CLASS.get(p["sku"], {}).get("class") == "A")
        b_cnt = sum(1 for p in all_wh_pallets if ABC_CLASS.get(p["sku"], {}).get("class") == "B")
        c_cnt = sum(1 for p in all_wh_pallets if ABC_CLASS.get(p["sku"], {}).get("class") == "C")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("อัตราการใช้พื้นที่คลัง", f"{(stored_cnt / total_loc_cap) * 100:.0f}%", f"{stored_cnt}/{total_loc_cap} ช่อง")
        kpi2.metric("น้ำหนักสต็อกรวม", f"{stored_wt:,.0f} kg", f"{len(all_wh_pallets)} พาเลต")
        kpi3.metric("สินค้า Fast-Moving (A)", f"{a_cnt} พาเลต", f"B: {b_cnt} | C: {c_cnt}")
        kpi4.metric("สินค้าใกล้หมดอายุ (<30 วัน)", f"{near_exp_cnt} พาเลต", f"เฝ้าระวัง: {watch_exp_cnt}", delta_color="inverse")

        st.divider()

        # ============================================================
        # SECTION 1: ABC VELOCITY SLOTTING & RELOCATION ENGINE
        # ============================================================
        st.markdown("#### ⚡ 1. การวิเคราะห์การจัดตำแหน่งสินค้า (ABC Velocity Slotting Matrix)")
        st.caption("จัดเรียงสินค้าตามอัตราความถี่ในการเบิกจ่าย: สินค้าหมุนเร็ว (Fast-Moving) ควรอยู่ใกล้ทางออก/ช่องเทียบท่าที่สุดเพื่อลดระยะทางการเคลื่อนย้าย")

        abc_c1, abc_c2 = st.columns([1, 1.2])

        with abc_c1:
            with st.container(border=True):
                st.markdown("##### 🎯 สัดส่วนความเร็วหมุนเวียน (ABC Velocity Share)")
                fig_abc, ax_abc = plt.subplots(figsize=(4.5, 3.2), dpi=130)
                fig_abc.patch.set_facecolor("#FFFFFF")
                ax_abc.set_facecolor("#FFFFFF")

                abc_labels = ["Class A (เร็ว)", "Class B (ปานกลาง)", "Class C (ช้า)"]
                abc_counts = [a_cnt, b_cnt, c_cnt]
                abc_colors = ["#059669", "#2563EB", "#64748B"]

                if sum(abc_counts) > 0:
                    wedges, texts, autotexts = ax_abc.pie(
                        abc_counts, labels=None, colors=abc_colors, autopct="%1.0f%%",
                        startangle=90, pctdistance=0.72,
                        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=2)
                    )
                    for t in autotexts:
                        t.set_fontsize(9)
                        t.set_color("white")
                        t.set_fontweight("bold")
                    ax_abc.legend(wedges, abc_labels, loc="center left", bbox_to_anchor=(0.85, 0.5), fontsize=8.5, frameon=False)
                    ax_abc.text(0, 0.11, "สต็อกคลัง", ha="center", va="center", fontsize=9.5, color="#64748B", fontweight="bold")
                    ax_abc.text(0, -0.13, f"{sum(abc_counts)} พาเลต", ha="center", va="center", fontsize=12.5, color="#0F172A", fontweight="bold")
                    fig_abc.tight_layout()
                    st.pyplot(fig_abc, use_container_width=True)
                    
                    # 3 Symmetrical Class Badges Underneath
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f"""<div style="border-left: 3px solid #059669; padding: 4px 8px; background: #F8FAFC; border-radius: 4px;">
                                <div style="font-size: 10px; color: #64748B;">Class A (เร็ว)</div>
                                <strong style="font-size: 13px; color: #0F172A;">{a_cnt} <span style="font-size: 10px; font-weight: normal; color: #64748B;">พาเลต</span></strong>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    with m2:
                        st.markdown(
                            f"""<div style="border-left: 3px solid #2563EB; padding: 4px 8px; background: #F8FAFC; border-radius: 4px;">
                                <div style="font-size: 10px; color: #64748B;">Class B (กลาง)</div>
                                <strong style="font-size: 13px; color: #0F172A;">{b_cnt} <span style="font-size: 10px; font-weight: normal; color: #64748B;">พาเลต</span></strong>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    with m3:
                        st.markdown(
                            f"""<div style="border-left: 3px solid #64748B; padding: 4px 8px; background: #F8FAFC; border-radius: 4px;">
                                <div style="font-size: 10px; color: #64748B;">Class C (ช้า)</div>
                                <strong style="font-size: 13px; color: #0F172A;">{c_cnt} <span style="font-size: 10px; font-weight: normal; color: #64748B;">พาเลต</span></strong>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("ไม่มีข้อมูลสินค้าในคลังนี้")
                plt.close(fig_abc)

        with abc_c2:
            with st.container(border=True):
                st.markdown("##### 📍 กฎการจัดวางตำแหน่งที่เหมาะสม (Optimal Slotting Rules)")
                st.markdown(
                    """
                    - 🟢 **Class A (Fast-Moving):** ควรอยู่ที่ชั้น **R01, R02** (ตำแหน่งหน้าสุด ใกล้จุดเบิกจ่าย)
                    - 🔵 **Class B (Medium-Moving):** ควรอยู่ที่ชั้น **R03, R04** (ตำแหน่งช่วงกลางคลัง)
                    - ⚪ **Class C (Slow-Moving):** ควรอยู่ที่ชั้น **R05, R06** (ตำแหน่งแถวด้านในสุด)
                    """
                )

                # Check for Sub-optimal Pallets
                misplaced = []
                for p in all_wh_pallets:
                    if p.get("location") and p.get("status") in ("AVAILABLE", "RESERVED"):
                        sku_rule = ABC_CLASS.get(p["sku"])
                        if sku_rule:
                            cur_rack = p["location"].split("-")[-1] if "-" in p["location"] else p["location"]
                            if cur_rack not in sku_rule["opt_racks"]:
                                misplaced.append({
                                    "code": p["code"],
                                    "sku": p["sku"],
                                    "name": p["name"],
                                    "class": sku_rule["class"],
                                    "cur_loc": p["location"],
                                    "opt_racks": [f"{wh}-{r}" for r in sku_rule["opt_racks"]]
                                })

                if misplaced:
                    st.warning(f"⚠️ ตรวจพบ **{len(misplaced)} พาเลต** ที่จัดเก็บบนชั้นวางไม่ตรงกับความเร็วหมุนเวียน (Sub-optimal)")
                else:
                    st.success("✅ สินค้าทั้งหมดจัดวางในตำแหน่งที่เหมาะสมตามหลัก ABC Slotting เรียบร้อยแล้ว")

        # Slotting Relocation Action Box
        if misplaced:
            st.markdown("##### 🔄 คำแนะนำและระบบย้ายพาเลตให้ตรงเกณฑ์ (Slotting Relocation Tool)")
            with st.container(border=True):
                for mp in misplaced[:4]:
                    r_c1, r_c2, r_c3 = st.columns([1.5, 2, 1.2], vertical_alignment="center")
                    with r_c1:
                        st.markdown(f"**{mp['code']}** ({mp['name']})")
                        st.caption(f"SKU: {mp['sku']} · Class {mp['class']}")
                    with r_c2:
                        st.markdown(f"ปัจจุบันอยู่: `{mp['cur_loc']}` ➔ ชั้นแนะนำ: `{' หรือ '.join(mp['opt_racks'])}`")
                    with r_c3:
                        # Find an available rack among optimal racks
                        avail_opt_loc = None
                        for opt_l in mp["opt_racks"]:
                            occ = scalar("SELECT COUNT(*) FROM pallets WHERE location = ? AND status IN ('AVAILABLE', 'RESERVED')", (opt_l,))
                            cap = scalar("SELECT capacity FROM locations WHERE id = ?", (opt_l,)) or 2
                            if occ < cap:
                                avail_opt_loc = opt_l
                                break
                        
                        if avail_opt_loc:
                            if st.button(f"ย้ายไป {avail_opt_loc}", key=f"btn_reloc_{mp['code']}", type="primary", use_container_width=True):
                                try:
                                    execute_action("relocate", mp["code"], avail_opt_loc)
                                    st.toast(f"ย้าย {mp['code']} ไปยัง {avail_opt_loc} สำเร็จ", icon="✅")
                                    st.success(f"ย้าย {mp['code']} ไปยัง {avail_opt_loc} เรียบร้อยแล้ว")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.caption("ชั้นแนะนำเต็ม")

        st.divider()

        # ============================================================
        # SECTION 2: FIFO / FEFO EXPIRY TRACKING DASHBOARD
        # ============================================================
        st.markdown("#### ⏳ 2. ระบบตรวจสอบและเฝ้าระวังอายุสินค้า (FEFO Expiry Tracker)")
        st.caption("บริหารสต็อกตามหลัก First-Expired, First-Out (FEFO) เพื่อลดความเสี่ยงสินค้าค้างสต็อกหรือเสื่อมสภาพ")

        fefo_items = []
        for p in all_wh_pallets:
            days = get_days_until_expiry(p.get("exp_date"))
            if days < 30:
                status_txt = "🔴 วิกฤต (< 30 วัน)"
            elif days < 60:
                status_txt = "🟡 เฝ้าระวัง (30-60 วัน)"
            else:
                status_txt = "🟢 ปกติ (> 60 วัน)"

            fefo_items.append({
                "รหัสพาเลต": p["code"],
                "ล็อตสินค้า": p.get("lot_number") or "-",
                "SKU": p["sku"],
                "ชื่อสินค้า": p["name"],
                "วันผลิต (MFG)": p.get("mfg_date") or "-",
                "วันหมดอายุ (EXP)": p.get("exp_date") or "-",
                "คงเหลือ (วัน)": days,
                "สถานะความเสี่ยง": status_txt,
                "ตำแหน่งชั้น": p.get("location") or "พื้นที่พัก/รถ",
                "สถานะงาน": PALLET_STATUS.get(p["status"], p["status"])
            })

        # Sort FEFO: earliest expiry first
        fefo_items.sort(key=lambda x: x["คงเหลือ (วัน)"])
        st.dataframe(pd.DataFrame(fefo_items), use_container_width=True, hide_index=True)

        st.divider()

        # ============================================================
        # SECTION 3: RACK UTILIZATION HEATMAP & CAPACITY BAR
        # ============================================================
        st.markdown(f"#### 📊 3. กราฟเปรียบเทียบการใช้งานรายชั้นวาง (Rack Utilization · {wh})")
        wh_locs = rows("SELECT id, capacity FROM locations WHERE warehouse = ? ORDER BY id ASC", (wh,))
        if wh_locs:
            rack_labels = [l["id"].split("-")[-1] for l in wh_locs]
            rack_caps = [l["capacity"] for l in wh_locs]
            rack_useds = [scalar("SELECT COUNT(*) FROM pallets WHERE location = ? AND status IN ('AVAILABLE', 'RESERVED')", (l["id"],)) for l in wh_locs]

            fig_rack, ax_rack = plt.subplots(figsize=(7.5, 3.2), dpi=140)
            fig_rack.patch.set_facecolor("#FFFFFF")
            ax_rack.set_facecolor("#FFFFFF")

            x_indices = range(len(rack_labels))
            bar_width = 0.45

            ax_rack.bar(x_indices, rack_caps, color="#F1F5F9", edgecolor="#CBD5E1", label="ความจุชั้นวาง", width=bar_width)
            bars_used = ax_rack.bar(x_indices, rack_useds, color="#3B82F6", edgecolor="#1E293B", label="จัดเก็บจริง", width=bar_width)

            for idx, (u, c) in enumerate(zip(rack_useds, rack_caps)):
                ax_rack.text(idx, u + 0.1, f"{u}/{c}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")

            ax_rack.set_xticks(x_indices)
            ax_rack.set_xticklabels([f"ชั้น {r}" for r in rack_labels], fontsize=9.5, fontweight="bold")
            ax_rack.set_ylabel("จำนวนพาเลต", fontsize=8.5)
            ax_rack.set_ylim(0, max(rack_caps) + 1.5)
            ax_rack.legend(loc="upper right", fontsize=8.5, frameon=True)
            ax_rack.spines["top"].set_visible(False)
            ax_rack.spines["right"].set_visible(False)
            fig_rack.tight_layout()
            st.pyplot(fig_rack, use_container_width=True)
            plt.close(fig_rack)

    with tab_inventory:
        st.subheader(f"รายการสต็อกสินค้าทั้งหมด · คลัง {wh}")
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            st_filter = st.selectbox("กรองตามสถานะ:", ["ทั้งหมด", "พร้อมใช้ (AVAILABLE)", "จองแล้ว (RESERVED)", "พักรับ (RECEIVING)"])
        with f_col2:
            kw = st.text_input("ค้นหารหัสพาเลต, SKU หรือชื่อสินค้า:", "")

        query = """
            SELECT p.*, pr.name AS product_name, pr.category 
            FROM pallets p JOIN products pr ON pr.sku = p.sku 
            WHERE p.warehouse = ?
        """
        q_params = [wh]
        if st_filter == "พร้อมใช้ (AVAILABLE)":
            query += " AND p.status = 'AVAILABLE'"
        elif st_filter == "จองแล้ว (RESERVED)":
            query += " AND p.status = 'RESERVED'"
        elif st_filter == "พักรับ (RECEIVING)":
            query += " AND p.status = 'RECEIVING'"

        if kw.strip():
            query += " AND (p.code LIKE ? OR p.sku LIKE ? OR pr.name LIKE ?)"
            q_params.extend([f"%{kw.strip()}%", f"%{kw.strip()}%", f"%{kw.strip()}%"])

        query += " ORDER BY p.code ASC"
        filtered_items = rows(query, tuple(q_params))
        show_pallet_table(filtered_items)


# --- PAGE 5: SHIPPING & OUTBOUND ---
def page_shipping():
    st.header("Shipping & Outbound Operations")
    st.caption("บริหารจัดการคำสั่งเบิกจ่าย หยิบสินค้า โหลดขึ้นรถบรรทุก และปล่อยรถออก")

    tab1, tab2 = st.tabs(["จัดการเที่ยวรถขาออก (Outbound Operations)", "สร้างใบเบิกจ่ายและจองสินค้า (New Order)"])

    with tab1:
        trips = rows("SELECT s.*, t.name FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.kind = 'OUT' AND s.status NOT IN ('DEPARTED', 'CANCELLED')")
        if not trips:
            st.info("ไม่มีเที่ยวรถขาออกที่กำลังดำเนินงานในระบบ")
            return

        lookup = {t["id"]: t for t in trips}
        sel_id = st.selectbox("เลือกเที่ยวรถขาออก:", list(lookup), format_func=lambda x: f"{lookup[x]['id']} · {lookup[x]['name']} ({lookup[x]['status']} · {lookup[x]['warehouse']})")
        trip = lookup[sel_id]

        c_left, c_right = st.columns([1, 1])
        with c_left:
            fig = draw_truck_graphic(trip["id"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            st.subheader("สถานะการเบิกจ่ายและโหลดสินค้า (Outbound Station)")
            manifest = rows("SELECT p.*, pr.name AS product_name, pr.category FROM pallets p JOIN products pr ON p.sku = p.sku WHERE p.outbound_id = ?", (trip["id"],))
            reserved = [m for m in manifest if m["status"] == "RESERVED"]
            picked = [m for m in manifest if m["status"] == "PICKED"]
            loaded = [m for m in manifest if m["status"] == "LOADED"]

            m1, m2, m3 = st.columns(3)
            m1.metric("1. รอหยิบ (Reserved)", len(reserved))
            m2.metric("2. พักจ่าย (Picked)", len(picked))
            m3.metric("3. บนรถ (Loaded)", len(loaded))

            sub_t1, sub_t2, sub_t3 = st.tabs(["1. หยิบสินค้า (Pick)", "2. โหลดขึ้นรถ (Load)", "3. ปรับลด/นำสินค้าลง (Unload)"])

            with sub_t1:
                if not reserved:
                    st.success("หยิบสินค้าครบทุกรายการแล้ว (พร้อมสำหรับขั้นตอนการโหลด)")
                else:
                    r_map = {r["code"]: r for r in reserved}
                    with st.form("pick_form"):
                        p_code = st.selectbox("เลือกพาเลตที่ต้องหยิบ:", list(r_map), format_func=lambda x: f"{x} · ชั้น {r_map[x]['location']} · {r_map[x]['product_name']}")
                        if st.form_submit_button("หยิบสินค้าไปพื้นที่พักจ่าย", type="primary", use_container_width=True):
                            try:
                                execute_action("pick", p_code, trip["id"])
                                st.toast(f"หยิบพาเลต {p_code} ไปยังพื้นที่พักจ่ายเรียบร้อย", icon="✅")
                                st.success(f"หยิบพาเลต {p_code} สำเร็จ")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    if st.button("หยิบสินค้าทั้งหมด (Pick All)", use_container_width=True):
                        try:
                            execute_action("pick_all", trip["id"])
                            st.toast("หยิบสินค้าทั้งหมดเข้าสู่พื้นที่พักจ่ายเรียบร้อย", icon="✅")
                            st.success("หยิบสินค้าทั้งหมดเข้าสู่พื้นที่พักจ่ายเรียบร้อย")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            with sub_t2:
                if trip["status"] != "DOCKED":
                    st.warning("⚠️ รถยังไม่ได้เข้าเทียบท่าที่ช่องจ่ายสินค้า กรุณานำรถเข้าเทียบท่าที่หน้า Dock Yard ก่อน")
                elif not picked:
                    st.info("ไม่มีสินค้าในพื้นที่พักจ่าย (สินค้าอาจยังรอหยิบ หรือโหลดขึ้นรถครบแล้ว)")
                else:
                    pk_map = {p["code"]: p for p in picked}
                    with st.form("load_form"):
                        l_code = st.selectbox("เลือกพาเลตพร้อมโหลด:", list(pk_map), format_func=lambda x: f"{x} · {pk_map[x]['product_name']} ({pk_map[x]['weight']} kg)")
                        if st.form_submit_button("โหลดขึ้นรถบรรทุก", type="primary", use_container_width=True):
                            try:
                                execute_action("load", l_code, trip["id"])
                                st.toast(f"โหลดพาเลต {l_code} ขึ้นรถบรรทุกสำเร็จ", icon="✅")
                                st.success(f"โหลดพาเลต {l_code} ขึ้นรถสำเร็จ")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    if st.button("โหลดสินค้าขึ้นรถทั้งหมด (Load All)", use_container_width=True):
                        try:
                            execute_action("load_all", trip["id"])
                            st.toast("โหลดสินค้าขึ้นรถครบถ้วนทุกรายการ", icon="📦")
                            st.success("โหลดสินค้าขึ้นรถครบถ้วน")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            with sub_t3:
                if not loaded:
                    st.info("ยังไม่มีพาเลตที่โหลดขึ้นรถ")
                else:
                    ld_map = {l["code"]: l for l in loaded}
                    with st.form("unload_form_outbound"):
                        u_code = st.selectbox("เลือกพาเลตบนรถที่ต้องการนำลง:", list(ld_map), format_func=lambda x: f"{x} (ช่อง {ld_map[x]['load_slot']}) · {ld_map[x]['product_name']}")
                        if st.form_submit_button("นำสินค้าลงจากรถกลับสู่พื้นที่พักจ่าย", use_container_width=True):
                            try:
                                execute_action("unload", u_code, trip["id"])
                                st.toast(f"นำพาเลต {u_code} ลงจากรถกลับสู่พื้นที่พักจ่าย", icon="📦")
                                st.success(f"นำพาเลต {u_code} ลงจากรถกลับสู่พื้นที่พักจ่ายเรียบร้อย")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

            if trip["status"] == "DOCKED" and len(loaded) == len(manifest) and len(manifest) > 0:
                if st.button("🚚 บันทึกปล่อยรถขาออก (Depart)", type="primary", use_container_width=True):
                    try:
                        execute_action("depart", trip["id"])
                        st.toast("บันทึกปล่อยรถขาออกเรียบร้อย", icon="🚚")
                        st.success("ปล่อยรถขาออกเรียบร้อย (Shipment Completed)")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.divider()
        col_mf1, col_mf2 = st.columns([3, 1], vertical_alignment="bottom")
        with col_mf1:
            st.subheader("รายการสินค้าในใบสั่งจ่าย (Order Manifest)")
        with col_mf2:
            if st.button("พิมพ์ใบส่งของ (Bill of Lading)", key="btn_bol_shipping", use_container_width=True):
                delivery_note_dialog(trip["id"])
        show_pallet_table(manifest)

    with tab2:
        wh_out = st.selectbox("เลือกคลังต้นทาง:", WAREHOUSES, key="wh_out_sel")
        avail_stock = rows(
            """
            SELECT p.*, pr.name AS product_name, pr.category 
            FROM pallets p JOIN products pr ON p.sku = pr.sku 
            WHERE p.warehouse = ? AND p.status = 'AVAILABLE' ORDER BY p.code ASC
            """,
            (wh_out,),
        )
        free_trucks = rows("SELECT * FROM trucks WHERE id NOT IN (SELECT truck_id FROM trips WHERE status NOT IN ('DEPARTED', 'CANCELLED'))")

        if not avail_stock:
            st.warning("ไม่มีสินค้าพร้อมใช้ในคลังนี้สำหรับจัดส่ง")
        elif not free_trucks:
            st.warning("ไม่มีรถว่างสำหรับจัดเที่ยวขนส่ง กรุณารอรถคันอื่นเสร็จสิ้น")
        else:
            s_map = {s["code"]: s for s in avail_stock}
            t_map = {t["id"]: t for t in free_trucks}
            with st.form("new_outbound_form"):
                c1, c2 = st.columns(2)
                with c1:
                    t_id = st.selectbox("เลือกรถบรรทุกขนส่ง:", list(t_map), format_func=lambda x: f"{x} · {t_map[x]['name']} ({t_map[x].get('plate', '-')}) · {t_map[x].get('carrier', '-')} (จุ {t_map[x]['slots']} ช่อง · {t_map[x]['max_weight']:,.0f} kg)")
                    partner = st.text_input("ปลายทาง / สาขา:", "สาขาเซ็นทรัลเวิลด์")
                with c2:
                    prio = st.selectbox("ความเร่งด่วน:", ["High", "Normal", "Low"], key="out_prio")
                    codes = st.multiselect("เลือกพาเลตสินค้าที่จะจ่ายออก:", list(s_map), format_func=lambda x: f"{x} · {s_map[x]['product_name']} ({s_map[x]['weight']} kg)")

                selected_weight = sum(s_map[c]["weight"] for c in codes)
                st.caption(f"เลือก: **{len(codes)} / {t_map[t_id]['slots']} ช่อง** · น้ำหนักรวม: **{selected_weight:,.0f} / {t_map[t_id]['max_weight']:,.0f} kg**")

                if st.form_submit_button("สร้างใบสั่งจ่ายและจองสินค้า", type="primary", use_container_width=True):
                    try:
                        execute_id = create_new_trip("OUT", t_id, wh_out, partner, prio, codes)
                        st.toast(f"สร้างใบสั่งจ่ายและจองสต็อก {execute_id} สำเร็จ", icon="📦")
                        st.success(f"สร้างใบสั่งจ่ายและจองสต็อกสำเร็จ (รหัส {execute_id})")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# --- PAGE 6: ACTIVITY LOGS ---

def page_barcode_terminal():
    st.header("Barcode Terminal")
    st.caption("สถานีสแกนเนอร์บาร์โค้ดสำหรับบันทึกการรับเข้า จัดเก็บ หยิบ และโหลดสินค้า รองรับทั้งเครื่องสแกนเนอร์ USB/Bluetooth และกล้อง")

    # Session state for scan history
    if "barcode_scan_history" not in st.session_state:
        st.session_state.barcode_scan_history = []
    if "barcode_last_result" not in st.session_state:
        st.session_state.barcode_last_result = None

    # Step 1: Metrics Snapshot
    inbound_count = scalar("SELECT COUNT(*) FROM pallets p JOIN trips s ON s.id = p.inbound_id WHERE s.status = 'DOCKED' AND p.status = 'ON_INBOUND'")
    receiving_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RECEIVING'")
    reserved_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RESERVED'")
    picked_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'PICKED'")
    avail_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'AVAILABLE'")

    # Metrics row
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    m_c1.metric("1. รอรับเข้า (Dock)", f"{inbound_count} พาเลต")
    m_c2.metric("2. รอจัดเก็บ (Staging)", f"{receiving_count} พาเลต")
    m_c3.metric("3. รอหยิบ (Picking)", f"{reserved_count} พาเลต")
    m_c4.metric("4. รอขึ้นรถ (Loading)", f"{picked_count} พาเลต")
    m_c5.metric("5. พร้อมใช้บนชั้น", f"{avail_count} พาเลต")

    # Step 2: Segmented Mode Selector
    mode_options = [
        f"📥 รับสินค้าเข้า ({inbound_count})",
        f"📦 จัดเก็บเข้าชั้น ({receiving_count})",
        f"🛒 หยิบสินค้า ({reserved_count})",
        f"🚚 โหลดขึ้นรถ ({picked_count})",
        "🔍 ตรวจสอบพิกัด (Lookup)"
    ]

    selected_mode = st.segmented_control(
        "ขั้นตอนการทำงานของสถานีสแกนเนอร์:",
        mode_options,
        default=mode_options[0],
    ) or mode_options[0]

    # Determine execution behavior & query based on selected mode
    if "📥 รับสินค้าเข้า" in selected_mode:
        current_mode = "RECEIVE"
        task_title = "ขั้นตอน: ขนถ่ายสินค้าลงจากรถขาเข้า (Inbound Receiving)"
        task_desc = "สแกนพาเลตบนรถที่เทียบท่า เพื่อย้ายเข้าสู่พื้นที่พักรับสินค้า (Staging Area)"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category, '1. รอขนถ่ายรับเข้า' AS stage_label FROM pallets p JOIN trips s ON s.id = p.inbound_id JOIN products pr ON pr.sku = p.sku WHERE s.status = 'DOCKED' AND p.status = 'ON_INBOUND' ORDER BY p.inbound_slot ASC"
    elif "📦 จัดเก็บเข้าชั้น" in selected_mode:
        current_mode = "PUTAWAY"
        task_title = "ขั้นตอน: จัดเก็บสินค้าเข้าชั้นวาง (Putaway to Rack)"
        task_desc = "สแกนพาเลตจากพื้นที่พักรับ เพื่อนำขึ้นจัดเก็บบนชั้นวางสินค้าที่ว่างอยู่"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category, '2. รอจัดเก็บบนชั้น' AS stage_label FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'RECEIVING' ORDER BY p.code ASC"
    elif "🛒 หยิบสินค้า" in selected_mode:
        current_mode = "PICK"
        task_title = "ขั้นตอน: หยิบสินค้าตามใบสั่งจ่าย (Picking)"
        task_desc = "สแกนพาเลตที่ถูกจองตามคำสั่งเบิก เพื่อย้ายจากชั้นวางมายังจุดเตรียมจ่ายสินค้า"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category, '3. รอหยิบสินค้า' AS stage_label FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'RESERVED' ORDER BY p.code ASC"
    elif "🚚 โหลดขึ้นรถ" in selected_mode:
        current_mode = "LOAD"
        task_title = "ขั้นตอน: โหลดสินค้าขึ้นรถบรรทุก (Outbound Loading)"
        task_desc = "สแกนพาเลตที่จัดเตรียมไว้ เพื่อบรรจุขึ้นช่องบรรทุกบนรถขาออกที่เข้าเทียบท่าแล้ว"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category, '4. รอโหลดขึ้นรถ' AS stage_label FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'PICKED' ORDER BY p.code ASC"
    else:
        current_mode = "LOOKUP"
        task_title = "ขั้นตอน: ตรวจสอบข้อมูลและพิกัดพาเลต (Pallet Lookup)"
        task_desc = "สแกนบาร์โค้ดเพื่อตรวจสอบตำแหน่ง สเปกสินค้า น้ำหนัก และประวัติเที่ยวรถ โดยไม่เปลี่ยนสถานะ"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category, 'ตรวจสอบข้อมูล' AS stage_label FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status NOT IN ('SHIPPED') ORDER BY p.code ASC LIMIT 10"

    st.divider()

    # Step 3: Dual-Pane Interactive Workspace
    col_scanner, col_pallets = st.columns([1, 1.1])

    with col_scanner:
        st.subheader(task_title)
        st.caption(task_desc)

        # Barcode Gun Input Form
        with st.form("barcode_gun_terminal_form", clear_on_submit=True):
            scan_code_input = st.text_input(
                "รหัสบาร์โค้ด / พาเลต:",
                placeholder="ยิงบาร์โค้ดด้วยเครื่องสแกน หรือพิมพ์ เช่น PAL-001 แล้วกด Enter",
                key="terminal_gun_input"
            )
            c_sub1, c_sub2 = st.columns([1.2, 1])
            with c_sub1:
                submit_scan = st.form_submit_button("⚡ บันทึกการสแกน (Enter)", type="primary", use_container_width=True)
            with c_sub2:
                submit_lookup = st.form_submit_button("🔍 ดูป้ายและพิกัด (Tag)", use_container_width=True)

        st.caption("💡 **Tips:** เมื่อใช้เครื่องสแกนบาร์โค้ด (USB/Bluetooth Gun) ระบบจะประมวลผลและล้างช่องพิมพ์เพื่อรอรับการยิงพาเลตถัดไปได้ทันที")

        # Camera Option with OpenCV
        with st.expander("📷 สแกนผ่านกล้องถ่ายภาพ (Camera Scanner)"):
            st.caption("เล็งกล้องไปที่ป้ายบาร์โค้ดของพาเลตเพื่อถอดรหัสอัตโนมัติ")
            cam_image = st.camera_input("ถ่ายภาพบาร์โค้ด", key="cam_barcode_reader")
            if cam_image is not None:
                decoded = decode_barcode_image(cam_image.getvalue())
                if decoded:
                    st.success(f"ตรวจพบบาร์โค้ด: **{decoded}**")
                    if st.button(f"⚡ บันทึกการสแกน {decoded}", type="primary", key="btn_apply_camera_scan"):
                        scan_code_input = decoded
                        submit_scan = True
                else:
                    st.warning("ไม่พบบาร์โค้ดในภาพ กรุณาถือกล้องให้นิ่งและมีแสงสว่างชัดเจน")

        # Handle Lookup button
        if submit_lookup and scan_code_input.strip():
            pallet_detail_dialog(scan_code_input.strip().upper())

        # Handle Scan Action
        if submit_scan and scan_code_input.strip():
            target_c = scan_code_input.strip().upper()
            try:
                res = execute_barcode_scan(target_c, action_mode=current_mode)
                st.session_state.barcode_last_result = res
                st.session_state.barcode_scan_history.insert(0, {
                    "เวลา": timestamp(),
                    "รหัสบาร์โค้ด": res["code"],
                    "กิจกรรม": res["action"],
                    "สถานะเดิม": PALLET_STATUS.get(res["prev_status"], res["prev_status"]),
                    "สถานะใหม่": PALLET_STATUS.get(res["new_status"], res["new_status"]),
                    "ข้อความ": res["message"]
                })
                st.toast(f"บันทึกสำเร็จ: {res['message']}", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"การสแกนไม่สำเร็จ: {str(e)}")

        # Last Scanned Result Display (Handheld PDA Card)
        last_res = st.session_state.barcode_last_result
        if last_res:
            p_info = last_res["pallet"]
            has_changed = (last_res["prev_status"] != last_res["new_status"])
            badge_color = "#059669" if has_changed else "#2563EB"
            
            with st.container(border=True):
                # PDA Beep Sound Synthesis (Offline HTML5 Web Audio API)
                audio_script = """
                <script>
                (function() {
                    try {
                        var AudioContext = window.AudioContext || window.webkitAudioContext;
                        if (AudioContext) {
                            var ctx = new AudioContext();
                            var osc = ctx.createOscillator();
                            var gain = ctx.createGain();
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            osc.type = 'sine';
                            osc.frequency.setValueAtTime(1046.5, ctx.currentTime);
                            gain.gain.setValueAtTime(0.08, ctx.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
                            osc.start();
                            osc.stop(ctx.currentTime + 0.1);
                        }
                    } catch(e) {}
                })();
                </script>
                """
                st.html(audio_script)

                st.markdown(
                    f"""<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #E2E8F0; padding-bottom:6px;'>
                        <span style='font-size:12px; font-weight:800; color:{badge_color};'>📱 ผลการสแกนล่าสุด (PDA Response)</span>
                        <span style='font-size:10px; background:#F1F5F9; color:#475569; padding:2px 6px; border-radius:4px; font-family:monospace;'>{last_res['action']}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                r_c1, r_c2 = st.columns([1, 1.4])
                with r_c1:
                    st.markdown(generate_barcode_svg(last_res["code"]), unsafe_allow_html=True)
                with r_c2:
                    st.markdown(f"**รหัส:** `{last_res['code']}`")
                    st.markdown(f"**สินค้า:** {p_info.get('name', p_info.get('sku', ''))}")
                    
                    if has_changed:
                        prev_th = PALLET_STATUS.get(last_res['prev_status'], last_res['prev_status'])
                        new_th = PALLET_STATUS.get(last_res['new_status'], last_res['new_status'])
                        st.markdown(f"**การเคลื่อนย้าย:** `{prev_th}` ➔ **`{new_th}`**")
                    else:
                        st.markdown(f"**สถานะ:** `{PALLET_STATUS.get(last_res['new_status'], last_res['new_status'])}`")
                    
                    loc_desc = p_info.get('location') or ('บนรถบรรทุก' if p_info.get('load_slot') or p_info.get('inbound_slot') else 'พื้นที่พักสินค้า')
                    st.caption(f"ตำแหน่ง: {loc_desc} · คลัง {p_info.get('warehouse', '-')}")

                st.success(f"บันทึกเรียบร้อย: {last_res['message']}")
                if st.button("🏷️ ดูป้ายกำกับสินค้า (Digital Pallet Tag)", key="btn_view_tag_last_scan", use_container_width=True):
                    pallet_detail_dialog(last_res["code"])

    with col_pallets:
        st.subheader("รายการพาเลตรอสแกน (Ready to Scan)")
        st.caption("สามารถยิงบาร์โค้ดด้วยเครื่องสแกน หรือกดปุ่ม ⚡ สแกน ด้านขวาของแต่ละรายการ")

        waiting_pallets = rows(target_query)
        if not waiting_pallets:
            st.success(f"ดำเนินการครบถ้วน ไม่มีรายการพาเลตรอสแกนในโหมดนี้")
        else:
            show_list = waiting_pallets[:6]
            st.caption(f"แสดง {len(show_list)} จากทั้งหมด {len(waiting_pallets)} พาเลตที่พร้อมดำเนินการ:")
            for pal in show_list:
                with st.container(border=True):
                    pc1, pc2, pc3 = st.columns([1.1, 1.4, 1.1])
                    with pc1:
                        st.markdown(f"**{pal['code']}**")
                        st.caption(f"SKU: {pal['sku']}")
                        st.caption(f"น้ำหนัก: {pal['weight']:,.0f} kg")
                    with pc2:
                        st.markdown(generate_barcode_svg(pal['code']), unsafe_allow_html=True)
                        stage_lbl = pal.get('stage_label') or PALLET_STATUS.get(pal['status'], pal['status'])
                        loc_txt = f"ชั้น {pal['location']}" if pal.get('location') else (f"ช่องรถ {pal.get('inbound_slot')}" if pal.get('inbound_slot') else "พื้นที่พัก")
                        st.caption(f"`{stage_lbl}` · {loc_txt}")
                    with pc3:
                        btn_label = "⚡ สแกน" if current_mode != "LOOKUP" else "🔍 ดูข้อมูล"
                        if st.button(btn_label, key=f"btn_scan_queue_{pal['code']}", type="primary", use_container_width=True):
                            try:
                                res = execute_barcode_scan(pal["code"], action_mode=current_mode)
                                st.session_state.barcode_last_result = res
                                st.session_state.barcode_scan_history.insert(0, {
                                    "เวลา": timestamp(),
                                    "รหัสบาร์โค้ด": res["code"],
                                    "กิจกรรม": res["action"],
                                    "สถานะเดิม": PALLET_STATUS.get(res["prev_status"], res["prev_status"]),
                                    "สถานะใหม่": PALLET_STATUS.get(res["new_status"], res["new_status"]),
                                    "ข้อความ": res["message"]
                                })
                                st.toast(f"บันทึกสำเร็จ: {res['message']}", icon="✅")
                                st.rerun()
                            except Exception as ex:
                                st.error(str(ex))
                        
                        if st.button("🏷️ ป้าย", key=f"btn_tag_queue_{pal['code']}", use_container_width=True):
                            pallet_detail_dialog(pal["code"])

    # Bottom Section: Scan History Table
    st.divider()
    col_h_left, col_h_right = st.columns([2, 1])
    with col_h_left:
        st.subheader("ประวัติการสแกนในเซสชัน (Scan History)")
    with col_h_right:
        if st.session_state.barcode_scan_history:
            if st.button("ล้างประวัติการสแกน", use_container_width=True):
                st.session_state.barcode_scan_history = []
                st.session_state.barcode_last_result = None
                st.rerun()

    if st.session_state.barcode_scan_history:
        st.dataframe(pd.DataFrame(st.session_state.barcode_scan_history), use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีประวัติการสแกนในเซสชันนี้")



def page_activity():
    st.header("Activity History & Traceability")
    st.caption("ตรวจสอบประวัติการทำงาน ย้อนรอยเส้นทางสินค้า และประวัติเที่ยวรถ (Audit Trail & Reporting)")

    c_f1, c_f2 = st.columns([1, 2])
    with c_f1:
        act_filter = st.selectbox("กรองตามประเภทกิจกรรม", ["ทั้งหมด", "PARK", "DEPART", "RECEIVE", "PUTAWAY", "PICK", "LOAD", "UNLOAD", "CREATE", "CANCEL", "SYSTEM"])
    with c_f2:
        keyword = st.text_input("🔍 ค้นหาประวัติด้วยรหัสพาเลต / เที่ยวรถ / คีย์เวิร์ด")

    sql = "SELECT created_at AS เวลา, action AS กิจกรรม, reference AS รหัสอ้างอิง, detail AS รายละเอียด FROM events WHERE 1=1"
    params = []
    if act_filter != "ทั้งหมด":
        sql += " AND action = ?"
        params.append(act_filter)
    if keyword.strip():
        sql += " AND (reference LIKE ? OR detail LIKE ?)"
        term = f"%{keyword.strip()}%"
        params.extend([term, term])
    sql += " ORDER BY id DESC LIMIT 200"

    event_data = rows(sql, tuple(params))
    df_events = pd.DataFrame(event_data)
    st.dataframe(df_events, use_container_width=True, hide_index=True)

    if not df_events.empty:
        csv_data = df_events.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 ดาวน์โหลดประวัติการทำงานเป็นไฟล์ CSV (Export Report)",
            data=csv_data,
            file_name=f"dockflow_activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.divider()
    col_act_h1, col_act_h2 = st.columns([2, 1])
    with col_act_h1:
        st.subheader("ประวัติเที่ยวรถที่เสร็จสิ้นแล้ว (Completed Trips History)")
    with col_act_h2:
        st.write("")
    completed = rows(
        """
        SELECT s.id AS เที่ยวรถ, 
               CASE s.kind WHEN 'IN' THEN 'ขาเข้า' ELSE 'ขาออก' END AS ประเภท, 
               t.name AS รถ, s.warehouse AS คลัง, s.partner AS คู่ค้าหรือปลายทาง, 
               s.created_at AS เวลาเข้า, s.departed_at AS เวลาออก
        FROM trips s JOIN trucks t ON t.id = s.truck_id 
        WHERE s.status = 'DEPARTED' ORDER BY s.departed_at DESC
        """
    )
    st.dataframe(pd.DataFrame(completed), use_container_width=True, hide_index=True)
    if completed:
        act_sel_col1, act_sel_col2 = st.columns([3, 1])
        with act_sel_col1:
            sel_comp_trip = st.selectbox("เลือกเที่ยวรถที่เสร็จสิ้นเพื่อดูเอกสารหลักฐาน", [c["เที่ยวรถ"] for c in completed], key="sel_comp_bol")
        with act_sel_col2:
            st.write("")
            st.write("")
            if st.button("ดูใบส่งมอบ (Bill of Lading)", key="btn_comp_bol", use_container_width=True):
                delivery_note_dialog(sel_comp_trip)


# =========================================================
# 7. AUTHENTICATION & LOGIN (Python Conditionals: if-else)
# =========================================================

def check_login():
    """
    ระบบตรวจสอบสิทธิ์เข้าใช้งาน (Authentication System)
    ประยุกต์ใช้ตรรกะพื้นฐานภาษา Python: Nested if-else conditionals
    และ Streamlit Session State ร่วมกับ st.form
    """
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = ""

    if not st.session_state.logged_in:
        col_pad1, col_box, col_pad2 = st.columns([1, 2, 1])
        with col_box:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center; margin-bottom: 2px;'>🚚 DOCKFLOW WMS</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #64748B; font-size: 14px;'>Smart Warehouse & Yard Management System</p>", unsafe_allow_html=True)
                st.divider()
                st.subheader("🔐 เข้าสู่ระบบ (System Login)")
                st.caption("ระบบรักษาความปลอดภัย: ตรวจสอบสิทธิ์ด้วยตรรกะโครงสร้างเงื่อนไข (if-else)")

                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input("👤 ชื่อผู้ใช้งาน (Username)")
                    password = st.text_input("🔑 รหัสผ่าน (Password)", type="password")
                    submit_login = st.form_submit_button("เข้าสู่ระบบ (Sign In)", use_container_width=True, type="primary")

                    if submit_login:
                        # ---------------------------------------------------------
                        # ตรรกะ Python จากบทเรียน: Nested if-else Statement
                        # ---------------------------------------------------------
                        if username == "admin":
                            if password == "1234":
                                st.session_state.logged_in = True
                                st.session_state.user = username
                                st.success("✅ ยืนยันตัวตนสำเร็จ กำลังนำเข้าสู่ระบบ...")
                                st.rerun()
                            else:
                                st.error("❌ รหัสผ่านไม่ถูกต้อง (Incorrect password)")
                        else:
                            st.error("❌ ไม่พบชื่อผู้ใช้นี้ในระบบ (Username not found)")
        st.stop()


# =========================================================
# 8. MAIN CONTROLLER & SIDEBAR
# =========================================================

initialize_database()
check_login()

with st.sidebar:
    st.markdown("## 🚚 DOCKFLOW WMS")
    st.caption("ระบบบริหารจัดการคลังสินค้าและลานเทียบท่า")
    st.markdown(f"🟢 **สถานะ:** ออนไลน์ | 👤 **ผู้ใช้:** `{st.session_state.get('user', 'admin')}`")
    if st.button("🚪 ออกจากระบบ (Logout)", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user = ""
        st.rerun()
    st.divider()

    # 1. Live Snapshot Container
    with st.container(border=True):
        st.markdown("<div style='font-size:12px; font-weight:700; color:#475569; margin-bottom:6px;'>ภาพรวมสถานะสด</div>", unsafe_allow_html=True)
        c_sb1, c_sb2, c_sb3 = st.columns(3)
        sb_docked = scalar("SELECT COUNT(*) FROM trips WHERE status = 'DOCKED'")
        sb_wait = scalar("SELECT COUNT(*) FROM trips WHERE status = 'WAITING'")
        sb_avail = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'AVAILABLE'")
        c_sb1.metric("เทียบท่า", f"{sb_docked}/4")
        c_sb2.metric("รอคิว", f"{sb_wait}")
        c_sb3.metric("พร้อมใช้", f"{sb_avail}")

    MENU = [
        ("Control Tower", "📊 1. Control Tower", "ศูนย์ควบคุมสถานะภาพรวม"),
        ("Dock Yard", "🅿️ 2. Dock Yard", "จัดการคิวรถและช่องเทียบท่า"),
        ("Smart Receiving", "📥 3. Receiving & Inbound", "รับเข้าและขนถ่ายสินค้า"),
        ("Warehouse Mapping", "🏢 4. Warehouse Mapping", "ผังคลังและตำแหน่งจัดเก็บ"),
        ("Shipping & Outbound", "📤 5. Shipping & Outbound", "เบิกจ่ายและโหลดสินค้า"),
        ("Barcode Terminal", "🔫 6. Barcode Terminal", "สถานีสแกนเนอร์บาร์โค้ด"),
    ]

    # 2. Modern Button Navigation with Text Line
    if "current_page" not in st.session_state or st.session_state.current_page not in [m[0] for m in MENU]:
        st.session_state.current_page = "Control Tower"

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div.stButton > button {
            text-align: left;
            display: flex;
            justify-content: flex-start;
            padding-left: 14px;
            font-weight: 600;
            font-size: 0.90rem;
            border-radius: 8px;
            margin-bottom: 2px;
        }
        div[data-testid="stSidebar"] .nav-desc {
            font-size: 0.72rem;
            color: #94A3B8;
            margin-top: -6px;
            margin-bottom: 9px;
            padding-left: 14px;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for page_key, label, desc in MENU:
        is_active = (st.session_state.current_page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_btn_{page_key}", use_container_width=True, type=btn_type):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()
        st.markdown(f"<div class='nav-desc'>{desc}</div>", unsafe_allow_html=True)

    workspace = st.session_state.current_page



pages = {
    "Control Tower": page_control_tower,
    "Dock Yard": page_dock_yard,
    "Smart Receiving": page_receiving,
    "Warehouse Mapping": page_warehouse,
    "Shipping & Outbound": page_shipping,
    "Barcode Terminal": page_barcode_terminal,
    "Activity Logs": page_activity,
}

pages[workspace]()