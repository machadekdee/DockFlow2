from contextlib import contextmanager
from datetime import datetime
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
thai_font_candidates = [
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

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [selected_thai_font, "Tahoma", "Leelawadee UI", "Segoe UI", "DejaVu Sans"]
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
                max_weight REAL NOT NULL CHECK(max_weight > 0)
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
                load_slot INTEGER
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

        conn.executemany(
            "INSERT INTO trucks VALUES (?, ?, ?, ?)",
            [(f"T{i:02}", f"TRUCK {i:02}", 6, 3000) for i in range(1, 9)],
        )

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

        # Inventory
        for i in range(1, 25):
            p = products[(i - 1) % len(products)]
            wh = "WH-A" if i <= 12 else "WH-B"
            loc = f"{wh}-R{((i - 1) % 6) + 1:02}"
            conn.execute(
                """
                INSERT INTO pallets(code, sku, qty, weight, warehouse, location, status)
                VALUES (?, ?, ?, ?, ?, ?, 'AVAILABLE')
                """,
                (f"PAL-{i:03}", p[0], 20, 20 * p[3], wh, loc),
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
                conn.execute(
                    """
                    INSERT INTO pallets(code, sku, qty, weight, warehouse, status, inbound_id, inbound_slot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (f"{pfx}-{i:02}", p[0], 20, 20 * p[3], wh, stat, trip_id, i),
                )

        log_event(conn, "SYSTEM", "INIT", "รีเซ็ตและสร้างข้อมูลจำลองระบบเรียบร้อย")


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
    ax.text(3, 9.0, f"ภารกิจ: {'ขนส่งสินค้าเข้า' if inbound else 'จ่ายสินค้าออก'} | คู่ค้า: {t_data['partner']}", ha="center", va="center", fontsize=9, color="#64748B")

    # Truck Body Container
    ax.add_patch(Rectangle((1.0, 1.8), 4.0, 6.5, facecolor="#E2E8F0", edgecolor="#64748B", linewidth=2, linestyle="--"))
    ax.text(3, 8.0, "-- ประตูท้ายรถ (REAR GATE) --", ha="center", va="center", fontsize=8, color="#475569", fontweight="bold")

    # Cabin (Head)
    ax.add_patch(FancyBboxPatch((1.2, 0.4), 3.6, 1.2, boxstyle="round,pad=0.1", facecolor="#4F46E5", edgecolor="#3730A3"))
    ax.text(3, 1.0, "หัวรถบรรทุก (CABIN)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(3, 0.65, f"พิกัดรับน้ำหนัก: {t_data.get('max_weight', 3000):,.0f} kg", ha="center", va="center", fontsize=7, color="#E0E7FF")

    # Wheels
    for wx, wy in [(0.8, 1.0), (4.8, 1.0), (0.8, 6.5), (4.8, 6.5)]:
        ax.add_patch(Rectangle((wx, wy), 0.4, 1.0, facecolor="#334155", edgecolor="#0F172A"))

    # Slots and Pallets grid inside truck
    slots_total = t_data["slots"]
    row_count = (slots_total + 1) // 2

    for r in range(row_count):
        for c in range(2):
            slot_idx = r * 2 + c + 1
            if slot_idx > slots_total:
                continue

            sx = 1.35 + c * 1.75
            sy = 6.8 - (r * 1.15)

            item = slot_map.get(slot_idx)
            if item:
                color = CATEGORY_COLOR.get(item["category"], "#0F766E")
                ax.add_patch(FancyBboxPatch((sx, sy), 1.5, 0.95, boxstyle="round,pad=0.05", facecolor=color, edgecolor="#0F172A", linewidth=1.5))
                ax.text(sx + 0.75, sy + 0.55, item["code"], ha="center", va="center", fontsize=9, color="white", fontweight="bold")
                ax.text(sx + 0.75, sy + 0.22, f"{item['qty']} หน่วย", ha="center", va="center", fontsize=7, color="#E2E8F0")
            else:
                ax.add_patch(FancyBboxPatch((sx, sy), 1.5, 0.95, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="#CBD5E1", linestyle=":"))
                ax.text(sx + 0.75, sy + 0.47, f"ช่อง {slot_idx:02}", ha="center", va="center", fontsize=8, color="#94A3B8")

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
            for s in range(1, count + 1):
                conn.execute(
                    """
                    INSERT INTO pallets(code, sku, qty, weight, warehouse, status, inbound_id, inbound_slot)
                    VALUES (?, ?, ?, ?, ?, 'ON_INBOUND', ?, ?)
                    """,
                    (make_id("RCV"), sku, qty, qty * prod["unit_weight"], warehouse, trip_id, s),
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
    """ฟังก์ชันแสดงผลรายการพาเลตแบบมาตรฐาน รองรับ fallback ปลอดภัย"""
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
        display.append({
            "รหัสพาเลต": it["code"],
            "SKU": it["sku"],
            "ชื่อสินค้า": it.get("product_name", it.get("name", it["sku"])),
            "หมวดหมู่": it.get("category", "-"),
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
    max_truck_wt = trip.get("max_weight", 3000)
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
            <div style="font-size: 11px; color: #475569; font-weight: 600;">ศูนย์กระจายสินค้าและคลังสินค้าอัจฉริยะ (Bangkok Central Distribution Hub)</div>
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
    
    tag_html = f"""
    <div style="border: 2px solid #0F172A; border-radius: 10px; padding: 16px 20px; background: #FFFFFF; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); max-width: 480px; margin: 0 auto 16px auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0F172A; padding-bottom: 8px; margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: 900; color: #0F172A; letter-spacing: 1.5px;">DOCKFLOW WMS</div>
            <div style="font-size: 11px; font-weight: 700; background: {cat_color}; color: #FFFFFF; padding: 2px 8px; border-radius: 4px;">{item['category']}</div>
        </div>
        <div style="text-align: center; margin-bottom: 8px;">
            {svg_barcode}
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px; margin-bottom: 10px;">
            <div style="font-size: 16px; font-weight: 800; color: #0F172A;">{item['name']}</div>
            <div style="font-size: 12px; color: #64748B; font-weight: 600;">SKU: <span style="color: #2563EB;">{item['sku']}</span></div>
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
        </div>
        <div style="font-size: 10px; color: #94A3B8; text-align: right; border-top: 1px dashed #CBD5E1; padding-top: 6px;">
            เอกสารควบคุม WMS System · รหัสอ้างอิง: {item['inbound_id'] or item['outbound_id'] or 'INIT'}
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
                    ax_cat.text(0, 0, "รวม\\n" + str(sum(sizes)) + " พาเลต", ha="center", va="center", fontsize=10, fontweight="bold", color="#1E293B")
                    fig_cat.tight_layout()
                    st.pyplot(fig_cat, use_container_width=True)
                    plt.close(fig_cat)
                    
                    # 4 Symmetrical Category Badges Underneath
                    m_cols = st.columns(len(cat_data))
                    for idx, c in enumerate(cat_data):
                        with m_cols[idx]:
                            color_pill = CATEGORY_COLOR.get(c['category'], '#64748B')
                            st.markdown(
                                f"""<div style="border-left: 3px solid {color_pill}; padding-left: 6px; background: #F8FAFC; border-radius: 4px; padding: 4px 6px;">
                                    <div style="font-size: 10px; color: #64748B; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{c['category']}</div>
                                    <strong style="font-size: 13px; color: #0F172A;">{c['cnt']} <span style="font-size: 10px; font-weight: normal;">พาเลต</span></strong>
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
                
                # 2 Symmetrical Summary Cards Underneath
                w_col1, w_col2 = st.columns(2)
                for idx, w in enumerate(wh_util):
                    target_col = w_col1 if idx == 0 else w_col2
                    with target_col:
                        p_val = pcts[idx]
                        p_color = "#3B82F6" if idx == 0 else "#8B5CF6"
                        target_col.markdown(
                            f"""<div style="border-left: 3px solid {p_color}; padding-left: 6px; background: #F8FAFC; border-radius: 4px; padding: 4px 8px;">
                                <div style="font-size: 10px; color: #64748B;">คลัง {w['wh']} (ความจุ {w['cap']} ช่อง)</div>
                                <strong style="font-size: 13px; color: #0F172A;">{w['used']} พาเลต <span style="font-size: 11px; color: {p_color};">({p_val}%)</span></strong>
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
    """แสดงคำแนะนำลำดับงานอัจฉริยะ (Smart Workflow Assistant) ตรวจสอบสถานะสดและชี้เป้าขั้นตอนถัดไป"""
    waiting_trips = rows("SELECT id, kind, truck_id FROM trips WHERE status = 'WAITING' ORDER BY priority = 'High' DESC LIMIT 1")
    docked_inbound_pallets = scalar("SELECT COUNT(*) FROM pallets p JOIN trips s ON s.id = p.inbound_id WHERE s.status = 'DOCKED' AND p.status = 'ON_INBOUND'")
    receiving_pallets = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RECEIVING'")
    reserved_pallets = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RESERVED'")
    picked_pallets = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'PICKED'")
    docked_out_ready = scalar("SELECT COUNT(*) FROM trips s WHERE s.kind = 'OUT' AND s.status = 'DOCKED' AND (SELECT COUNT(*) FROM pallets WHERE outbound_id = s.id AND status != 'LOADED') = 0 AND (SELECT COUNT(*) FROM pallets WHERE outbound_id = s.id) > 0")

    guide_color = "#2563EB"
    guide_title = "🧭 แนะนำขั้นตอนการทำงานถัดไป (Next Recommended Action)"
    guide_msg = "ระบบคลังสินค้าพร้อมปฏิบัติงาน ทุกขั้นตอนอยู่ในสถานะเรียบร้อย"
    target_page = None
    target_btn = None

    if waiting_trips:
        tr = waiting_trips[0]
        guide_color = "#DC2626"
        guide_title = "🚦 ขั้นตอนแนะนำ: มีรถรอเข้าจอดเทียบท่า"
        guide_msg = f"มีรถเที่ยว {tr['id']} ({'ขาเข้า' if tr['kind'] == 'IN' else 'ขาออก'}) อยู่ในคิวรอเข้าจอด ➔ กรุณาจัดช่องจอดเทียบท่า"
        target_page = "Dock Yard"
        target_btn = "👉 ไปที่หน้า Dock Yard เพื่อนำรถเข้าจอด"
    elif docked_inbound_pallets > 0:
        guide_color = "#0284C7"
        guide_title = "📥 ขั้นตอนแนะนำ: ขนถ่ายสินค้าลงจากรถขาเข้า"
        guide_msg = f"มีสินค้า {docked_inbound_pallets} พาเลตจอดเทียบท่าแล้ว ➔ ขนลงสู่พื้นที่พักรับ หรือใช้ Barcode Scanner ยิงสแกน"
        target_page = "Smart Receiving"
        target_btn = "👉 ไปที่หน้า Smart Receiving เพื่อขนถ่ายลง"
    elif receiving_pallets > 0:
        guide_color = "#D97706"
        guide_title = "📍 ขั้นตอนแนะนำ: จัดเก็บสินค้าเข้าชั้นวาง (Putaway)"
        guide_msg = f"มีสินค้าตกค้างในพื้นที่พักรับ {receiving_pallets} พาเลต ➔ นำเข้าจัดเก็บบนชั้นวาง หรือยิงสแกนบาร์โค้ดจัดเก็บทันที"
        target_page = "Warehouse Mapping"
        target_btn = "👉 ไปที่หน้า Warehouse Mapping เพื่อจัดเก็บเข้าชั้น"
    elif reserved_pallets > 0:
        guide_color = "#8B5CF6"
        guide_title = "🛒 ขั้นตอนแนะนำ: หยิบสินค้าตามใบจ่าย (Pick)"
        guide_msg = f"มีสินค้าจองรอหยิบ {reserved_pallets} พาเลต ➔ หยิบจากชั้นวางไปยังพื้นที่พักจ่าย"
        target_page = "Shipping & Outbound"
        target_btn = "👉 ไปที่หน้า Shipping เพื่อหยิบสินค้า"
    elif picked_pallets > 0:
        guide_color = "#4F46E5"
        guide_title = "📦 ขั้นตอนแนะนำ: โหลดสินค้าขึ้นรถขาออก (Load)"
        guide_msg = f"มีสินค้าในพื้นที่พักจ่าย {picked_pallets} พาเลตพร้อมโหลด ➔ นำขึ้นรถบรรทุกขาออกที่เทียบท่า"
        target_page = "Shipping & Outbound"
        target_btn = "👉 ไปที่หน้า Shipping เพื่อโหลดสินค้าขึ้นรถ"
    elif docked_out_ready > 0:
        guide_color = "#059669"
        guide_title = "🚀 ขั้นตอนแนะนำ: ปล่อยรถขาออกส่งสินค้า (Depart)"
        guide_msg = "สินค้าโหลดขึ้นรถครบ 100% แล้ว ➔ กดปล่อยรถออกและพิมพ์ใบส่งมอบสินค้า (Bill of Lading)"
        target_page = "Shipping & Outbound"
        target_btn = "👉 ไปที่หน้า Shipping เพื่อปล่อยรถและพิมพ์ใบส่งมอบ"

    with st.container(border=True):
        g_c1, g_c2 = st.columns([3, 1.2])
        with g_c1:
            st.markdown(f"<div style='color:{guide_color}; font-weight:800; font-size:13px;'>{guide_title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#334155; font-size:12px; margin-top:2px;'>{guide_msg}</div>", unsafe_allow_html=True)
        with g_c2:
            if target_page and target_btn:
                st.write("")
                if st.button(target_btn, key="guide_shortcut_btn", type="primary", use_container_width=True):
                    st.session_state.current_page = target_page
                    st.rerun()
            else:
                st.markdown("<div style='text-align:right; font-size:12px; color:#10B981; font-weight:bold; padding-top:6px;'>✅ สถานะระบบพร้อมสมบูรณ์</div>", unsafe_allow_html=True)


def page_control_tower():
    st.header("Control Tower & Dashboard")
    render_smart_workflow_guide()
    st.caption("ศูนย์กลางการมองเห็นสถานะคลังสินค้า ลานจอด และรถบรรทุกแบบเรียลไทม์ (Auto-Refreshing Command Center)")
    control_tower_fragment()


# --- PAGE 2: DOCK YARD ---
def page_dock_yard():
    st.header("Dock Yard Management")
    render_smart_workflow_guide()
    st.caption("จัดการคิวรถบรรทุก จัดการลำดับความสำคัญ และนำรถเข้าจอดประจำช่องปฏิบัติงาน")
    st.info("💡 **ขั้นตอนการทำงาน:** 1. ดูคิวรถรอเข้าจอด ➔ 2. เลือกช่องเทียบท่า (Dock) ➔ 3. กด **นำรถเข้าจอด (Park)** เพื่อเริ่มปฏิบัติงาน")

    waiting = rows(
        """
        SELECT s.*, t.name AS truck_name, t.slots,
               (SELECT COUNT(*) FROM pallets p WHERE p.inbound_id = s.id OR p.outbound_id = s.id) AS pallet_count
        FROM trips s JOIN trucks t ON t.id = s.truck_id 
        WHERE s.status = 'WAITING'
        ORDER BY CASE s.priority WHEN 'High' THEN 1 WHEN 'Normal' THEN 2 ELSE 3 END, s.created_at ASC
        """
    )

    with st.container(border=True):
        st.subheader("🚦 คิวรถรอเข้าจอด (Priority Queue)")
        if not waiting:
            st.success("ไม่มีรถรอเข้าจอดในขณะนี้ ทุกลำดับการขนส่งปฏิบัติงานเรียบร้อย")
        else:
            q_table = []
            for w in waiting:
                q_table.append({
                    "ความเร่งด่วน": f"{PRIORITY_ICON[w['priority']]} {w['priority']}",
                    "รหัสเที่ยว": w["id"],
                    "ประเภท": "ขาเข้า (IN)" if w["kind"] == "IN" else "ขาออก (OUT)",
                    "รถบรรทุก": w["truck_name"],
                    "คลัง": w["warehouse"],
                    "คู่ค้า/ปลายทาง": w["partner"],
                    "จำนวนพาเลต": f"{w['pallet_count']} พาเลต",
                    "เวลาเข้าคิว": w["created_at"],
                })
            st.dataframe(pd.DataFrame(q_table), use_container_width=True, hide_index=True)

            col1, col2 = st.columns([2, 1])
            lookup = {t["id"]: t for t in waiting}
            with col1:
                selected = st.selectbox(
                    "เลือกเที่ยวรถเพื่อดำเนินการ",
                    list(lookup),
                    format_func=lambda x: f"{PRIORITY_ICON[lookup[x]['priority']]} [{lookup[x]['priority']}] {x} · {lookup[x]['truck_name']} ({lookup[x]['kind']} · {lookup[x]['warehouse']})",
                )
            trip = lookup[selected]

            occupied_docks = {r["dock"] for r in rows("SELECT dock FROM trips WHERE status = 'DOCKED'")}
            free_docks = [d for d, cfg in DOCKS.items() if cfg["kind"] == trip["kind"] and cfg["warehouse"] == trip["warehouse"] and d not in occupied_docks]

            with col2:
                st.write("")
                st.write("")
                if st.button("❌ ยกเลิกเที่ยวรถนี้ (Cancel Trip)", use_container_width=True):
                    try:
                        execute_action("cancel_trip", selected)
                        st.toast(f"❌ ยกเลิกเที่ยวรถ {selected} สำเร็จ", icon="🗑️")
                        st.success(f"ยกเลิกเที่ยวรถ {selected} สำเร็จ คืนสถานะสินค้าเรียบร้อย")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            if free_docks:
                c_dock, c_btn = st.columns([2, 1])
                with c_dock:
                    dock_choice = st.selectbox("เลือกช่องจอดที่ว่างและตรงเงื่อนไข", free_docks, format_func=lambda x: f"{x} · {DOCKS[x]['title']}")
                with c_btn:
                    st.write("")
                    st.write("")
                    if st.button("🅿️ นำรถเข้าจอด (Park Truck)", type="primary", use_container_width=True):
                        try:
                            execute_action("park", selected, dock_choice)
                            st.toast(f"🅿️ นำรถเข้าจอดช่อง {dock_choice} สำเร็จ!", icon="✅")
                            st.success(f"นำรถเข้าจอดช่อง {dock_choice} สำเร็จ!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            else:
                st.warning("⚠️ ไม่มีช่องจอดที่ว่างสำหรับประเภทงานและคลังนี้ กรุณารอรถคันอื่นออกจากลาน")

    st.subheader("ภาพสถานะรถในช่องจอดปัจจุบัน (Live Docked Trucks)")
    docked_trips = rows("SELECT s.id, s.dock, s.kind, s.warehouse, s.partner, t.name FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.status = 'DOCKED' ORDER BY s.dock ASC")

    if not docked_trips:
        st.info("ยังไม่มีรถจอดในช่องปฏิบัติงาน")
    else:
        sel_docked = st.selectbox(
            "เลือกรถในช่องจอดเพื่อดูแผนผังการบรรทุก",
            [d["id"] for d in docked_trips],
            format_func=lambda x: next(f"ช่อง {d['dock']}: {d['name']} ({x} · {'ขาเข้า' if d['kind'] == 'IN' else 'ขาออก'})" for d in docked_trips if d["id"] == x),
        )
        c_left, c_right = st.columns([1, 1.2])
        with c_left:
            fig = draw_truck_graphic(sel_docked)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c_right:
            dt_info = next(d for d in docked_trips if d["id"] == sel_docked)
            with st.container(border=True):
                st.subheader(f"🚚 {dt_info['name']} (ช่อง {dt_info['dock']})")
                st.markdown(f"**ประเภท:** {'ขนส่งสินค้าเข้า (IN)' if dt_info['kind'] == 'IN' else 'จ่ายสินค้าออก (OUT)'}")
                st.markdown(f"**คลังสินค้า:** `{dt_info['warehouse']}` | **คู่ค้า:** `{dt_info['partner']}`")

                if dt_info["kind"] == "IN":
                    on_truck = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'ON_INBOUND'", (sel_docked,))
                    recv = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'RECEIVING'", (sel_docked,))
                    avail = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ? AND status = 'AVAILABLE'", (sel_docked,))
                    st.metric("สินค้าค้างบนรถ", f"{on_truck} พาเลต")
                    st.caption(f"พักรับแล้ว: {recv} | จัดเก็บแล้ว: {avail}")
                    if on_truck == 0:
                        st.success("สินค้าถูกขนลงครบแล้ว พร้อมปล่อยรถออกจากลาน")
                        if st.button("🚚 ปล่อยรถออกจากลาน (Depart)", type="primary"):
                            try:
                                execute_action("depart", sel_docked)
                                st.toast("🚚 รถขาเข้าออกจากลานสำเร็จ!", icon="✅")
                                st.success("ปล่อยรถออกจากลานสำเร็จ")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                else:
                    loaded = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ? AND status = 'LOADED'", (sel_docked,))
                    total = scalar("SELECT COUNT(*) FROM pallets WHERE outbound_id = ?", (sel_docked,))
                    st.metric("ความคืบหน้าการโหลด", f"{loaded} / {total} พาเลต")
                    if loaded == total and total > 0:
                        st.success("สินค้าโหลดขึ้นรถครบถ้วน พร้อมปล่อยรถออก")
                        if st.button("🚚 ปล่อยรถออกจากลาน (Depart)", type="primary"):
                            try:
                                execute_action("depart", sel_docked)
                                st.toast("🎉 รถขาออกเดินทางสำเร็จ! สินค้าส่งมอบครบถ้วน", icon="🚚")
                                st.balloons()
                                st.success("ปล่อยรถออกจากลานสำเร็จ")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))


# --- PAGE 3: SMART RECEIVING ---
def page_receiving():
    st.header("Smart Receiving & Inbound")
    render_smart_workflow_guide()
    st.caption("กระบวนการรับเข้า: ตรวจสอบสินค้าบนรถ → ขนลงพื้นที่พักรับ → จัดเก็บเข้าชั้นวาง")
    st.info("💡 **ขั้นตอนการรับสินค้า:** 1. เลือกรถที่เทียบท่า ➔ 2. กดยืนยัน **ขนลงสู่พื้นที่พักรับ** ➔ 3. จัดเก็บเข้าชั้นวางที่เมนู Warehouse Mapping ➔ 4. กดยืนยันปล่อยรถออกจากลาน")

    tab1, tab2 = st.tabs(["📦 ปฏิบัติการขนลงจากรถ (Unload Station)", "➕ ลงทะเบียนรถขาเข้าใหม่ (New Inbound)"])

    with tab1:
        trips = rows("SELECT s.*, t.name FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.kind = 'IN' AND s.status NOT IN ('DEPARTED', 'CANCELLED')")
        if not trips:
            st.info("ไม่มีเที่ยวรถขาเข้าที่กำลังดำเนินการในระบบ")
            return

        lookup = {t["id"]: t for t in trips}
        sel_id = st.selectbox("เลือกเที่ยวรถขาเข้า", list(lookup), format_func=lambda x: f"{lookup[x]['id']} · {lookup[x]['name']} ({lookup[x]['status']} · {lookup[x]['warehouse']})")
        trip = lookup[sel_id]

        col_left, col_right = st.columns([1.1, 1.4])
        with col_left:
            fig = draw_truck_graphic(trip["id"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_right:
            st.subheader("Unloading Station")
            pending = rows("SELECT p.*, pr.name AS product_name FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.inbound_id = ? AND p.status = 'ON_INBOUND' ORDER BY p.inbound_slot ASC", (trip["id"],))
            total_pallets = scalar("SELECT COUNT(*) FROM pallets WHERE inbound_id = ?", (trip["id"],))
            unloaded_cnt = total_pallets - len(pending)

            st.progress(unloaded_cnt / max(total_pallets, 1), text=f"ขนลงแล้ว {unloaded_cnt} / {total_pallets} พาเลต")

            if trip["status"] != "DOCKED":
                st.warning("⚠️ รถคันนี้ยังไม่ได้เข้าจอดที่ช่องรับสินค้า กรุณานำรถเข้าจอดก่อนขนถ่าย")
                if trip["status"] == "WAITING":
                    occupied_docks = {r["dock"] for r in rows("SELECT dock FROM trips WHERE status = 'DOCKED'")}
                    free_docks = [d for d, cfg in DOCKS.items() if cfg["kind"] == trip["kind"] and cfg["warehouse"] == trip["warehouse"] and d not in occupied_docks]
                    if free_docks:
                        target_dock = free_docks[0]
                        if st.button(f"🅿️ นำรถเข้าจอดช่อง {target_dock} ทันที (ลัดขั้นตอนด่วน)", type="primary", use_container_width=True):
                            try:
                                execute_action("park", trip["id"], target_dock)
                                st.toast(f"🚚 นำรถเข้าจอดช่อง {target_dock} สำเร็จ!", icon="✅")
                                st.success(f"นำรถเข้าจอดช่อง {target_dock} เรียบร้อยแล้ว")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
            elif pending:
                p_lookup = {p["code"]: p for p in pending}
                with st.form("unload_form"):
                    chosen = st.selectbox("เลือกพาเลตบนรถเพื่อขนลง", list(p_lookup), format_func=lambda x: f"{x} (ช่อง {p_lookup[x]['inbound_slot']}) · {p_lookup[x]['product_name']}")
                    if st.form_submit_button("⬇ ขนพาเลตนี้ลงสู่พื้นที่พักรับ", type="primary", use_container_width=True):
                        try:
                            execute_action("receive", chosen, trip["id"])
                            st.toast(f"📥 ขน {chosen} ลงสู่พื้นที่พักรับ", icon="✅")
                            st.success(f"ขน {chosen} ลงสู่พื้นที่พักรับเรียบร้อย")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                if st.button("⚡ ขนลงทั้งหมดในคลิกเดียว (Unload All)", use_container_width=True):
                    try:
                        execute_action("receive_all", trip["id"])
                        st.toast("⚡ ขนสินค้าลงจากรถทั้งหมดสำเร็จ!", icon="📥")
                        st.success("ขนสินค้าลงจากรถทั้งหมดสำเร็จ!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.success("✅ ขนสินค้าลงจากรถครบทุกพาเลตแล้ว! สามารถกดปล่อยรถออกจากลานได้")

            if trip["status"] == "DOCKED" and not pending:
                with st.expander("🚚 ยืนยันปล่อยรถขาเข้าออกจากลาน", expanded=True):
                    if st.button("ยืนยันรถขาเข้าออกจากลาน (Depart Truck)", type="primary", use_container_width=True):
                        try:
                            execute_action("depart", trip["id"])
                            st.toast("🚚 รถขาเข้าออกจากลานสำเร็จ!", icon="✅")
                            st.success("รถออกจากลานสำเร็จ")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        st.divider()
        st.subheader("รายการพาเลตทั้งหมดในเที่ยวรถนี้ (Shipment Manifest)")
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
            st.warning("ไม่มีรถว่างสำหรับสร้างเที่ยวใหม่ กรุณารอรถเที่ยวอื่นเสร็จสิ้น")
        else:
            t_map = {t["id"]: t for t in trucks}
            p_map = {p["sku"]: p for p in prods}
            with st.form("new_inbound_form"):
                a, b = st.columns(2)
                with a:
                    t_id = st.selectbox("เลือกหัวรถบรรทุก", list(t_map), format_func=lambda x: f"{x} ({t_map[x]['name']} · ความจุ {t_map[x]['slots']} ช่อง · รับน้ำหนัก {t_map[x]['max_weight']:,.0f} kg)")
                    wh = st.selectbox("คลังปลายทาง", WAREHOUSES)
                    partner = st.text_input("ชื่อผู้ส่งสินค้า (Supplier)", "NORTH SUPPLIER CO.")
                with b:
                    sku = st.selectbox("สินค้า", list(p_map), format_func=lambda x: f"{x} · {p_map[x]['name']} ({p_map[x]['unit_weight']} kg/หน่วย)")
                    qty = st.number_input("จำนวนหน่วยต่อพาเลต", 1, 500, 20)
                    cnt = st.number_input("จำนวนพาเลต", 1, t_map[t_id]["slots"], min(4, t_map[t_id]["slots"]))
                prio = st.selectbox("ความเร่งด่วน (Priority)", ["High", "Normal", "Low"])

                calc_weight = qty * cnt * p_map[sku]["unit_weight"]
                st.caption(f"📊 คำนวณน้ำหนักรวม: **{calc_weight:,.0f} kg** / พิกัดรถสูงสุด: **{t_map[t_id]['max_weight']:,.0f} kg**")

                if st.form_submit_button("บันทึกและสร้างเที่ยวรถขาเข้า", type="primary"):
                    try:
                        execute_action_id = create_new_trip("IN", t_id, wh, partner, prio, (sku, qty, cnt))
                        st.toast(f"📋 สร้างเที่ยวรถขาเข้า {execute_action_id} สำเร็จ!", icon="🚚")
                        st.success(f"สร้างเที่ยวรถขาเข้า {execute_action_id} สำเร็จ!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# --- PAGE 4: WAREHOUSE MAPPING ---
def page_warehouse():
    st.header("Warehouse & Storage Mapping")
    render_smart_workflow_guide()
    st.caption("ผังคลังสินค้า ตำแหน่งชั้นจัดเก็บ และการจัดการสินค้าพร้อมใช้ (Putaway & Inventory Control)")
    st.info("💡 **การตรวจสอบผังจัดเก็บ:** 1. ตรวจสอบสถานะความจุของชั้นวาง ➔ 2. นำสินค้าเข้าชั้นวาง (Putaway) ➔ 3. คลิกที่ช่องหรือค้นหาเพื่อดูข้อมูลและป้ายพาเลต (Pallet Tag)")

    # Locate Pallet in Warehouse Map (Dedicated Search & Rack Locator)
    with st.container(border=True):
        qs_col1, qs_col2 = st.columns([3, 1.2])
        with qs_col1:
            scan_query = st.text_input("🔍 ค้นหาตำแหน่งจัดเก็บพาเลตในผัง (Search & Locate Pallet)", placeholder="พิมพ์รหัสพาเลต เช่น PAL-001, PAL-015 เพื่อดูว่าเก็บอยู่ล็อคไหน ชั้นไหน...", key="quick_scan_input")
        with qs_col2:
            st.write("")
            st.write("")
            if st.button("🔎 ตรวจสอบพิกัดชั้นวาง", use_container_width=True, type="primary", key="btn_quick_scan"):
                if scan_query.strip():
                    pallet_detail_dialog(scan_query.strip().upper())
                else:
                    st.toast("กรุณาระบุรหัสพาเลตที่ต้องการค้นหา", icon="⚠️")
        st.caption("💡 **คำแนะนำการใช้งาน:** หน้านี้ใช้สำหรับ **ตรวจสอบผังคลังและค้นหาพิกัดชั้นวาง** · หากต้องการ **ยิงสแกนเนอร์เพื่อเปลี่ยนสถานะ/เคลื่อนย้ายสินค้าจริง** ให้ใช้งานที่หน้า **[🔫 6. Barcode Terminal]**")

    wh = st.radio("เลือกคลังสินค้า", WAREHOUSES, horizontal=True)

    stage_items = rows(
        """
        SELECT p.*, pr.name AS product_name, pr.category 
        FROM pallets p JOIN products pr ON pr.sku = p.sku 
        WHERE p.warehouse = ? AND p.status = 'RECEIVING' ORDER BY p.code ASC
        """,
        (wh,),
    )

    with st.container(border=True):
        st.subheader("📥 จัดเก็บสินค้าจากพื้นที่พักรับเข้าชั้นวาง (Putaway Station)")
        if not stage_items:
            st.success("✅ ไม่มีสินค้าค้างในพื้นที่พักรับของคลังนี้ สินค้าทั้งหมดจัดเก็บบนชั้นวางเรียบร้อย")
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
                st.error("⚠️ ช่องจัดเก็บในคลังนี้เต็มทั้งหมดแล้ว กรุณาเคลียร์สินค้าออกก่อน")
            else:
                p_map = {item["code"]: item for item in stage_items}
                l_map = {l["id"]: l for l in locs}

                c_single, c_batch = st.columns([2, 1])
                with c_single:
                    with st.form("putaway_form"):
                        c1, c2 = st.columns(2)
                        pal_code = c1.selectbox("เลือกพาเลตจากพื้นที่พักรับ", list(p_map), format_func=lambda x: f"{x} · {p_map[x]['product_name']}")
                        loc_id = c2.selectbox("เลือกตำแหน่งชั้นจัดเก็บ", list(l_map), format_func=lambda x: f"{x} (ว่าง {l_map[x]['capacity'] - l_map[x]['used']} ช่อง)")
                        if st.form_submit_button("📍 ยืนยันจัดเก็บเข้าชั้นวาง", type="primary", use_container_width=True):
                            try:
                                execute_action("putaway", pal_code, loc_id)
                                st.toast(f"📍 จัดเก็บ {pal_code} เข้าชั้นวาง {loc_id} สำเร็จ!", icon="✅")
                                st.success(f"จัดเก็บพาเลต {pal_code} ไปยัง {loc_id} สำเร็จ!")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                with c_batch:
                    st.write("")
                    st.write("")
                    if st.button("⚡ จัดเก็บแนะนำทั้งหมดอัตโนมัติ (Putaway All)", use_container_width=True):
                        try:
                            execute_action("putaway_all", wh)
                            st.toast(f"⚡ จัดเก็บสินค้าในคลัง {wh} ครบทุกพาเลต!", icon="📍")
                            st.success(f"จัดเก็บสินค้าในคลัง {wh} ทั้งหมดเรียบร้อย!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    st.subheader(f"แผนผังชั้นจัดเก็บสินค้า · คลัง {wh}")
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
        SELECT p.code, p.location, pr.name, p.status 
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
                st.markdown(f"**📍 ตำแหน่ง: `{loc['id']}`**")
                st.progress(loc["used"] / loc["capacity"], text=f"บรรจุ {loc['used']} / {loc['capacity']} พาเลต")
                stored = pallets_by_loc.get(loc["id"], [])
                for s in stored:
                    tag = " (จอง)" if s["status"] == "RESERVED" else ""
                    if st.button(f"📦 {s['code']}{tag} · {s['name'][:10]}...", key=f"btn_{s['code']}"):
                        pallet_detail_dialog(s["code"])
                if not stored:
                    st.caption("— ช่องจัดเก็บว่าง —")

    st.subheader("ค้นหาและกรองสต็อกสินค้า (Inventory Search & Filter)")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        st_filter = st.selectbox("กรองตามสถานะ", ["ทั้งหมด", "พร้อมใช้ (AVAILABLE)", "จองแล้ว (RESERVED)", "พักรับ (RECEIVING)"])
    with f_col2:
        kw = st.text_input("ค้นหารหัสพาเลต หรือ SKU หรือชื่อสินค้า", "")

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
    render_smart_workflow_guide()
    st.caption("กระบวนการจ่ายออก: สร้างใบจองสินค้า → หยิบสินค้า (Pick) → โหลดขึ้นรถบรรทุก (Load) → ปล่อยรถออก")
    st.info("💡 **ขั้นตอนการจ่ายสินค้า:** 1. สร้างใบจ่ายและจองสต็อก (Reserve) ➔ 2. หยิบสินค้า (Pick) ➔ 3. โหลดขึ้นรถ (Load) ➔ 4. ปล่อยรถออกและพิมพ์ใบส่งของ (Depart & Bill of Lading)")

    tab1, tab2 = st.tabs(["🚚 ปฏิบัติการโหลดรถขาออก (Outbound Operations)", "➕ สร้างใบจ่ายและจองสินค้าใหม่ (New Order)"])

    with tab1:
        trips = rows("SELECT s.*, t.name FROM trips s JOIN trucks t ON t.id = s.truck_id WHERE s.kind = 'OUT' AND s.status NOT IN ('DEPARTED', 'CANCELLED')")
        if not trips:
            st.info("ไม่มีเที่ยวรถขาออกที่กำลังดำเนินงาน")
            return

        lookup = {t["id"]: t for t in trips}
        sel_id = st.selectbox("เลือกเที่ยวรถขาออก", list(lookup), format_func=lambda x: f"{lookup[x]['id']} · {lookup[x]['name']} ({lookup[x]['status']} · {lookup[x]['warehouse']})")
        trip = lookup[sel_id]

        c_left, c_right = st.columns([1.1, 1.4])
        with c_left:
            fig = draw_truck_graphic(trip["id"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            st.subheader("Outbound Station")
            manifest = rows("SELECT p.*, pr.name AS product_name, pr.category FROM pallets p JOIN products pr ON p.sku = p.sku WHERE p.outbound_id = ?", (trip["id"],))
            reserved = [m for m in manifest if m["status"] == "RESERVED"]
            picked = [m for m in manifest if m["status"] == "PICKED"]
            loaded = [m for m in manifest if m["status"] == "LOADED"]

            m1, m2, m3 = st.columns(3)
            m1.metric("1. รอหยิบ (Reserved)", len(reserved))
            m2.metric("2. พักจ่าย (Picked)", len(picked))
            m3.metric("3. บนรถ (Loaded)", len(loaded))

            sub_t1, sub_t2, sub_t3 = st.tabs(["1. หยิบสินค้า (Pick)", "2. โหลดขึ้นรถ (Load)", "3. แก้ไข/เอาลง (Unload)"])

            with sub_t1:
                if not reserved:
                    st.success("✅ หยิบสินค้าครบทุกรายการแล้ว (พร้อมสำหรับขั้นตอนการโหลด)")
                else:
                    r_map = {r["code"]: r for r in reserved}
                    with st.form("pick_form"):
                        p_code = st.selectbox("เลือกพาเลตที่ต้องหยิบ", list(r_map), format_func=lambda x: f"{x} · ชั้น {r_map[x]['location']} · {r_map[x]['product_name']}")
                        if st.form_submit_button("🛒 ยืนยันหยิบสินค้าไปพื้นที่พักจ่าย", type="primary", use_container_width=True):
                            try:
                                execute_action("pick", p_code, trip["id"])
                                st.toast(f"🛒 หยิบพาเลต {p_code} ไปยังพื้นที่พักจ่ายแล้ว", icon="✅")
                                st.success(f"หยิบพาเลต {p_code} สำเร็จ")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    if st.button("⚡ หยิบทั้งหมดในคลิกเดียว (Pick All)", use_container_width=True):
                        try:
                            execute_action("pick_all", trip["id"])
                            st.toast("🛒 หยิบสินค้าทั้งหมดเข้าสู่พื้นที่พักจ่ายเรียบร้อย!", icon="⚡")
                            st.success("หยิบสินค้าทั้งหมดเข้าสู่พื้นที่พักจ่ายเรียบร้อย!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            with sub_t2:
                if trip["status"] != "DOCKED":
                    st.warning("⚠️ รถยังไม่ได้เข้าจอดที่ช่องจ่ายสินค้า กรุณานำรถเข้าจอดที่ Dock Yard ก่อน")
                elif not picked:
                    st.info("ไม่มีสินค้าในพื้นที่พักจ่าย (สินค้าอาจยังรอหยิบ หรือโหลดขึ้นรถครบแล้ว)")
                else:
                    pk_map = {p["code"]: p for p in picked}
                    with st.form("load_form"):
                        l_code = st.selectbox("เลือกพาเลตพร้อมโหลด", list(pk_map), format_func=lambda x: f"{x} · {pk_map[x]['product_name']} ({pk_map[x]['weight']} kg)")
                        if st.form_submit_button("📦 โหลดขึ้นรถบรรทุก", type="primary", use_container_width=True):
                            try:
                                execute_action("load", l_code, trip["id"])
                                st.toast(f"📦 โหลดพาเลต {l_code} ขึ้นรถบรรทุกสำเร็จ!", icon="✅")
                                st.success(f"โหลดพาเลต {l_code} ขึ้นรถสำเร็จ!")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    if st.button("⚡ โหลดทั้งหมดในคลิกเดียว (Load All)", use_container_width=True):
                        try:
                            execute_action("load_all", trip["id"])
                            st.toast("📦 โหลดสินค้าขึ้นรถครบถ้วนทุกรายการ!", icon="⚡")
                            st.success("โหลดสินค้าขึ้นรถครบถ้วน!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            with sub_t3:
                if not loaded:
                    st.info("ยังไม่มีพาเลตที่โหลดขึ้นรถ")
                else:
                    ld_map = {l["code"]: l for l in loaded}
                    with st.form("unload_form_outbound"):
                        u_code = st.selectbox("เลือกพาเลตบนรถที่ต้องการนำลง", list(ld_map), format_func=lambda x: f"{x} (ช่อง {ld_map[x]['load_slot']}) · {ld_map[x]['product_name']}")
                        if st.form_submit_button("↩ นำสินค้าลงจากรถกลับสู่พื้นที่พักจ่าย", use_container_width=True):
                            try:
                                execute_action("unload", u_code, trip["id"])
                                st.toast(f"↩ นำพาเลต {u_code} ลงจากรถกลับสู่พื้นที่พักจ่าย", icon="📦")
                                st.success(f"นำพาเลต {u_code} ลงจากรถกลับสู่พื้นที่พักจ่ายเรียบร้อย")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

            if trip["status"] == "DOCKED" and len(loaded) == len(manifest) and len(manifest) > 0:
                with st.expander("🚚 ยืนยันปล่อยรถขาออกออกจากลาน", expanded=True):
                    if st.button("ตรวจสอบครบและปล่อยรถออก (Depart Truck)", type="primary", use_container_width=True):
                        try:
                            execute_action("depart", trip["id"])
                            st.toast("🎉 ปล่อยรถขาออกส่งสินค้าสำเร็จครบทุกชิ้น (Shipment Completed)!", icon="🚚")
                            st.balloons()
                            st.success("รถขาออกเดินทางออกจากคลังเรียบร้อย!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        st.divider()
        col_mf1, col_mf2 = st.columns([2, 1])
        with col_mf1:
            st.markdown("**รายการสินค้าในใบจ่ายนี้ (Order Manifest)**")
        with col_mf2:
            if st.button("พิมพ์ใบส่งของ (Bill of Lading)", key="btn_bol_shipping", use_container_width=True):
                delivery_note_dialog(trip["id"])
        show_pallet_table(manifest)

    with tab2:
        wh_out = st.selectbox("เลือกคลังต้นทาง", WAREHOUSES, key="wh_out_sel")
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
                    t_id = st.selectbox("เลือกรถบรรทุกขนส่ง", list(t_map), format_func=lambda x: f"{x} ({t_map[x]['name']} · {t_map[x]['slots']} ช่อง · รับน้ำหนัก {t_map[x]['max_weight']:,.0f} kg)")
                    partner = st.text_input("ปลายทาง / สาขา", "สาขาเซ็นทรัลเวิลด์")
                with c2:
                    prio = st.selectbox("ความเร่งด่วน", ["High", "Normal", "Low"], key="out_prio")
                    codes = st.multiselect("เลือกพาเลตสินค้าที่จะจ่ายออก", list(s_map), format_func=lambda x: f"{x} · {s_map[x]['product_name']} ({s_map[x]['weight']} kg)")

                selected_weight = sum(s_map[c]["weight"] for c in codes)
                st.caption(f"📊 เลือก: **{len(codes)} / {t_map[t_id]['slots']} ช่อง** · น้ำหนักรวม: **{selected_weight:,.0f} / {t_map[t_id]['max_weight']:,.0f} kg**")

                if st.form_submit_button("สร้างใบจ่ายและจองสินค้าทันที", type="primary"):
                    try:
                        execute_id = create_new_trip("OUT", t_id, wh_out, partner, prio, codes)
                        st.toast(f"📋 สร้างใบจ่ายและจองสต็อก {execute_id} สำเร็จ!", icon="📦")
                        st.success(f"สร้างใบจ่ายสินค้าและจองสต็อกสำเร็จ (รหัส {execute_id})!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# --- PAGE 6: ACTIVITY LOGS ---

def page_barcode_terminal():
    st.header("🔫 Barcode Terminal (สถานีสแกนเนอร์บาร์โค้ดประจำคลัง)")
    st.caption("ศูนย์รวมการยิงสแกนเนอร์จุดเดียวของระบบ WMS · รองรับทั้งเครื่องยิงบาร์โค้ด (Barcode Gun) และกล้อง OpenCV · บันทึกการเคลื่อนย้ายลงฐานข้อมูลจริง")

    render_smart_workflow_guide()

    # Session state for scan history
    if "barcode_scan_history" not in st.session_state:
        st.session_state.barcode_scan_history = []
    if "barcode_last_result" not in st.session_state:
        st.session_state.barcode_last_result = None

    # Step 1: Work Task Selector
    st.markdown("#### 1. เลือกขั้นตอนงานที่ต้องการสแกน (Select Warehouse Task)")
    
    # Calculate live queue counts for each task
    inbound_count = scalar("SELECT COUNT(*) FROM pallets p JOIN trips s ON s.id = p.inbound_id WHERE s.status = 'DOCKED' AND p.status = 'ON_INBOUND'")
    receiving_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RECEIVING'")
    reserved_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'RESERVED'")
    picked_count = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'PICKED'")
    total_active = scalar("SELECT COUNT(*) FROM pallets WHERE status NOT IN ('SHIPPED')")

    task_choices = [
        f"📥 1. ขนถ่ายรับของลงรถ (Receive) — [รอสแกน {inbound_count} พาเลต]",
        f"📍 2. จัดเก็บเข้าชั้นวาง (Putaway) — [รอสแกน {receiving_count} พาเลต]",
        f"🛒 3. หยิบสินค้าเตรียมจ่าย (Pick) — [รอสแกน {reserved_count} พาเลต]",
        f"📦 4. โหลดสินค้าขึ้นรถ (Load) — [รอสแกน {picked_count} พาเลต]",
        f"🔍 5. สแกนตรวจสอบทั่วไป (Check & Audit) — [สต็อก {total_active} พาเลต]",
    ]

    selected_task_label = st.radio(
        "เลือกงานที่กำลังปฏิบัติการ:",
        task_choices,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Determine task mode code
    if "1. ขนถ่ายรับของ" in selected_task_label:
        current_mode = "RECEIVE"
        task_name = "ขนถ่ายสินค้าลงจากรถขาเข้า"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category FROM pallets p JOIN trips s ON s.id = p.inbound_id JOIN products pr ON pr.sku = p.sku WHERE s.status = 'DOCKED' AND p.status = 'ON_INBOUND' ORDER BY p.inbound_slot ASC"
        expected_status_hint = "สินค้าบนรถขาเข้า ➔ ขนลงสู่พื้นที่พักรับ"
    elif "2. จัดเก็บเข้าชั้นวาง" in selected_task_label:
        current_mode = "PUTAWAY"
        task_name = "จัดเก็บสินค้าจากพื้นที่พักรับเข้าชั้นวาง"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'RECEIVING' ORDER BY p.code ASC"
        expected_status_hint = "สินค้าพื้นที่พักรับ ➔ จัดเก็บขึ้นชั้นวาง (พร้อมใช้)"
    elif "3. หยิบสินค้า" in selected_task_label:
        current_mode = "PICK"
        task_name = "หยิบสินค้าจากชั้นวางไปพื้นที่พักจ่าย"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'RESERVED' ORDER BY p.code ASC"
        expected_status_hint = "สินค้าจองบนชั้น ➔ หยิบไปพื้นที่พักจ่าย"
    elif "4. โหลดสินค้าขึ้นรถ" in selected_task_label:
        current_mode = "LOAD"
        task_name = "โหลดสินค้าจากพื้นที่พักจ่ายขึ้นรถบรรทุก"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status = 'PICKED' ORDER BY p.code ASC"
        expected_status_hint = "สินค้าพื้นที่พักจ่าย ➔ โหลดบรรจุขึ้นรถขาออก"
    else:
        current_mode = "LOOKUP"
        task_name = "ตรวจสอบข้อมูลและตำแหน่งพาเลต"
        target_query = "SELECT p.*, pr.name AS prod_name, pr.category FROM pallets p JOIN products pr ON pr.sku = p.sku WHERE p.status NOT IN ('SHIPPED') ORDER BY p.code ASC LIMIT 12"
        expected_status_hint = "สแกนเพื่อตรวจสอบข้อมูล พิกัด และตำแหน่งจัดเก็บ"

    st.divider()

    # Step 2: Dual-Pane Interactive Workspace
    col_scanner, col_pallets = st.columns([1.1, 1.3])

    with col_scanner:
        st.markdown(f"#### 2. สถานีสแกนเนอร์: {task_name}")
        st.caption(f"🎯 **การเคลื่อนย้าย:** {expected_status_hint}")

        # Barcode Gun Input Form
        with st.form("quick_barcode_gun_form", clear_on_submit=True):
            scan_code_input = st.text_input(
                "ยิงสแกนเนอร์บาร์โค้ด (หรือพิมพ์รหัสพาเลต):",
                placeholder="เช่น PAL-001, RCV-A-01 แล้วกด Enter หรือคลิกปุ่ม...",
                key="gun_input_box"
            )
            submit_scan = st.form_submit_button(
                "🔫 ยิงสแกนเนอร์และบันทึกฐานข้อมูล (Scan Barcode)", 
                type="primary", 
                use_container_width=True
            )

        # Camera Option with OpenCV
        with st.expander("📷 เปิดกล้องสแกนบาร์โค้ดจริง (Camera Scanner with OpenCV)"):
            st.caption("ถ่ายรูปบาร์โค้ดบนกล่อง/หน้าจอ ระบบจะถอดรหัสบาร์โค้ดด้วย OpenCV อัตโนมัติ")
            cam_image = st.camera_input("ถ่ายภาพบาร์โค้ด", key="cam_barcode_reader")
            if cam_image is not None:
                decoded = decode_barcode_image(cam_image.getvalue())
                if decoded:
                    st.success(f"🎯 กล้องตรวจพบบาร์โค้ด: **{decoded}**")
                    if st.button(f"⚡ ยิงรหัส {decoded} เข้าระบบทันที", type="primary"):
                        scan_code_input = decoded
                        submit_scan = True
                else:
                    st.warning("ไม่พบบาร์โค้ดในภาพ กรุณาถือกล้องให้นิ่งและมีแสงสว่างชัดเจน")

        # Handle Scan Action
        if submit_scan and scan_code_input.strip():
            target_c = scan_code_input.strip().upper()
            try:
                # If auto or specific mode
                res = execute_barcode_scan(target_c, action_mode="AUTO" if current_mode != "LOOKUP" else "LOOKUP")
                st.session_state.barcode_last_result = res
                st.session_state.barcode_scan_history.insert(0, {
                    "เวลา": timestamp(),
                    "รหัสบาร์โค้ด": res["code"],
                    "กิจกรรม": res["action"],
                    "สถานะเดิม": PALLET_STATUS.get(res["prev_status"], res["prev_status"]),
                    "สถานะใหม่": PALLET_STATUS.get(res["new_status"], res["new_status"]),
                    "ข้อความ": res["message"]
                })
                st.toast(f"✅ บันทึกสำเร็จ: {res['message']}", icon="🔫")
                st.rerun()
            except Exception as e:
                st.error(f"❌ การสแกนล้มเหลว: {str(e)}")

        # Last Scanned Result Card
        last_res = st.session_state.barcode_last_result
        if last_res:
            with st.container(border=True):
                st.markdown("<div style='color:#059669; font-weight:bold; font-size:13px;'>✅ ผลการสแกนล่าสุด (Scan Verified)</div>", unsafe_allow_html=True)
                r_c1, r_c2 = st.columns([1, 1.4])
                with r_c1:
                    st.markdown(generate_barcode_svg(last_res["code"]), unsafe_allow_html=True)
                with r_c2:
                    st.markdown(f"**รหัส:** `{last_res['code']}`")
                    p_info = last_res["pallet"]
                    st.markdown(f"**สินค้า:** {p_info.get('sku', '')}")
                    if last_res["prev_status"] != last_res["new_status"]:
                        st.markdown(f"**สถานะ:** `{PALLET_STATUS.get(last_res['prev_status'], last_res['prev_status'])}` ➔ **`{PALLET_STATUS.get(last_res['new_status'], last_res['new_status'])}`**")
                    else:
                        st.markdown(f"**สถานะ:** `{PALLET_STATUS.get(last_res['new_status'], last_res['new_status'])}`")
                    st.caption(f"ตำแหน่ง: {p_info['location'] or 'พื้นที่พัก / บนรถ'} · คลัง {p_info['warehouse']}")
                st.success(f"📋 {last_res['message']}")

    with col_pallets:
        st.markdown(f"#### 3. รายการพาเลตที่รอให้คุณยิงสแกน (Queue Ready to Scan)")
        st.caption("💡 คุณสามารถเอาเครื่องยิงบาร์โค้ดยิงรหัสที่เห็น หรือ **กดปุ่ม [🔫 ยิงสแกนชิ้นนี้]** เพื่อจำลองการทำงานทันที")

        waiting_pallets = rows(target_query)
        if not waiting_pallets:
            st.success(f"🎉 **ไม่มีสินค้าตกค้างในขั้นตอนนี้แล้ว!** (งาน {task_name} ดำเนินการครบ 100%)")
            st.info("คุณสามารถเลือกทำงานในขั้นตอนถัดไปจากแถบด้านบน หรือดูประวัติการสแกนด้านล่างได้ครับ")
        else:
            st.caption(f"พบสินค้าที่พร้อมสแกนทั้งหมด **{len(waiting_pallets)} พาเลต**:")
            # Display items in interactive clean cards
            for pal in waiting_pallets:
                with st.container(border=True):
                    pc1, pc2, pc3 = st.columns([1.1, 1.4, 1.1])
                    with pc1:
                        st.markdown(f"**📦 {pal['code']}**")
                        st.caption(f"SKU: {pal['sku']}")
                        st.caption(f"น้ำหนัก: {pal['weight']:,.0f} kg")
                    with pc2:
                        st.markdown(generate_barcode_svg(pal['code']), unsafe_allow_html=True)
                        cur_st = PALLET_STATUS.get(pal['status'], pal['status'])
                        loc_txt = f"ชั้น {pal['location']}" if pal.get('location') else (f"ช่องรถ {pal.get('inbound_slot')}" if pal.get('inbound_slot') else "พื้นที่พัก")
                        st.caption(f"สถานะ: `{cur_st}` · {loc_txt}")
                    with pc3:
                        st.write("")
                        btn_label = "🔫 ยิงสแกนชิ้นนี้" if current_mode != "LOOKUP" else "🔍 สแกนเช็ก"
                        if st.button(btn_label, key=f"btn_scan_queue_{pal['code']}", type="primary", use_container_width=True):
                            try:
                                res = execute_barcode_scan(pal["code"], action_mode="AUTO" if current_mode != "LOOKUP" else "LOOKUP")
                                st.session_state.barcode_last_result = res
                                st.session_state.barcode_scan_history.insert(0, {
                                    "เวลา": timestamp(),
                                    "รหัสบาร์โค้ด": res["code"],
                                    "กิจกรรม": res["action"],
                                    "สถานะเดิม": PALLET_STATUS.get(res["prev_status"], res["prev_status"]),
                                    "สถานะใหม่": PALLET_STATUS.get(res["new_status"], res["new_status"]),
                                    "ข้อความ": res["message"]
                                })
                                st.toast(f"✅ บันทึกสำเร็จ: {res['message']}", icon="🔫")
                                st.rerun()
                            except Exception as ex:
                                st.error(str(ex))

    # Bottom Section: Scan History Table
    st.divider()
    st.subheader("📜 ประวัติการยิงสแกนบาร์โค้ดสด (Live Barcode Scan Audit Trail)")
    if st.session_state.barcode_scan_history:
        st.dataframe(pd.DataFrame(st.session_state.barcode_scan_history), use_container_width=True, hide_index=True)
        if st.button("ล้างประวัติการสแกนบนหน้าจอนี้"):
            st.session_state.barcode_scan_history = []
            st.session_state.barcode_last_result = None
            st.rerun()
    else:
        st.info("ยังไม่มีประวัติการยิงสแกนในเซสชันนี้ — เริ่มต้นสแกนพาเลตทางด้านบนได้เลยครับ")



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
        st.subheader("ประวัติเที่ยวรถที่เสร็จสิ้นภารกิจแล้ว (Completed Trips History)")
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
# 7. MAIN CONTROLLER & SIDEBAR
# =========================================================

initialize_database()

with st.sidebar:
    st.markdown("## 🚚 DOCKFLOW WMS")
    st.caption("ศูนย์ควบคุมคลังสินค้าและลานจอดรถอัจฉริยะ (Smart Command Center)")
    st.markdown("🟢 **สถานะ:** ออนไลน์")
    st.divider()

    # 1. Live Snapshot Container
    with st.container(border=True):
        st.markdown("**Status showcase**")
        c_sb1, c_sb2, c_sb3 = st.columns(3)
        sb_docked = scalar("SELECT COUNT(*) FROM trips WHERE status = 'DOCKED'")
        sb_wait = scalar("SELECT COUNT(*) FROM trips WHERE status = 'WAITING'")
        sb_avail = scalar("SELECT COUNT(*) FROM pallets WHERE status = 'AVAILABLE'")
        c_sb1.metric("เทียบท่า", f"{sb_docked}/4")
        c_sb2.metric("รอคิว", f"{sb_wait}")
        c_sb3.metric("พร้อมใช้", f"{sb_avail}")

    # 2. Modern Button Navigation with Text Line
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Control Tower"

    MENU = [
        ("Control Tower", "📊 1. Control Tower", "ศูนย์ควบคุม & มอนิเตอร์สด"),
        ("Dock Yard", "🅿️ 2. Dock Yard", "จัดการคิวรถ & ช่องจอด"),
        ("Smart Receiving", "📥 3. Smart Receiving", "รับสินค้าเข้า & ขนถ่ายลง"),
        ("Warehouse Mapping", "🏢 4. Warehouse Mapping", "ผังคลังสินค้า & พิกัดชั้นวาง"),
        ("Shipping & Outbound", "📤 5. Shipping & Outbound", "จ่ายออก หยิบ & โหลดรถ"),
        ("Barcode Terminal", "🔫 6. Barcode Terminal", "สถานีสแกนเนอร์บาร์โค้ดหน้างาน"),
        ("Activity Logs", "📋 7. Activity Logs", "ประวัติย้อนรอย & ส่งออก CSV"),
    ]

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div.stButton > button {
            text-align: left;
            display: flex;
            justify-content: flex-start;
            padding-left: 14px;
            font-weight: 600;
            font-size: 0.92rem;
            border-radius: 8px;
            margin-bottom: 2px;
        }
        div[data-testid="stSidebar"] .nav-desc {
            font-size: 0.74rem;
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

st.divider()
st.caption("DOCKFLOW WMS · Python + Streamlit Complete Project · สถาปัตยกรรมคลังสินค้าและลานจอดรถอัจฉริยะ")