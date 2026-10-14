# 🚚 DOCKFLOW WMS
**ระบบบริหารจัดการคลังสินค้าและลานเทียบท่าอัจฉริยะ (Smart Warehouse & Yard Management System)**

DOCKFLOW WMS เป็นเว็บแอปพลิเคชันบริหารจัดการคลังสินค้าและลานเทียบท่าอัจฉริยะ พัฒนาด้วย **Python** และ **Streamlit** ผสานกระบวนการของ **WMS (Warehouse Management System)** และ **YMS (Yard Management System)** เข้าด้วยกันอย่างสมบูรณ์

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **📊 Control Tower:** ศูนย์ควบคุมสถานะภาพรวม Telemetry อัปเดตสดทุก 3 วินาที พร้อมชาร์ตวิเคราะห์สต็อกและความจุคลัง
2. **🅿️ Dock Yard & Fleet Management:**
   - จัดการคิวรถรอเทียบท่า และควบคุมช่องเทียบท่า (Docks D01–D04)
   - ทะเบียนยานพาหนะและกองรถ (Fleet Registry) บันทึกป้ายทะเบียน บริษัทขนส่ง พนักงานขับรถ และประเภทรถ
   - ระบบ **Gate-In**: เปิดเที่ยวงานและนำรถเข้าคิวเทียบท่าได้โดยตรง
3. **📥 Smart Receiving & Inbound:** ขนถ่ายสินค้าลงจากรถสู่พื้นที่พักรับ (Staging Area) และบันทึกปล่อยรถออกจากลาน
4. **🏢 Warehouse Mapping & Analytics:**
   - ผังตำแหน่งชั้นวางสินค้าแบบไดนามิก (Rack Layout R01–R06)
   - **ABC Velocity Slotting Matrix:** วิเคราะห์ความเร็วในการหมุนเวียนสินค้า พร้อมเครื่องมือย้ายตำแหน่ง (Relocate Tool)
   - **FEFO Expiry Tracker:** ระบบเฝ้าระวังวันหมดอายุล่วงหน้า (First-Expired, First-Out) แจ้งเตือน 3 ระดับ
5. **📤 Shipping & Outbound:** สร้างใบเบิกจ่าย จองสต็อก หยิบ โหลดขึ้นรถ และพิมพ์ใบส่งมอบสินค้าสากล (**Bill of Lading**)
6. **🔫 Barcode Terminal:** สถานีสแกนเนอร์บาร์โค้ด รองรับทั้งเครื่องสแกนบาร์โค้ด USB/Bluetooth และกล้องถ่ายภาพ (Camera Scanner) พร้อมเสียงสังเคราะห์ (Web Audio API)
7. **📋 Activity History & Audit Trail:** บันทึกประวัติการดำเนินงานทุกกิจกรรม และส่งออกรายงานเป็น CSV

---

## 🚀 วิธีการติดตั้งและรันระบบ (Installation & Run)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันแอปพลิเคชัน
```bash
streamlit run main.py
```
เปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```
DockFlow/
├── main.py                 # ซอร์สโค้ดหลักของระบบทั้งหมด
├── requirements.txt        # รายการไลบรารีที่จำเป็น
├── dockflow_complete.db    # ฐานข้อมูล SQLite (สร้างอัตโนมัติเมื่อเริ่มระบบ)
├── PRESENTATION_GUIDE.md   # คู่มือการนำเสนอและสรุป Flow การทำงาน
└── README.md               # เอกสารแนะนำโปรเจกต์
```
