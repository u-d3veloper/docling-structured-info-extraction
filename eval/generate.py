"""Generate a synthetic evaluation dataset of maritime Bills of Lading.

Produces, under ``eval/``:

- ``ground_truth/<id>.json``: the authoritative values, exactly matching the
  fields of ``schemas/base.yaml``.
- ``clean/<id>.pdf``: a one-page PDF rendered from the JSON with one of three
  layout templates (positions, labels and fonts differ).
- ``scanned/<id>.pdf``: the clean PDF rasterised at 175 dpi and degraded
  (rotation, Gaussian noise, slight blur, JPEG compression) at either a
  ``light`` or a ``heavy`` level, then re-wrapped as an image-only PDF
  (no text layer).
- ``manifest.csv``: one row per document with id, template, degradation level,
  and which traps are present.

Run from the repo root::

    uv run python eval/generate.py

Deterministic: all randomness is seeded.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from faker import Faker
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

SEED = 20261117
N_DOCS = 30
RASTER_DPI = 175

ROOT = Path(__file__).resolve().parent
GT_DIR = ROOT / "ground_truth"
CLEAN_DIR = ROOT / "clean"
SCAN_DIR = ROOT / "scanned"
MANIFEST_PATH = ROOT / "manifest.csv"


# ----------------------------------------------------------------------------
# ISO 6346 check digit
# ----------------------------------------------------------------------------

# Letter → numerical value, skipping multiples of 11 (11, 22, 33).
_LETTER_VALUES = {}
_v = 10
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if _v % 11 == 0:
        _v += 1
    _LETTER_VALUES[_c] = _v
    _v += 1


def iso6346_check_digit(prefix: str) -> int:
    """Return the ISO 6346 check digit for a 10-char prefix (4 letters + 6 digits)."""
    if len(prefix) != 10 or not prefix[:4].isalpha() or not prefix[4:].isdigit():
        raise ValueError(f"Invalid ISO 6346 prefix: {prefix!r}")
    total = 0
    for i, ch in enumerate(prefix.upper()):
        value = _LETTER_VALUES[ch] if ch.isalpha() else int(ch)
        total += value * (1 << i)
    return total % 11 % 10  # 10 wraps to 0


def _test_iso6346() -> None:
    """Round-trip + 3 known-good numbers (Maersk, Wikipedia worked example, synthetic)."""
    assert iso6346_check_digit("MRKU706178") == 4, "MRKU706178 should give 4"
    assert iso6346_check_digit("CSQU305438") == 3, "CSQU305438 should give 3"
    assert iso6346_check_digit("TEMU123456") == 5, "TEMU123456 should give 5"
    # Round-trip check with a value we build ourselves.
    rng = random.Random(0)
    for _ in range(50):
        pfx = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)) + "U"
        pfx += "".join(str(rng.randint(0, 9)) for _ in range(6))
        cd = iso6346_check_digit(pfx)
        full = pfx + str(cd)
        # The full number is self-consistent: computing the check digit of its
        # first 10 chars must give back the same digit.
        assert iso6346_check_digit(full[:10]) == int(full[10])


# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------

# 20 real ports with UN/LOCODE.
PORTS = [
    ("Singapore", "SGSIN"),
    ("Rotterdam", "NLRTM"),
    ("Shanghai", "CNSHA"),
    ("Hamburg", "DEHAM"),
    ("Antwerp", "BEANR"),
    ("Los Angeles", "USLAX"),
    ("Long Beach", "USLGB"),
    ("Hong Kong", "HKHKG"),
    ("Busan", "KRPUS"),
    ("Dubai (Jebel Ali)", "AEJEA"),
    ("Yokohama", "JPYOK"),
    ("Ningbo", "CNNGB"),
    ("Felixstowe", "GBFXT"),
    ("Qingdao", "CNTAO"),
    ("Tianjin", "CNTSN"),
    ("Valencia", "ESVLC"),
    ("Le Havre", "FRLEH"),
    ("New York", "USNYC"),
    ("Oakland", "USOAK"),
    ("Piraeus", "GRPIR"),
]

# 20 plausible vessel names (mix of invented + real-sounding line names).
VESSELS = [
    "MV NORTHERN STAR",
    "MV PACIFIC VOYAGER",
    "MSC ANNA",
    "MV BALTIC WIND",
    "MV ATLANTIC SPIRIT",
    "MV SOUTHERN CROSS",
    "CMA CGM MARCO POLO",
    "MV OCEAN HARMONY",
    "MV EMERALD SEA",
    "EVER GIVEN",
    "EVER GLORY",
    "MV MARITIME PRIDE",
    "MAERSK ESSEN",
    "MSC GULSUN",
    "MV BLUE HORIZON",
    "MV GOLDEN GATE",
    "MV CORAL STAR",
    "MV SILVER ARROW",
    "HAPAG EXPRESS",
    "MV COPENHAGEN",
]

# Real 4-letter container owner/category codes (3-letter owner + U).
OWNER_CODES = [
    "MRKU", "MSKU", "MSCU", "TGHU", "TCLU", "TRIU", "TEMU", "BEAU",
    "SEGU", "GESU", "FCIU", "CAIU", "CMAU", "APZU", "OOLU", "EGHU",
    "EMCU", "MAEU", "HLXU", "HLBU",
]

# Real shipping-line / carrier names, one per BL.
CARRIERS = [
    "OCEANLINE SHIPPING CO.",
    "MAERSK LINE",
    "MSC MEDITERRANEAN SHIPPING COMPANY",
    "CMA CGM",
    "HAPAG-LLOYD AG",
    "EVERGREEN MARINE CORP.",
    "ONE (OCEAN NETWORK EXPRESS)",
    "COSCO SHIPPING LINES",
    "YANG MING MARINE TRANSPORT",
    "ZIM INTEGRATED SHIPPING",
]

# Commodity descriptions for the ``lots`` field.
COMMODITIES = [
    "pallets electronic components",
    "pallets textile goods",
    "pallets automotive parts",
    "pallets machinery spare parts",
    "cartons frozen seafood",
    "drums industrial lubricant",
    "pallets ceramic tiles",
    "bags rice (jasmine)",
    "cartons pharmaceutical products",
    "pallets solar panels",
    "bales cotton fabric",
    "crates furniture",
    "pallets packaged food",
    "cartons toys",
    "pallets steel coils",
]

# Date format variants used when the ``date_format`` trap is set.
DATE_FORMATS = ["%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%B %d, %Y"]

# Weight unit variants (for on-document distractors).
WEIGHT_FORMATS = [
    "{:,} KG",
    "{:,} KGS",
    "{} kg",
    "{} KGS",
]


# ----------------------------------------------------------------------------
# Ground-truth generation
# ----------------------------------------------------------------------------


@dataclass
class DocSpec:
    """Context for one generated document: ground truth + rendering hints."""

    doc_id: str
    template: int  # 1, 2 or 3
    degrade_level: str  # "light" or "heavy"
    traps: dict  # which traps are active
    data: dict  # schema fields → values
    # Render-only extras (consignee, notify party, weights) printed on the PDF
    # but NOT in the schema / ground truth.
    extras: dict


def _container_number(rng: random.Random) -> str:
    owner = rng.choice(OWNER_CODES)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(6))
    return owner + digits + str(iso6346_check_digit(owner + digits))


def _format_date(d, fmt: str) -> str:
    """Format a datetime.date with uppercase month abbreviations where applicable."""
    s = d.strftime(fmt)
    if fmt == "%d %b %Y":  # "14 OCT 2026"
        s = s.upper()
    return s


def build_doc_spec(index: int, rng: random.Random, faker: Faker) -> DocSpec:
    """Build the ground truth and rendering context for one document."""
    doc_id = f"bl_{index:03d}"
    template = (index % 3) + 1  # 10 of each template across 30 docs
    degrade_level = "light" if index % 2 == 0 else "heavy"

    # Traps: active on ~1/3 of docs. We pick a reproducible subset.
    traps_active = (index * 7 + 1) % 3 == 0
    traps = {
        "consignee_to_order": traps_active,
        "notify_same_as_consignee": traps_active,
        "date_format": DATE_FORMATS[index % len(DATE_FORMATS)] if traps_active else "%d %b %Y",
        "weight_format": WEIGHT_FORMATS[index % len(WEIGHT_FORMATS)] if traps_active else WEIGHT_FORMATS[0],
    }

    carrier = CARRIERS[index % len(CARRIERS)]
    bl_number = f"{carrier.split()[0][:4].upper()}-{2026}-{rng.randint(100000, 999999):06d}"

    # Shipper: realistic company address built from Faker.
    shipper_name = faker.company()
    shipper_addr = faker.street_address()
    shipper_city = faker.city()
    shipper_country = faker.country()
    shipper = f"{shipper_name}, {shipper_addr}, {shipper_city}, {shipper_country}"

    # Consignee / notify party (NOT in the schema — printed on the PDF).
    if traps["consignee_to_order"]:
        consignee = "TO ORDER"
    else:
        consignee = (
            f"{faker.company()}, {faker.street_address()}, "
            f"{faker.city()}, {faker.country()}"
        )
    if traps["notify_same_as_consignee"]:
        notify = "SAME AS CONSIGNEE"
    else:
        notify = (
            f"{faker.company()}, {faker.street_address()}, "
            f"{faker.city()}, {faker.country()}"
        )

    # Ports (distinct for loading / discharge).
    pol, pod = rng.sample(PORTS, 2)
    port_of_loading = f"{pol[0]} ({pol[1]})"
    port_of_discharge = f"{pod[0]} ({pod[1]})"
    place_of_acceptance = f"{pol[0]} CY"
    place_of_delivery = f"{pod[0]} CY"

    vessel = rng.choice(VESSELS)
    voyage = f"V-{rng.randint(100, 999)}{rng.choice('EW')}"

    # Date: pick a day in 2026; string format depends on trap.
    day = rng.randint(1, 365)
    base = datetime.date(2026, 1, 1) + datetime.timedelta(days=day - 1)
    date_str = _format_date(base, traps["date_format"])

    # Mark: short code that looks like a shipping mark.
    mark = f"{shipper_name.split()[0].upper()[:4]}/{pod[1][-3:]}/LOT-{rng.choice('ABCD')}"

    # 1 to 4 containers.
    n_containers = rng.randint(1, 4)
    container_numbers = [_container_number(rng) for _ in range(n_containers)]

    # One lot description per container.
    lots = [
        f"{rng.randint(10, 40)} {rng.choice(COMMODITIES)}"
        for _ in range(n_containers)
    ]

    # Per-container gross weight (kg) — only printed on the PDF.
    weights = [rng.randint(8_000, 28_000) for _ in range(n_containers)]
    weight_strs = [traps["weight_format"].format(w) for w in weights]

    data = {
        "company_name": carrier,
        "bl_number": bl_number,
        "shipper": shipper,
        "port_of_loading": port_of_loading,
        "port_of_discharge": port_of_discharge,
        "place_of_acceptance": place_of_acceptance,
        "place_of_delivery": place_of_delivery,
        "shipped_on_board_date": date_str,
        "vessel_name": vessel,
        "voyage_number": voyage,
        "mark": mark,
        "container_numbers": container_numbers,
        "lots": lots,
    }
    extras = {
        "consignee": consignee,
        "notify_party": notify,
        "weights": weight_strs,
    }
    return DocSpec(
        doc_id=doc_id,
        template=template,
        degrade_level=degrade_level,
        traps=traps,
        data=data,
        extras=extras,
    )


# ----------------------------------------------------------------------------
# PDF rendering — three layout templates
# ----------------------------------------------------------------------------


def _wrap(s: str, width: int) -> list[str]:
    """Trivial word-wrap used to fit addresses in a fixed-width box."""
    words = s.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_template_1(c: canvas.Canvas, spec: DocSpec) -> None:
    """Classic two-column header, table at the bottom. Helvetica throughout."""
    w, h = A4
    d, x = spec.data, spec.extras
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, d["company_name"])
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 68, f"B/L NO.  {d['bl_number']}")

    # Left column
    y = h - 110
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Shipper")
    c.setFont("Helvetica", 9)
    for line in _wrap(d["shipper"], 55):
        y -= 12
        c.drawString(40, y, line)

    y -= 20
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Consignee")
    c.setFont("Helvetica", 9)
    for line in _wrap(x["consignee"], 55):
        y -= 12
        c.drawString(40, y, line)

    y -= 20
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Notify Party")
    c.setFont("Helvetica", 9)
    for line in _wrap(x["notify_party"], 55):
        y -= 12
        c.drawString(40, y, line)

    # Right column (ports, vessel, dates)
    rx = 330
    ry = h - 110
    pairs = [
        ("Vessel", d["vessel_name"]),
        ("Voyage No.", d["voyage_number"]),
        ("Port of Loading", d["port_of_loading"]),
        ("Port of Discharge", d["port_of_discharge"]),
        ("Place of Receipt", d["place_of_acceptance"]),
        ("Place of Delivery", d["place_of_delivery"]),
        ("Shipped on Board Date", d["shipped_on_board_date"]),
        ("Shipping Mark", d["mark"]),
    ]
    for label, value in pairs:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(rx, ry, label)
        c.setFont("Helvetica", 9)
        c.drawString(rx + 110, ry, value)
        ry -= 16

    # Table of containers
    ty = 320
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, ty, "Container No.")
    c.drawString(200, ty, "Description of Goods")
    c.drawString(450, ty, "Gross Weight")
    c.line(40, ty - 3, 555, ty - 3)
    ty -= 18
    c.setFont("Helvetica", 9)
    for cn, lot, wt in zip(d["container_numbers"], d["lots"], x["weights"]):
        c.drawString(40, ty, cn)
        c.drawString(200, ty, lot)
        c.drawString(450, ty, wt)
        ty -= 14


def render_template_2(c: canvas.Canvas, spec: DocSpec) -> None:
    """Single-column labelled key/value layout. Times-Roman family."""
    w, h = A4
    d, x = spec.data, spec.extras
    c.setFont("Times-Bold", 18)
    c.drawCentredString(w / 2, h - 50, "BILL OF LADING")
    c.setFont("Times-Roman", 10)
    c.drawCentredString(w / 2, h - 68, d["company_name"])
    c.drawCentredString(w / 2, h - 82, f"Document reference: {d['bl_number']}")

    y = h - 120
    pairs = [
        ("Shipper / Exporter", d["shipper"]),
        ("Consignee", x["consignee"]),
        ("Notify Address", x["notify_party"]),
        ("Place of Receipt", d["place_of_acceptance"]),
        ("Port of Loading", d["port_of_loading"]),
        ("Ocean Vessel / Voyage No.", f"{d['vessel_name']} / {d['voyage_number']}"),
        ("Port of Discharge", d["port_of_discharge"]),
        ("Place of Delivery", d["place_of_delivery"]),
        ("On-Board Date", d["shipped_on_board_date"]),
        ("Marks & Numbers", d["mark"]),
    ]
    for label, value in pairs:
        c.setFont("Times-Bold", 9)
        c.drawString(50, y, f"{label}:")
        c.setFont("Times-Roman", 10)
        # Wrap long values across the single wide column.
        lines = _wrap(value, 80)
        c.drawString(190, y, lines[0] if lines else "")
        for extra in lines[1:]:
            y -= 12
            c.drawString(190, y, extra)
        y -= 18

    # Cargo block
    y -= 8
    c.setFont("Times-Bold", 11)
    c.drawString(50, y, "Particulars Furnished by Shipper")
    y -= 14
    c.setFont("Times-Roman", 10)
    for cn, lot, wt in zip(d["container_numbers"], d["lots"], x["weights"]):
        c.drawString(50, y, f"Container {cn}  -  {lot}  -  Gross wt. {wt}")
        y -= 14


def render_template_3(c: canvas.Canvas, spec: DocSpec) -> None:
    """Dense top-labels-in-caps layout. Mixed Helvetica + Courier for codes."""
    w, h = A4
    d, x = spec.data, spec.extras

    # Top band with carrier + BL number aligned right.
    c.setFillGray(0.92)
    c.rect(0, h - 70, w, 70, fill=1, stroke=0)
    c.setFillGray(0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h - 40, d["company_name"])
    c.setFont("Courier-Bold", 11)
    c.drawRightString(w - 30, h - 40, f"BL# {d['bl_number']}")

    # Field groups (labels ALL CAPS, small, above values)
    def group(label: str, value: str, x0: float, y0: float, max_w: int = 60) -> None:
        c.setFont("Helvetica-Bold", 7)
        c.drawString(x0, y0, label.upper())
        c.setFont("Helvetica", 9)
        for i, line in enumerate(_wrap(value, max_w)):
            c.drawString(x0, y0 - 12 - 11 * i, line)

    y = h - 110
    group("From (Shipper)", d["shipper"], 30, y, 50)
    group("To (Consignee)", x["consignee"], 300, y, 50)

    y -= 90
    group("Notify", x["notify_party"], 30, y, 50)
    group("Shipping Mark", d["mark"], 300, y, 50)

    y -= 70
    group("Vessel", d["vessel_name"], 30, y)
    group("Voyage", d["voyage_number"], 180, y)
    group("Loading Port", d["port_of_loading"], 300, y)
    group("Discharge Port", d["port_of_discharge"], 430, y)

    y -= 40
    group("Place of Receipt", d["place_of_acceptance"], 30, y)
    group("Place of Delivery", d["place_of_delivery"], 180, y)
    group("On Board", d["shipped_on_board_date"], 300, y)

    # Container list, Courier for the numbers.
    y -= 50
    c.setFont("Helvetica-Bold", 8)
    c.drawString(30, y, "CONTAINERS & GOODS")
    y -= 14
    for cn, lot, wt in zip(d["container_numbers"], d["lots"], x["weights"]):
        c.setFont("Courier-Bold", 10)
        c.drawString(30, y, cn)
        c.setFont("Helvetica", 9)
        c.drawString(130, y, lot)
        c.drawRightString(w - 30, y, wt)
        y -= 14


TEMPLATES = {1: render_template_1, 2: render_template_2, 3: render_template_3}


def render_clean_pdf(spec: DocSpec, out_path: Path) -> None:
    """Render the ground truth for ``spec`` as a clean, single-page PDF."""
    c = canvas.Canvas(str(out_path), pagesize=A4)
    TEMPLATES[spec.template](c, spec)
    c.showPage()
    c.save()


# ----------------------------------------------------------------------------
# Rasterisation + degradation
# ----------------------------------------------------------------------------


DEGRADE_PARAMS = {
    "light": {
        "max_rotation_deg": 1.0,
        "noise_sigma": 3.0,
        "blur_radius": 0.4,
        "jpeg_quality": 80,
    },
    "heavy": {
        "max_rotation_deg": 2.0,
        "noise_sigma": 8.0,
        "blur_radius": 1.2,
        "jpeg_quality": 40,
    },
}


def _degrade_image(img: Image.Image, params: dict, rng: random.Random) -> Image.Image:
    """Apply rotation / noise / blur / JPEG re-compression to a page image."""
    # Rotation around the center on a white background.
    angle = rng.uniform(-params["max_rotation_deg"], params["max_rotation_deg"])
    img = img.rotate(
        angle,
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=(255, 255, 255),
    )

    # Gaussian noise.
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(0.0, params["noise_sigma"], arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Slight blur.
    if params["blur_radius"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))

    # JPEG re-compression round-trip.
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=params["jpeg_quality"])
    buf.seek(0)
    return Image.open(buf).copy()


def rasterize_and_degrade(clean_pdf: Path, out_pdf: Path, level: str, seed: int) -> None:
    """Rasterise ``clean_pdf`` then wrap degraded pages as an image-only PDF."""
    params = DEGRADE_PARAMS[level]
    rng = random.Random(seed)
    np.random.seed(seed)

    doc = pymupdf.open(clean_pdf)
    out = pymupdf.open()
    try:
        for page in doc:
            # Rasterise at the chosen DPI (72 is the PDF base).
            zoom = RASTER_DPI / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            degraded = _degrade_image(img, params, rng)

            jpeg_buf = io.BytesIO()
            degraded.convert("RGB").save(
                jpeg_buf, format="JPEG", quality=params["jpeg_quality"]
            )
            jpeg_bytes = jpeg_buf.getvalue()

            # Create an A4 page in the output document and paste the JPEG in
            # full-bleed. No text layer is added, so the resulting PDF behaves
            # like a scan.
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=jpeg_bytes)

        out.save(out_pdf)
    finally:
        out.close()
        doc.close()


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


def write_ground_truth(spec: DocSpec, out_path: Path) -> None:
    """Dump the schema-matching ground truth as JSON."""
    out_path.write_text(
        json.dumps(spec.data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_manifest(specs: list[DocSpec], out_path: Path) -> None:
    """One row per document: id, template, degrade level, trap flags."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "template",
                "degrade_level",
                "consignee_to_order",
                "notify_same_as_consignee",
                "date_format",
                "weight_format",
            ]
        )
        for s in specs:
            w.writerow(
                [
                    s.doc_id,
                    s.template,
                    s.degrade_level,
                    int(s.traps["consignee_to_order"]),
                    int(s.traps["notify_same_as_consignee"]),
                    s.traps["date_format"],
                    s.traps["weight_format"],
                ]
            )


def main() -> None:
    _test_iso6346()

    for d in (GT_DIR, CLEAN_DIR, SCAN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    faker = Faker()
    Faker.seed(SEED)

    specs: list[DocSpec] = []
    for i in range(1, N_DOCS + 1):
        spec = build_doc_spec(i, rng, faker)
        specs.append(spec)

        write_ground_truth(spec, GT_DIR / f"{spec.doc_id}.json")
        clean_pdf = CLEAN_DIR / f"{spec.doc_id}.pdf"
        render_clean_pdf(spec, clean_pdf)
        rasterize_and_degrade(
            clean_pdf,
            SCAN_DIR / f"{spec.doc_id}.pdf",
            spec.degrade_level,
            seed=SEED + i,
        )

    write_manifest(specs, MANIFEST_PATH)
    print(f"Generated {N_DOCS} documents under {ROOT}")


if __name__ == "__main__":
    main()
