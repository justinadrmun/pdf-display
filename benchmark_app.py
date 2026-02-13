"""
Benchmark & Demo App — Compare all PDF display strategies side by side.
Run: just benchmark
"""

import base64
import csv
import io
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="PDF Display Benchmark", layout="wide")
st.title("PDF Display Strategy Benchmark")


# ============================================================================
# Benchmark logging
# ============================================================================
@dataclass
class BenchmarkResult:
    strategy: str
    start_page: int
    end_page: int
    num_pages: int
    dpi: int
    backend_time_s: float
    payload_raw_bytes: int = 0
    payload_base64_bytes: int = 0
    extracted_bytes: int = 0
    savings_pct: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _add_result(result: BenchmarkResult):
    """Append a benchmark result to session state."""
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = []
    st.session_state.benchmark_results.append(result)


def _get_results() -> list[BenchmarkResult]:
    return st.session_state.get("benchmark_results", [])


# ============================================================================
# Upload & page range controls
# ============================================================================
uploaded = st.file_uploader("Upload a PDF to test", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to compare rendering strategies.")
    st.stop()

pdf_bytes = uploaded.getvalue()

# Detect total pages
try:
    import fitz
    _doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(_doc)
    _doc.close()
except Exception:
    total_pages = 999

col_a, col_b, col_c = st.columns(3)
with col_a:
    start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1, step=1)
with col_b:
    end_page = st.number_input("End page", min_value=1, max_value=total_pages, value=min(10, total_pages), step=1)
with col_c:
    dpi = st.select_slider("DPI (image strategies)", options=[72, 100, 150, 200, 300], value=150)

if end_page < start_page:
    st.error("End page must be ≥ start page")
    st.stop()

num_pages = end_page - start_page + 1
page_range = list(range(start_page, end_page + 1))

st.caption(f"**{uploaded.name}** — {total_pages} pages total, benchmarking pages {start_page}–{end_page} ({num_pages} pages)")

# Clear previous results when inputs change
run_key = f"{uploaded.name}_{start_page}_{end_page}_{dpi}"
if st.session_state.get("_last_run_key") != run_key:
    st.session_state.benchmark_results = []
    st.session_state._last_run_key = run_key

st.divider()

# ============================================================================
# Strategy 1: PyMuPDF → Images (st.image)
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Strategy 1: Pages → Images")
    st.caption("PyMuPDF renders pages server-side → st.image()")

    try:
        from docs.pdf_display_strategies_docs import display_pdf_page_range_as_images

        t0 = time.perf_counter()
        display_pdf_page_range_as_images(pdf_bytes, start_page=start_page, end_page=end_page, dpi=dpi)
        total = time.perf_counter() - t0
        st.success(f"Total backend time: {total:.3f}s for {num_pages} pages")

        _add_result(BenchmarkResult(
            strategy="1_pymupdf_image",
            start_page=start_page,
            end_page=end_page,
            num_pages=num_pages,
            dpi=dpi,
            backend_time_s=round(total, 4),
            payload_raw_bytes=len(pdf_bytes),
        ))
    except ImportError as e:
        st.error(f"Missing dependency: {e}\n\nRun: pip install pymupdf")

# ============================================================================
# Strategy 2: PyMuPDF extraction → st.pdf()
# ============================================================================
with col2:
    st.subheader("Strategy 2: Extract → st.pdf()")
    st.caption("PyMuPDF page extraction → native st.pdf()")

    try:
        from docs.pdf_display_strategies_docs import display_pdf_pages_native

        t0 = time.perf_counter()
        display_pdf_pages_native(pdf_bytes, start_page=start_page, end_page=end_page, height=500)
        total = time.perf_counter() - t0
        st.success(f"Total backend time: {total:.3f}s for {num_pages} pages")

        _add_result(BenchmarkResult(
            strategy="2_pymupdf_st_pdf",
            start_page=start_page,
            end_page=end_page,
            num_pages=num_pages,
            dpi=dpi,
            backend_time_s=round(total, 4),
            payload_raw_bytes=len(pdf_bytes),
        ))
    except ImportError as e:
        st.error(f"Missing dependency: {e}\n\nRun: pip install pymupdf streamlit-pdf")
    except Exception as e:
        st.warning(f"st.pdf() error: {e}")

st.divider()

# ============================================================================
# Strategy 3: Baseline — streamlit_pdf_viewer (full PDF, filter on frontend)
# ============================================================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("Baseline: streamlit_pdf_viewer")
    st.caption("Full PDF sent to frontend, pages_to_render filters on client")

    try:
        from streamlit_pdf_viewer import pdf_viewer

        t0 = time.perf_counter()
        pdf_viewer(pdf_bytes, pages_to_render=page_range, height=500, key="baseline")
        total = time.perf_counter() - t0

        payload_b64 = len(base64.b64encode(pdf_bytes))
        st.success(f"Backend: {total:.3f}s (frontend rendering NOT measured)")
        st.warning(f"⚠️ Full PDF sent as base64: {payload_b64:,} bytes. In K8s, expect 5-7s additional frontend lag.")

        _add_result(BenchmarkResult(
            strategy="3_baseline_pdf_viewer",
            start_page=start_page,
            end_page=end_page,
            num_pages=num_pages,
            dpi=dpi,
            backend_time_s=round(total, 4),
            payload_raw_bytes=len(pdf_bytes),
            payload_base64_bytes=payload_b64,
        ))
    except ImportError:
        st.error("streamlit_pdf_viewer not installed.\n\nRun: pip install streamlit-pdf-viewer")

# ============================================================================
# Strategy 4: Optimized — extract pages THEN streamlit_pdf_viewer
# ============================================================================
with col4:
    st.subheader("Optimized: Extract + streamlit_pdf_viewer")
    st.caption("PyMuPDF extracts pages BEFORE sending to frontend")

    try:
        from docs.pdf_display_strategies_docs import _extract_pages_pymupdf
        from streamlit_pdf_viewer import pdf_viewer

        t0 = time.perf_counter()

        extracted_bytes = _extract_pages_pymupdf(pdf_bytes, page_range)
        extract_time = time.perf_counter() - t0

        original_size = len(pdf_bytes)
        extracted_size = len(extracted_bytes)
        savings = (1 - extracted_size / original_size) * 100

        pdf_viewer(
            extracted_bytes,
            pages_to_render=list(range(1, num_pages + 1)),
            height=500,
            key="optimized",
        )
        total = time.perf_counter() - t0

        payload_b64 = len(base64.b64encode(extracted_bytes))
        st.success(
            f"Backend: {total:.3f}s (extract: {extract_time:.3f}s) • "
            f"Payload: {original_size:,} → {extracted_size:,} bytes ({savings:.0f}% smaller)"
        )
        st.info("💡 Smaller payload = faster WebSocket transfer in K8s = faster frontend render")

        _add_result(BenchmarkResult(
            strategy="4_optimized_pdf_viewer",
            start_page=start_page,
            end_page=end_page,
            num_pages=num_pages,
            dpi=dpi,
            backend_time_s=round(total, 4),
            payload_raw_bytes=original_size,
            payload_base64_bytes=payload_b64,
            extracted_bytes=extracted_size,
            savings_pct=round(savings, 1),
        ))
    except ImportError as e:
        st.error(f"Missing dependency: {e}")

st.divider()

# ============================================================================
# Strategy 5: Hybrid (image preview + expandable full viewer)
# ============================================================================
st.subheader("Strategy 5: Hybrid — Instant Preview + Expandable")
st.caption("Image loads instantly for each page, click to open interactive viewer")

try:
    from docs.pdf_display_strategies_docs import display_pdf_hybrid

    t0 = time.perf_counter()
    for pg in page_range:
        expand_key = f"hybrid_{pg}"
        expanded = st.session_state.get(f"{expand_key}_expanded", False)
        display_pdf_hybrid(
            pdf_bytes,
            page_number=pg,
            dpi=dpi,
            key=expand_key,
            show_full_viewer=expanded,
        )
    total = time.perf_counter() - t0

    _add_result(BenchmarkResult(
        strategy="5_hybrid",
        start_page=start_page,
        end_page=end_page,
        num_pages=num_pages,
        dpi=dpi,
        backend_time_s=round(total, 4),
        payload_raw_bytes=len(pdf_bytes),
    ))
except ImportError as e:
    st.error(f"Missing dependency: {e}")

st.divider()

# ============================================================================
# Payload Size Analysis
# ============================================================================
st.subheader("📊 Payload Size Analysis")
st.caption(f"Comparing payload sizes for pages {start_page}–{end_page}")

try:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    sizes = {
        "Full PDF (raw)": len(pdf_bytes),
        "Full PDF (base64)": len(base64.b64encode(pdf_bytes)),
    }

    # Extract the page range
    dst = fitz.open()
    for pg in page_range:
        idx = min(pg - 1, len(doc) - 1)
        dst.insert_pdf(doc, from_page=idx, to_page=idx)
    range_bytes = dst.tobytes()
    sizes[f"Pages {start_page}–{end_page} PDF (raw)"] = len(range_bytes)
    sizes[f"Pages {start_page}–{end_page} PDF (base64)"] = len(base64.b64encode(range_bytes))
    dst.close()

    # Image sizes for the range
    total_png = 0
    total_jpg = 0
    zoom = dpi / 72
    for pg in page_range:
        idx = min(pg - 1, len(doc) - 1)
        page = doc[idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        total_png += len(pix.tobytes("png"))
        total_jpg += len(pix.tobytes("jpeg"))
    sizes[f"Pages as PNG ({dpi} DPI, total)"] = total_png
    sizes[f"Pages as JPEG ({dpi} DPI, total)"] = total_jpg

    doc.close()

    for label, size in sizes.items():
        c1, c2 = st.columns([3, 1])
        c1.text(label)
        c2.text(f"{size:>12,} bytes ({size / 1024:.0f} KB)")

except ImportError:
    st.warning("Install pymupdf for size analysis: pip install pymupdf")

st.divider()

# ============================================================================
# Benchmark Results Log
# ============================================================================
st.subheader("📋 Benchmark Results Log")

results = _get_results()

if not results:
    st.info("Run a benchmark to see results here.")
else:
    # Display as a table
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in results])
    display_cols = [
        "strategy", "num_pages", "start_page", "end_page", "dpi",
        "backend_time_s", "payload_raw_bytes", "payload_base64_bytes",
        "extracted_bytes", "savings_pct",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True, hide_index=True)

    # Export buttons
    col_csv, col_json, col_clear = st.columns(3)

    with col_csv:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
        st.download_button(
            "⬇️ Download CSV",
            csv_buf.getvalue(),
            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with col_json:
        json_str = json.dumps([asdict(r) for r in results], indent=2)
        st.download_button(
            "⬇️ Download JSON",
            json_str,
            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    with col_clear:
        if st.button("🗑️ Clear results"):
            st.session_state.benchmark_results = []
            st.rerun()

    # Summary comparison
    st.subheader("⚡ Comparison Summary")
    summary = {}
    for r in results:
        if r.strategy not in summary or r.backend_time_s < summary[r.strategy].backend_time_s:
            summary[r.strategy] = r

    if summary:
        fastest = min(summary.values(), key=lambda r: r.backend_time_s)
        for name, r in sorted(summary.items(), key=lambda kv: kv[1].backend_time_s):
            ratio = r.backend_time_s / fastest.backend_time_s if fastest.backend_time_s > 0 else 1
            bar = "█" * min(int(ratio * 10), 50)
            marker = " ← fastest" if r.strategy == fastest.strategy else ""
            st.text(f"{r.strategy:<30} {r.backend_time_s:>8.4f}s  {bar}{marker}")
