import base64
import io
import time
from typing import Optional
import streamlit as st

def display_pdf_as_image(pdf_bytes: bytes, page_number: int = 1, dpi: int = 150, width: Optional[int] = None, caption: Optional[str] = None):
    import fitz
    start = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 1 or page_number > len(doc):
        st.error(f"Page {page_number} not found (PDF has {len(doc)} pages)")
        doc.close()
        return
    page = doc[page_number - 1]
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    img_bytes = pixmap.tobytes("png")
    render_time = time.perf_counter() - start
    doc.close()
    if width:
        st.image(img_bytes, caption=caption, width=width)
    else:
        st.image(img_bytes, caption=caption, use_container_width=True)
    st.caption(f"Page {page_number} • Rendered in {render_time:.3f}s")

def display_pdf_page_range_as_images(pdf_bytes: bytes, start_page: int = 1, end_page: int = 1, dpi: int = 150):
    import fitz
    start = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    end_page = min(end_page, len(doc))
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(start_page, end_page + 1):
        page = doc[page_num - 1]
        pixmap = page.get_pixmap(matrix=matrix)
        img_bytes = pixmap.tobytes("png")
        st.image(img_bytes, caption=f"Page {page_num}", use_container_width=True)
    render_time = time.perf_counter() - start
    doc.close()
    st.caption(f"Pages {start_page}–{end_page} • Rendered in {render_time:.3f}s")

def display_pdf_pages_native(pdf_bytes: bytes, start_page: int = 1, end_page: int = 1, height: int = 500):
    import fitz
    start = time.perf_counter()
    src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    end_page = min(end_page, len(src_doc))
    dst_doc = fitz.open()
    dst_doc.insert_pdf(src_doc, from_page=start_page - 1, to_page=end_page - 1)
    output_bytes = dst_doc.tobytes()
    extract_time = time.perf_counter() - start
    src_doc.close()
    dst_doc.close()
    render_start = time.perf_counter()
    st.pdf(output_bytes, height=height)
    render_time = time.perf_counter() - render_start
    st.caption(f"Pages {start_page}–{end_page} • Extract: {extract_time:.3f}s, Render call: {render_time:.3f}s")

def display_pdf_with_loading_overlay(pdf_bytes: bytes, pages_to_render: list[int] = (), height: Optional[int] = None, width="100%", key: Optional[str] = None, estimated_load_seconds: float = 5.0):
    from streamlit_pdf_viewer import pdf_viewer
    if pages_to_render:
        optimized_bytes = _extract_pages_pymupdf(pdf_bytes, pages_to_render)
        adjusted_pages = list(range(1, len(pages_to_render) + 1))
    else:
        optimized_bytes = pdf_bytes
        adjusted_pages = list(pages_to_render)
    overlay_id = f"pdf-loading-{key or 'default'}"
    st.markdown(f"""
        <style>
            #{overlay_id} {{ position: relative; min-height: {height or 400}px; }}
            #{overlay_id}::before {{ content: 'Loading PDF...'; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); display: flex; align-items: center; justify-content: center; font-size: 16px; color: #666; z-index: 1000; border-radius: 8px; animation: pdfFadeOut 0.5s ease-in forwards; animation-delay: {estimated_load_seconds}s; }}
            @keyframes pdfFadeOut {{ from {{ opacity: 1; }} to {{ opacity: 0; pointer-events: none; }} }}
            #{overlay_id} .pulse-dot {{ display: inline-block; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 80%, 100% {{ opacity: 0; }} 40% {{ opacity: 1; }} }}
        </style>
        <div id='{overlay_id}'></div>
        """, unsafe_allow_html=True)
    pdf_viewer(optimized_bytes, width=width, height=height, key=key, pages_to_render=adjusted_pages)

def _extract_pages_pymupdf(pdf_bytes: bytes, pages: list[int]) -> bytes:
    import fitz
    src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst_doc = fitz.open()
    for page_num in pages:
        if 1 <= page_num <= len(src_doc):
            dst_doc.insert_pdf(src_doc, from_page=page_num - 1, to_page=page_num - 1)
    result = dst_doc.tobytes()
    src_doc.close()
    dst_doc.close()
    return result

def display_pdf_via_sas_url(sas_url: str, page: int = 1, height: int = 600, width: str = "100%"):
    url_with_page = f"{sas_url}#page={page}"
    pdf_display = f"""
    <iframe src='{url_with_page}' width='{width}' height='{height}px' type='application/pdf' style='border: 1px solid #ddd; border-radius: 4px;'>
        <p>Your browser does not support PDF viewing. <a href='{sas_url}' download>Download the PDF</a>.</p>
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def generate_sas_url_for_page(blob_manager, blob_path: str, expiry_hours: int = 1) -> str:
    from datetime import datetime, timedelta, timezone
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
    sas_token = generate_blob_sas(
        account_name=blob_manager.account_name,
        container_name=blob_manager.container_name,
        blob_name=blob_path,
        account_key=blob_manager.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
    )
    return f"https://{blob_manager.account_name}.blob.core.windows.net/{blob_manager.container_name}/{blob_path}?{sas_token}"

def display_pdf_hybrid(pdf_bytes: bytes, page_number: int = 1, dpi: int = 150, key: Optional[str] = None, show_full_viewer: bool = False):
    import fitz
    widget_key = key or f"pdf_hybrid_{page_number}"
    if not show_full_viewer:
        display_pdf_as_image(pdf_bytes, page_number=page_number, dpi=dpi)
        if st.button("📄 Open interactive viewer (enables text selection)", key=f"{widget_key}_expand"):
            st.session_state[f"{widget_key}_expanded"] = True
            st.rerun()
    else:
        display_pdf_pages_native(pdf_bytes, start_page=page_number, end_page=page_number, height=600)
