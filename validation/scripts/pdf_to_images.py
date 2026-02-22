"""
pdf_to_images.py
将 math/ 目录下的所有 PDF 转换为高分辨率 PNG 图片（300 DPI）。
使用 PyMuPDF (fitz) — 纯 Python，Windows 无需额外安装 Poppler。
输出：validation/output/images/<pdf_name>/page_001.png ...
"""

import sys
from pathlib import Path
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERROR] 缺少依赖: PyMuPDF")
    print("请运行: pip install PyMuPDF")
    sys.exit(1)


def convert_pdfs(input_dir: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """
    将 input_dir 下所有 PDF 转换为 PNG 图片。
    返回所有生成的图片路径列表。
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] 在 {input_dir} 下未找到 PDF 文件")
        return []

    all_image_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyMuPDF 使用 matrix 缩放来控制 DPI（默认 72 DPI）
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    print(f"[INFO] 共找到 {len(pdf_files)} 个 PDF，开始转换（DPI={dpi}）...\n")

    for pdf_path in tqdm(pdf_files, desc="PDF → PNG"):
        safe_name = pdf_path.stem.replace(" ", "_")
        img_dir = output_dir / "images" / safe_name
        img_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"\n[ERROR] 打开失败 {pdf_path.name}: {e}")
            continue

        for i, page in enumerate(doc, start=1):
            img_path = img_dir / f"page_{i:03d}.png"
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(img_path))
            all_image_paths.append(img_path)

        page_count = len(doc)
        doc.close()
        print(f"  ✅ {pdf_path.name} → {page_count} 页 → {img_dir.name}/")

    print(f"\n[INFO] 转换完成，共生成 {len(all_image_paths)} 张图片")
    return all_image_paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / "math"
    output_dir = project_root / "validation" / "output"

    convert_pdfs(input_dir, output_dir)
