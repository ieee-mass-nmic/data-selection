from pathlib import Path
from pypdf import PdfReader


for path in (Path("paper/AAAI27.pdf"), Path("paper/supplementary.pdf")):
    reader = PdfReader(path)
    print(f"## {path}")
    print(f"metadata={dict(reader.metadata or {})}")
    print(f"attachments={sorted((reader.attachments or {}).keys())}")
    print(f"pages={len(reader.pages)}")
    for number, page in enumerate(reader.pages, start=1):
        annotations = page.get("/Annots") or []
        print(
            f"page={number} mediabox={tuple(page.mediabox)} "
            f"rotation={page.get('/Rotate', 0)} annotations={len(annotations)}"
        )
