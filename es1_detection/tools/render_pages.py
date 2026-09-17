"""Renderizza le pagine del PDF in fogli 2x2 per il controllo visivo."""
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

PDF = r"C:\Users\matte\IdeaProjects\Deep_Learning\docs\Studio_Esercizio1_Object_Detection.pdf"
OUT = Path(__file__).parent / "pages"
OUT.mkdir(exist_ok=True)
pdf = pdfium.PdfDocument(PDF)
imgs = [pdf[i].render(scale=1.1).to_pil() for i in range(len(pdf))]
w, h = imgs[0].size
for k in range(0, len(imgs), 4):
    sheet = Image.new("RGB", (w * 2, h * 2), "white")
    for j, im in enumerate(imgs[k:k + 4]):
        sheet.paste(im, ((j % 2) * w, (j // 2) * h))
    sheet.save(OUT / f"sheet_{k // 4 + 1}.png")
print("pagine:", len(imgs), "fogli:", (len(imgs) + 3) // 4)
