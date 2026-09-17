"""Relazione di studio dell'esercizio 1 (object detection YOLO su Object365)."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).parent
ES1 = Path(r"C:\Users\matte\IdeaProjects\Deep_Learning\es1_detection")
OUT_PDF = Path(r"C:\Users\matte\IdeaProjects\Deep_Learning\docs\Studio_Esercizio1_Object_Detection.pdf")
PHOTO = ES1 / "outputs" / "final_s16_jitter_r34_cosine_e30" / "foto" / "foto_1.jpg"
PHOTO2 = ES1 / "outputs" / "final_s16_jitter_r34_cosine_e30" / "foto" / "foto_ufficio.jpg"
facts = json.loads((HERE / "facts.json").read_text())
extra = json.loads((HERE / "extra.json").read_text())

F = r"C:\Windows\Fonts"
for name, file in (("Arial", "arial"), ("Arial-Bold", "arialbd"), ("Arial-Italic", "ariali"),
                   ("Arial-BoldItalic", "arialbi"), ("Consolas", "consola")):
    pdfmetrics.registerFont(TTFont(name, f"{F}\\{file}.ttf"))
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

BLUE = colors.HexColor("#1f4e79")
LIGHT_BLUE = colors.HexColor("#e8f0f8")
LIGHT_ORANGE = colors.HexColor("#fdf1e0")
GREY = colors.HexColor("#f3f3f3")
body = ParagraphStyle("body", fontName="Arial", fontSize=10.3, leading=14.6, alignment=TA_JUSTIFY, spaceAfter=6)
caption = ParagraphStyle("caption", parent=body, fontSize=8.6, leading=11, alignment=TA_CENTER,
                         textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=10)
h1 = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=17, leading=21, textColor=BLUE, spaceBefore=4, spaceAfter=10)
h2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3, alignment=0)
code = ParagraphStyle("code", fontName="Consolas", fontSize=8.4, leading=11, backColor=GREY,
                      borderPadding=(5, 6, 5, 6), leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
cell = ParagraphStyle("cell", fontName="Arial", fontSize=9, leading=11.5)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Arial-Bold", textColor=colors.white)


class Doc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                         topMargin=2 * cm, bottomMargin=2 * cm,
                         title="Esercizio 1 - Object Detection con YOLO: guida di studio", author="Matteo Barbieri")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self._decorate)])

    def _decorate(self, canv, doc):
        if doc.page == 1:
            return
        canv.saveState()
        canv.setFont("Arial", 8)
        canv.setFillColor(colors.HexColor("#777777"))
        canv.drawString(2.1 * cm, 1.2 * cm, "Esercizio 1 — Object Detection con YOLO · guida di studio")
        canv.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def H1(text, need_cm=6):
    story.append(CondPageBreak(need_cm * cm))
    story.append(Paragraph(text, h1))


def H2(text, need_cm=3.5):
    story.append(CondPageBreak(need_cm * cm))
    story.append(Paragraph(text, h2))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 4))


def img(path, width_cm, cap=None):
    w, h = ImageReader(str(path)).getSize()
    flow = [Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)]
    if cap:
        flow.append(Paragraph(cap, caption))
    story.append(KeepTogether(flow))


def formula(name, width_cm):
    w, h = ImageReader(str(HERE / f"{name}.png")).getSize()
    story.append(Spacer(1, 3))
    story.append(Image(str(HERE / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w))
    story.append(Spacer(1, 6))


def box(title, text, bg=LIGHT_BLUE):
    content = [Paragraph(f"<b>{title}</b>", ParagraphStyle("bt", parent=body, spaceAfter=3, textColor=BLUE))]
    if isinstance(text, list):
        content += [Paragraph(t, bullet, bulletText="•") for t in text]
    else:
        content.append(Paragraph(text, ParagraphStyle("bb", parent=body, spaceAfter=0)))
    t = Table([[content]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(Spacer(1, 3))
    story.append(t)
    story.append(Spacer(1, 9))


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), cell_b if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fc")))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([t]) if len(rows) <= 12 else t)
    story.append(Spacer(1, 10))


def it(n):
    """Intero con separatore delle migliaia all'italiana."""
    return f"{n:,}".replace(",", ".")


def dec(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


mdl = facts["model"]
m8, m16, m32 = mdl["8"], mdl["16"], mdl["32"]
pc, cc = facts["per_class"], facts["crowd_class"]
n_img = sum(facts["splits"].values())
person_share = pc["Person"] / facts["n_objects"]
cov = facts["anchor_coverage"]

# ---------------------------------------------------------------- esperimenti (letti da outputs/)
EXPERIMENTS = [
    ("e0_baseline", "baseline", "nessuna: configurazione di partenza", "riferimento per tutti i confronti"),
    ("e1_anchor8", "baseline", "8 anchor: 4 scale × 2 rapporti", "anchor più aderenti alle forme reali (box mal coperte dal 33% al 13%)"),
    ("e2_stride16", "baseline", "stride 16 (griglia 26×26)", "campo recettivo doppio: oggetti grandi visti per intero"),
    ("e3_stride32", "baseline", "stride 32 (griglia 13×13)", "campo recettivo massimo, ma celle da 32 px"),
    ("e4_jitter_s16", "stride 16", "zoom e traslazione casuali (±30%)", "più varietà di scala e posizione: meno overfitting"),
    ("e5_cosine_s16j", "e4", "learning rate con decadimento coseno", "passi più piccoli verso la fine: convergenza più stabile"),
    ("e6_resnet34_s16j", "e4", "backbone ResNet34", "feature più profonde a parità di stride"),
    ("e7_freeze3_s16j", "e4", "backbone congelato per 3 epoche", "la testa casuale non rovina subito le feature pre-addestrate"),
    ("final_s16_jitter_r34_cosine_e30", "e4", "ResNet34 + coseno + 30 epoche", "combinazione delle modifiche che hanno funzionato"),
]


def experiment_status(name):
    hist = ES1 / "outputs" / name / "history.json"
    if not hist.exists():
        return None
    h = json.loads(hist.read_text())
    done = [r for r in h if r.get("map50") is not None]
    if not done:
        return None
    best = max(done, key=lambda r: r["map50"])
    return {"epochs": len(h), "best": best}


statuses = {name: experiment_status(name) for name, *_ in EXPERIMENTS}
completed = {k: v for k, v in statuses.items() if v and v["epochs"] >= 20}
risultati = json.loads((HERE / "risultati.json").read_text(encoding="utf-8"))
errori = json.loads((HERE / "errori_finale.json").read_text(encoding="utf-8"))
TEST = {"map50": 0.4442, "precision": 0.6598, "recall": 0.5307, "f1": 0.5883, "conf": 0.9, "n_gt": 8383,
        "per_class": {   # da outputs/<finale>/test_report.txt
            "Person": (0.7451, 0.7033, 0.7236, 0.7197, 4115),
            "Chair": (0.6177, 0.3883, 0.4768, 0.4316, 1419),
            "Table": (0.3580, 0.1834, 0.2425, 0.1903, 687),
            "Cabinet": (0.4655, 0.2746, 0.3455, 0.2772, 761),
            "Car": (0.6391, 0.5236, 0.5756, 0.5238, 487),
            "Lamp": (0.4943, 0.4627, 0.4780, 0.4328, 469),
            "Picture": (0.6091, 0.4653, 0.5276, 0.5249, 288),
            "Monitor": (0.5833, 0.4013, 0.4755, 0.4532, 157)},
        "recall_medi": 0.5056, "recall_grandi": 0.5776, "n_medi": 5459, "n_grandi": 2924}

# ================================================================== copertina
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("Esercizio 1", ParagraphStyle("c0", parent=h1, fontSize=15, alignment=TA_CENTER,
                                                      textColor=colors.HexColor("#555555"))))
story.append(Paragraph("Object Detection con YOLO", ParagraphStyle("c1", parent=h1, fontSize=28, leading=34, alignment=TA_CENTER)))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Guida di studio: teoria, dati, modello, valutazione e domande d'esame",
                       ParagraphStyle("c2", parent=body, fontSize=13, alignment=TA_CENTER)))
story.append(Spacer(1, 2.5 * cm))
cover = Table([[Paragraph(
    "<b>In una riga:</b> un detector in stile <b>YOLO</b> scritto a mano (griglia, anchor, loss, IoU, NMS) "
    "sopra una <b>ResNet18 pre-addestrata</b>, addestrato su un sottoinsieme di <b>Object365</b> "
    f"({it(n_img)} immagini, 8 classi di oggetti medio-grandi) per riconoscere oggetti in una foto.",
    ParagraphStyle("cv", parent=body, fontSize=11, leading=16, alignment=TA_CENTER))]], colWidths=[14 * cm])
cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE), ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                           ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
story.append(cover)
story.append(Spacer(1, 4 * cm))
story.append(Paragraph("Matteo Barbieri — Apprendimento Automatico e Profondo", ParagraphStyle("c3", parent=body, alignment=TA_CENTER)))
story.append(PageBreak())
story.append(Paragraph("Indice", ParagraphStyle("toc_title", parent=h1)))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("toc0", fontName="Arial", fontSize=10.5, leading=17)]
story.append(toc)
story.append(PageBreak())

# ================================================================== 1
H1("1. Il quadro in una pagina")
P("Questa guida spiega l'esercizio 1 dall'inizio alla fine: la consegna, le scelte sui dati, la teoria della "
  "detection con YOLO, come è costruita e addestrata la rete e come si misurano i risultati. I numeri sui dati e "
  "sul modello sono calcolati sul progetto reale.")
table([
    ["Aspetto", "Scelta nel progetto"],
    ["Compito", "Object detection: trovare gli oggetti in una foto e disegnare una bounding box con la classe"],
    ["Dataset", f"Object365, sottoinsieme di {it(n_img)} immagini e {it(facts['n_objects'])} box"],
    ["Classi", "Person, Chair, Table, Cabinet, Car, Lamp, Picture, Monitor (oggetti medio-grandi)"],
    ["Tecnica", "YOLO (griglia + anchor), implementato a mano: target, loss, decodifica, IoU, NMS"],
    ["Rete", f"ResNet18 pre-addestrata su ImageNet (troncata) + collo + testa 1×1; {dec(m8['total'] / 1e6, 2)} M parametri a stride 8"],
    ["Valutazione", "mAP@0.5 sul validation per confrontare le varianti; test set usato una sola volta alla fine"],
    ["Ottimizzazione", "8 esperimenti controllati: ognuno cambia una sola cosa rispetto alla baseline"],
    ["Risultato", f"<b>mAP@0.5 = {dec(TEST['map50'], 3)} sul test</b> (baseline 0,221), F1 {dec(TEST['f1'], 3)}, "
                  f"precision {dec(TEST['precision'], 2)}, recall {dec(TEST['recall'], 2)}"],
], [3.2, 13.4])
box("Le 6 idee da portare all'esame", [
    "Il risultato dipende più dalle <b>scelte di progetto</b> che dal modello: stride e augmentation, misurati con "
    "esperimenti controllati, hanno raddoppiato il mAP.",
    "<b>YOLO</b> divide l'immagine in una griglia: ogni oggetto è di competenza della cella che contiene il suo "
    "centro, e la rete predice tutte le box in un solo passaggio (<i>You Only Look Once</i>).",
    "Gli <b>anchor</b> sono forme di partenza: ogni cella ha più anchor, e un oggetto viene assegnato a quello con la "
    "forma più simile (IoU calcolata solo su larghezza e altezza).",
    "La rete non predice coordinate assolute ma <b>correzioni</b>: σ(t) sposta il centro dentro la cella, e<super>t</super> "
    "allarga o stringe l'anchor.",
    "La <b>loss</b> somma quattro termini (coordinate, objectness, no-object, classe); il peso del no-object evita "
    "sia la rete che non vede mai nulla sia quella che vede oggetti ovunque.",
    "L'<b>IoU</b> misura quanto due box si sovrappongono; serve per la <b>NMS</b> (eliminare i duplicati) e per "
    "decidere se una predizione è corretta.",
    "La <b>precision</b> dipende dalla soglia di confidenza: per confrontare modelli si usa l'<b>AP</b>, l'area sotto "
    "la curva precision-recall, mediata sulle classi (<b>mAP</b>).",
])
if len(completed) < len(EXPERIMENTS):
    box("Stato dei risultati", (
        f"Gli esperimenti di ottimizzazione sono in corso ({len(completed)} completati su {len(EXPERIMENTS)}). "
        "I capitoli su dati, teoria, modello e valutazione sono definitivi; i capitoli 12 e 13 si completano con i "
        "numeri finali quando la campagna termina."), LIGHT_ORANGE)

# ================================================================== 2
H1("2. La consegna e il dataset Object365")
H2("2.1 Cosa chiede l'esercizio")
P("La traccia chiede di <i>applicare una delle tecniche viste a lezione per l'object detection tramite bounding "
  "box</i>, per costruire un riconoscitore di oggetti in un'immagine presa con una fotocamera, usando "
  "<b>Object365</b> ed eseguendo un <b>subsampling</b> se necessario. Due parole chiave guidano le scelte: "
  "“tecnica vista a lezione” (quindi YOLO con anchor, IoU e NMS, implementati e non importati da una libreria) e "
  "“subsampling” (il dataset completo è ingestibile).")
H2("2.2 Object365")
P("Object365 contiene circa 600.000 immagini ad alta qualità con 365 categorie e circa 10 milioni di box. Il "
  "progetto usa la copia su HuggingFace <font face='Consolas'>surenreddy/object365</font>, divisa in file parquet "
  "da circa 120 MB (35 file per ogni blocco): ogni riga contiene l'immagine JPEG e le sue annotazioni. Ogni "
  "annotazione ha la box in pixel (angolo in alto a sinistra, larghezza, altezza), l'id della categoria e il flag "
  "<b>iscrowd</b>. Se ne scaricano <b>15 file</b> (circa 1,9 GB).")

# ================================================================== 3
H1("3. Le scelte sui dati")
H2("3.1 Otto classi scelte sui dati")
P("Le 8 classi non sono scelte a intuito. Si sono misurate, per tutte le categorie presenti nei file scaricati, "
  "frequenza e dimensione tipica delle box, tenendo quelle <b>frequenti e di dimensione medio-grande</b>. Una prima "
  "versione del progetto usava Bottle, Cup, Hat e Book: le prime tre hanno lato minore mediano intorno al 3% del "
  "lato dell'immagine (troppo piccole per il detector), e Book era per due terzi fatto di box crowd (intere "
  "librerie).")
table([
    ["Classe", "Categorie Object365 (id)", "Box", "Box crowd"],
    ["Person", "Person (1)", it(pc["Person"]), it(cc["Person"])],
    ["Chair", "Chair (3), Stool (48)", it(pc["Chair"]), it(cc["Chair"])],
    ["Table", "Desk (10), Side Table (169)", it(pc["Table"]), it(cc["Table"])],
    ["Cabinet", "Cabinet/shelf (13)", it(pc["Cabinet"]), it(cc["Cabinet"])],
    ["Car", "Car (6), SUV (35), Van (50), Pickup Truck (88)", it(pc["Car"]), it(cc["Car"])],
    ["Lamp", "Lamp (7)", it(pc["Lamp"]), it(cc["Lamp"])],
    ["Picture", "Picture/Frame (17)", it(pc["Picture"]), it(cc["Picture"])],
    ["Monitor", "Monitor/TV (38)", it(pc["Monitor"]), it(cc["Monitor"])],
], [2.6, 8.4, 2.8, 2.8])
bullets([
    "<b>Categorie sorelle unite</b>: auto, SUV, furgoni e pick-up sono quasi indistinguibili. Tenendo solo “Car”, un SUV "
    "nella foto sarebbe etichettato come sfondo mentre un'auto identica accanto è un oggetto: la rete riceverebbe "
    "segnali contraddittori. Lo stesso vale per scrivanie e tavolini, sedie e sgabelli.",
    "<b>Id verificati a occhio</b>: gli id numerici sono stati controllati ritagliando esempi dalle immagini. Dining "
    "Table (99) e Coffee Table (168) sono stati esclusi perché gli esempi erano assenti o sbagliati. I ritagli "
    "mostrano anche un certo <b>rumore nelle etichette</b> (circa un ritaglio su dieci palesemente errato).",
    f"<b>Sbilanciamento</b>: Person da sola è il {dec(100 * person_share, 0)}% delle box; Monitor meno del 2%. "
    "Ci si aspettano prestazioni più basse sulle classi rare.",
])
img(HERE / "classi.png", 15.5, "Box per classe: oggetti del compito, oggetti piccoli e box crowd (tutte ignorate in valutazione tranne le prime).")

H2("3.2 Box crowd e oggetti piccoli: le regioni ignorate")
P(f"In Object365 una box <b>crowd</b> racchiude un gruppo di oggetti che non sono stati annotati uno per uno (una "
  f"folla, una fila di sedie). Nei dati sono {it(facts['n_crowd'])} su {it(facts['n_objects'] + facts['n_crowd'])}. Chiedere alla rete di "
  "predire una sola box attorno a un gruppo è un obiettivo sbagliato; cancellarle sarebbe ugualmente sbagliato, "
  "perché le persone restano nella foto e la rete imparerebbe che “lì non c'è nulla”. La soluzione, la stessa di "
  "COCO, è trattarle come <b>regioni ignorate</b>: non generano obiettivi e non producono penalità.")
img(HERE / "esempio_ignore.png", 13.5,
    "Un'immagine del validation set: in verde gli oggetti da trovare, in arancione una box crowd (un gruppo di "
    "persone dietro la ringhiera), in giallo due oggetti troppo piccoli. Le ultime due categorie sono ignorate.")
P("Lo stesso trattamento vale per gli <b>oggetti piccoli</b>. Un detector a scala singola non può localizzare un "
  "oggetto più piccolo di una cella della griglia. Quanti oggetti finiscono sotto questo limite dipende dalla griglia:")
table([["Griglia", "Lato di una cella", "Oggetti più piccoli di una cella"]] +
      [[f"{g}×{g}", f"1/{g} dell'immagine", f"{dec(v)}%"] for g, v in facts["ignored_by_grid"].items()],
      [4, 6, 6.6])
box("La definizione del compito", (
    f"Il compito valutato è: <b>trovare gli oggetti con lato minore almeno 1/20 del lato dell'immagine</b> "
    f"({it(facts['task_objects'])} oggetti). La soglia è fissa e non dipende dalla griglia del modello: così tutte le "
    "varianti vengono confrontate sugli stessi oggetti. Se ogni variante usasse la propria soglia, una griglia più "
    "grossolana toglierebbe gli oggetti difficili e sembrerebbe migliore senza esserlo."))

H2("3.3 Formato delle etichette e split")
P("Ogni immagine ha un file di testo con una riga per box: <font face='Consolas'>classe cx cy w h crowd</font>. "
  "Le coordinate sono <b>normalizzate</b> in [0, 1] rispetto alla dimensione dell'immagine (centro, larghezza, "
  "altezza), quindi restano valide dopo il ridimensionamento. Le immagini sono divise a caso (seme fisso) in:")
table([["Split", "Immagini", "Uso"],
       ["train", it(facts["splits"]["train"]), "aggiornare i pesi"],
       ["validation", it(facts["splits"]["val"]), "scegliere il checkpoint e confrontare le varianti"],
       ["test", it(facts["splits"]["test"]), "valutazione finale, una sola volta"]], [4, 3.5, 9.1])

# ================================================================== 4
H1("4. Richiami sulle CNN")
H2("4.1 Convoluzione, stride, pooling")
P("Una <b>convoluzione</b> fa scorrere un piccolo filtro (per esempio 3×3) sull'immagine e a ogni posizione calcola "
  "un prodotto scalare: il risultato è una <b>feature map</b>. I pesi del filtro sono appresi. Due proprietà la "
  "rendono adatta alle immagini: <b>parameter sharing</b> (lo stesso rilevatore di bordi è utile ovunque) e "
  "<b>connessioni sparse</b> (ogni uscita dipende solo da una piccola zona dell'ingresso).")
bullets([
    "<b>Stride</b>: di quanti pixel si sposta il filtro. Con stride 2 la feature map ha metà risoluzione.",
    "<b>Max pooling</b>: tiene il massimo di ogni regione; riduce la risoluzione e rende le feature più robuste a "
    "piccoli spostamenti.",
    "<b>Stride totale</b> della rete: il prodotto degli stride di tutti gli strati. Con stride totale 8, un'immagine "
    "416×416 diventa una feature map 52×52: ogni posizione corrisponde a un blocco di 8×8 pixel.",
])
H2("4.2 Campo recettivo")
P("Il <b>campo recettivo</b> di un'unità è la porzione di immagine che può influenzarla. Cresce a ogni strato: "
  "un filtro k×k aggiunge (k − 1) × j pixel, dove j è lo stride accumulato fino a quel punto. È decisivo per la "
  "detection: per riconoscere un tavolo che occupa metà foto, la cella deve “vedere” una parte consistente del tavolo.")
H2("4.3 Batch normalization, ResNet e transfer learning")
bullets([
    "<b>Batch normalization</b>: normalizza le attivazioni di ogni strato usando media e varianza del mini-batch "
    "(più scala e traslazione apprese). Stabilizza e accelera il training e ha un leggero effetto di regolarizzazione.",
    "<b>ResNet</b>: introduce le <b>connessioni residue</b> (l'uscita di un blocco è x + F(x)). Il gradiente passa "
    "direttamente attraverso la somma, rendendo addestrabili reti profonde. ResNet18 ha 4 stadi (layer1–layer4) che "
    "portano lo stride da 4 a 32.",
    "<b>Transfer learning</b>: si parte da una rete pre-addestrata su ImageNet (1,2 milioni di immagini, 1000 classi), "
    "i cui filtri riconoscono già bordi, texture e parti di oggetti, e la si adatta al nuovo compito (fine-tuning). "
    "Con pochi dati il vantaggio è enorme rispetto ai pesi casuali.",
])

# ================================================================== 5
H1("5. Dalla localizzazione a YOLO")
H2("5.1 Classificazione con localizzazione")
P("Nelle slide il punto di partenza è un'immagine con un solo oggetto. La rete produce un vettore "
  "y = [p<sub>c</sub>, b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>, c<sub>1</sub>, …, c<sub>n</sub>]: "
  "p<sub>c</sub> dice se c'è un oggetto, le b descrivono la box, le c la classe. Se p<sub>c</sub> = 0 il resto è "
  "“don't care” e non entra nella loss.", ParagraphStyle("body_sub", parent=body, leading=18))
H2("5.2 Sliding window e sua versione convoluzionale")
P("Per più oggetti, l'idea ingenua è la <b>sliding window</b>: far scorrere una finestra sull'immagine, a più "
  "dimensioni, e classificare ogni ritaglio. Il problema è il <b>costo</b>: migliaia di ritagli, ognuno passato "
  "nella rete. La soluzione delle slide è implementarla <b>convoluzionalmente</b>: trasformando gli strati densi "
  "in convoluzioni, una sola passata sull'immagine intera produce le uscite di tutte le finestre insieme, "
  "condividendo i calcoli.")
H2("5.3 L'idea di YOLO")
bullets([
    "L'immagine è divisa in una <b>griglia S×S</b>; la rete produce, per ogni cella, un vettore come quello di 5.1.",
    "Ogni oggetto è assegnato a <b>una sola cella</b>: quella che contiene il suo centro.",
    "Il centro è espresso <b>rispetto alla cella</b> (tra 0 e 1), larghezza e altezza rispetto alla cella "
    "(possono superare 1 se l'oggetto è più grande).",
    "Per gestire più oggetti con il centro nella stessa cella, ogni cella ha più <b>anchor box</b>: forme predefinite. "
    "Ogni oggetto va all'anchor di forma più simile. Le slide notano che numero e forma degli anchor si scelgono "
    "“a mano”.",
    "Tutto in <b>un solo passaggio</b> della rete: da qui “You Only Look Once” e la velocità del metodo.",
])

# ================================================================== 6
H1("6. YOLO nel progetto", need_cm=12)
H2("6.1 Cella responsabile e anchor")
ge = extra["grid_example"]
img(HERE / "griglia_anchor.png", 10,
    "Immagine 416×416 come la vede la rete, con una griglia 13×13 per leggibilità. In verde la persona, in giallo la "
    "cella che ne contiene il centro, tratteggiati i 5 anchor centrati nella cella; in rosso quello scelto.")
P(f"La persona ha larghezza {dec(ge['gt_wh'][0], 3)} e altezza {dec(ge['gt_wh'][1], 3)} (frazioni dell'immagine). "
  "Confrontando solo le forme, cioè immaginando box e anchor centrati nello stesso punto, le IoU con i 5 anchor "
  "valgono " + ", ".join(dec(v, 2) for v in ge["anchor_ious"]) +
  f": vince l'anchor {ge['best_anchor'] + 1}, che diventa <b>responsabile</b> dell'oggetto. Solo quell'anchor, in "
  "quella cella, riceve gli obiettivi di coordinate, objectness e classe.")
H2("6.2 Scegliere gli anchor")
P(f"Nella configurazione di partenza gli anchor sono 5 forme scelte a mano. Sul dataset attuale il "
  f"{dec(cov['default'][0], 0)}% delle box non ha nessun anchor con IoU di forma almeno 0,5: tutti gli anchor sono "
  "verticali, mentre il 12% degli oggetti è orizzontale (tavoli, auto) e un quarto è molto alto (persone in piedi). "
  f"Una costruzione sistematica, sempre “a mano” ma con due regole (4 scale che raddoppiano per 2 rapporti "
  f"d'aspetto), scende al {dec(cov['8'][0], 0)}%. È uno degli esperimenti.")
img(HERE / "anchor.png", 15.5, "Ogni punto è una box del dataset (larghezza, altezza); le croci sono gli anchor.")
H2("6.3 Cosa predice la rete")
P(f"Per ogni cella e per ogni anchor la rete produce 5 + C numeri, con C = 8 classi: t<sub>x</sub>, t<sub>y</sub>, "
  f"t<sub>w</sub>, t<sub>h</sub> (la box), t<sub>o</sub> (objectness: c'è un oggetto?) e 8 logit di classe. Con "
  f"griglia 52×52 e 5 anchor sono {it(m8['predictions'])} box candidate per immagine. L'uscita ha forma "
  f"(5, 52, 52, 13).")
H2("6.4 Decodifica delle coordinate")
formula("f_decode", 13)
img(HERE / "decodifica.png", 12.5, "Come i numeri grezzi diventano una box.")
bullets([
    "<b>σ (sigmoide) sul centro</b>: σ(t) è sempre tra 0 e 1, quindi il centro predetto non esce mai dalla cella "
    "responsabile. Senza, nei primi passi di training le box salterebbero ovunque.",
    "<b>Esponenziale sulle dimensioni</b>: e<super>t</super> è sempre positivo (niente box di larghezza negativa) e "
    "lavora in scala moltiplicativa: t = 0 lascia l'anchor com'è, t = log 2 lo raddoppia.",
])
_tw, _th = ImageReader(str(HERE / "f_targets.png")).getSize()
story.append(KeepTogether([
    Paragraph("In training gli obiettivi si ottengono invertendo le formule: l'offset del centro dentro la cella e il "
              "logaritmo del rapporto tra dimensione vera e dimensione dell'anchor.", body),
    Image(str(HERE / "f_targets.png"), width=6.5 * cm, height=6.5 * cm * _th / _tw),
    Spacer(1, 6)]))

# ================================================================== 7
H1("7. L'architettura", need_cm=17)
img(HERE / "architettura.png", 13, "La rete del progetto con la configurazione di partenza (stride 8).")
bullets([
    "<b>Backbone</b>: ResNet18 pre-addestrata, <b>troncata dopo layer2</b> per avere stride 8 e quindi una griglia "
    "fine 52×52. Si scartano layer3 e layer4.",
    "<b>Collo</b>: due blocchi conv 3×3 + batch norm + ReLU a risoluzione costante, per recuperare parte della "
    "profondità persa con il troncamento.",
    "<b>Testa</b>: una convoluzione 1×1 che produce i 65 numeri per posizione. È l'equivalente convoluzionale di uno "
    "strato denso applicato a ogni cella (la sliding window convoluzionale delle slide).",
    "La rete è <b>completamente convoluzionale</b>: funziona con qualunque dimensione multipla di 32.",
    "In alternativa si può usare un backbone scritto interamente da zero (<font face='Consolas'>PRETRAINED_BACKBONE = False</font>); "
    "la parte YOLO resta identica.",
])
box("Nota", (
    "Questo capitolo descrive la <b>configurazione di partenza</b> (stride 8, ResNet18), quella da cui sono partiti "
    "gli esperimenti. La configurazione finale, scelta misurando, usa stride 16 e ResNet34: si veda il capitolo 13."))
H2("7.1 Il compromesso dello stride")
table([
    ["Stride", "Griglia", "Box candidate", "Parametri", "Campo recettivo", "Oggetti sotto una cella"],
    *[[str(s), f"{mdl[str(s)]['grid']}×{mdl[str(s)]['grid']}", it(mdl[str(s)]["predictions"]),
       f"{dec(mdl[str(s)]['total'] / 1e6, 2)} M", f"~{mdl[str(s)]['rf']} px",
       f"{dec(facts['ignored_by_grid'][str(416 // s)])}%"] for s in (8, 16, 32)],
], [2, 2.4, 3, 2.8, 3.2, 3.2])
img(HERE / "campo_recettivo.png", 16,
    "Campo recettivo teorico (azzurro) della cella gialla, per i tre stride, rispetto alla persona in verde.")
P("Lo stride regola un compromesso. Stride piccolo: griglia fine, adatta agli oggetti piccoli, ma pochi strati e "
  f"campo recettivo ridotto (~{m8['rf']} px su 416): un oggetto grande non “entra” nella zona vista da una cella. "
  f"Stride grande: la ResNet completa vede quasi tutta l'immagine (~{m32['rf']} px) con feature più astratte, ma "
  "una cella di 32 pixel non basta a separare gli oggetti medi. In una valutazione preliminare con stride 8 il "
  "recall sugli oggetti grandi è risultato <b>più basso</b> di quello sui medi, e le classi più grandi (Table, "
  "Cabinet) hanno avuto l'AP più bassa: è il sintomo di un campo recettivo insufficiente, che gli esperimenti "
  "e2 ed e3 verificano.")

# ================================================================== 8
H1("8. La loss")
H2("8.1 Costruzione dei target")
bullets([
    "Per ogni oggetto normale: cella del centro, anchor di forma più simile, obiettivi di coordinate e classe su "
    "quella sola coppia cella-anchor, objectness = 1.",
    "Per ogni <b>oggetto sotto soglia</b> (lato minore sotto una cella): nessun obiettivo, e la cella del centro è "
    "<b>ignorata</b> per tutti gli anchor.",
    "Per ogni <b>box crowd</b>: nessun obiettivo, e sono ignorate <b>tutte le celle</b> il cui centro cade dentro la box.",
    "Tutte le altre coppie cella-anchor sono “vuote”: objectness = 0.",
])
H2("8.2 I quattro termini", need_cm=6)
table([
    ["Termine", "Funzione", "Dove si applica", "Peso"],
    ["Coordinate (x, y)", "MSE tra σ(t<sub>x</sub>), σ(t<sub>y</sub>) e l'offset vero", "anchor responsabili", "λ<sub>coord</sub> = 5"],
    ["Coordinate (w, h)", "MSE tra t<sub>w</sub>, t<sub>h</sub> e log(g/a)", "anchor responsabili", "λ<sub>coord</sub> = 5"],
    ["Objectness", "BCE con obiettivo 1", "anchor responsabili", "1"],
    ["No-object", "BCE con obiettivo 0", "anchor vuoti e non ignorati", "λ<sub>noobj</sub> = 10"],
    ["Classe", "cross-entropy sulle 8 classi", "anchor responsabili", "1"],
], [3.3, 5.8, 4.5, 3])
formula("f_loss", 13)
bullets([
    "<b>Perché λ<sub>coord</sub> = 5</b>: le coordinate sono la parte difficile del compito e i loro errori numerici "
    "sono piccoli; il peso le mette sullo stesso piano degli altri termini.",
    f"<b>Lo sbilanciamento</b>: su {it(m8['predictions'])} coppie cella-anchor, un'immagine con 8 oggetti ne ha 8 "
    "responsabili e più di 13.000 vuote. Se il termine no-object fosse una somma, dominerebbe: la rete imparerebbe "
    "che la risposta migliore è “non c'è mai nulla”.",
    "<b>Media sulle celle vuote</b>: il progetto normalizza il no-object come media sulle celle vuote, non per numero "
    "di oggetti. Così il suo peso non cambia quando cambia la griglia (52×52 ha 16 volte le celle di 13×13).",
    "<b>Perché λ<sub>noobj</sub> = 10</b>: nel YOLO originale (98 predizioni per immagine, circa 4 oggetti, somma con "
    "peso 0,5) il peso complessivo del “non c'è nulla” è circa 12 volte quello degli oggetti. Con la media, un λ "
    "intorno a 10 riproduce quel rapporto. Con λ = 0,5 la penalità sui falsi positivi sarebbe circa 24 volte più "
    "debole: la rete vedrebbe oggetti ovunque e la precision crollerebbe.",
])
box("Un limite noto", (
    "Se due oggetti hanno il centro nella stessa cella e scelgono lo stesso anchor, l'ultimo sovrascrive il primo: "
    "uno dei due non ha obiettivo. Con una griglia 52×52 e più anchor è raro, ma con griglie grossolane cresce."
), LIGHT_ORANGE)

# ================================================================== 9
H1("9. Inferenza: IoU e non-max suppression")
H2("9.1 Dalle uscite alle box")
formula("f_score", 7.5)
bullets([
    "Si decodificano tutte le box candidate e si calcola lo <b>score</b>: probabilità che ci sia un oggetto per "
    "probabilità della classe più probabile.",
    "Si scartano le box con score sotto la <b>soglia di confidenza</b>; per sicurezza se ne tengono al massimo 300 "
    "per immagine.",
    "Si applica la <b>NMS per classe</b>: due box di classi diverse possono sovrapporsi legittimamente (una persona "
    "davanti a un'auto).",
])
H2("9.2 Intersection over Union")
formula("f_iou", 9.5)
ie = extra["iou_example"]
story.append(KeepTogether([
    Table([[Image(str(HERE / "iou.png"), width=6.4 * cm, height=6.4 * cm * ImageReader(str(HERE / "iou.png")).getSize()[1] / ImageReader(str(HERE / "iou.png")).getSize()[0]),
            [Paragraph("<b>Esempio</b>", ParagraphStyle("x", parent=body, textColor=BLUE)),
             Paragraph(f"Area verde {dec(ie['areaA'], 2)}, area rossa {dec(ie['areaB'], 2)}, intersezione "
                       f"{dec(ie['inter'], 2)}.", body),
             Paragraph(f"IoU = {dec(ie['inter'], 2)} / ({dec(ie['areaA'], 2)} + {dec(ie['areaB'], 2)} − "
                       f"{dec(ie['inter'], 2)}) = <b>{dec(ie['iou'], 3)}</b>.", body),
             Paragraph("Vale 1 per box identiche, 0 per box disgiunte. È simmetrica e non dipende dalla scala.", body)]]],
          colWidths=[7 * cm, 9.6 * cm], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))]))
H2("9.3 Non-max suppression")
P("Più celle e anchor vicini tendono a predire lo stesso oggetto. L'algoritmo delle slide:")
story.append(Preformatted(
    "ordina le box per score decrescente\n"
    "finché restano box:\n"
    "    prendi quella con score massimo  -> è una predizione finale\n"
    "    scarta tutte le altre con IoU > soglia (0.45) rispetto a lei", code))
img(HERE / "nms.png", 14.5, "Cinque box sopra la soglia sullo stesso oggetto: tutte hanno IoU tra "
    + " e ".join(dec(v, 2) for v in (min(extra["nms_ious_with_best"]), max(extra["nms_ious_with_best"])))
    + " con la migliore, quindi l'NMS tiene solo quella con score 0,91.")

# ================================================================== 10
H1("10. Come si misura un detector")
H2("10.1 Predizioni corrette e sbagliate")
bullets([
    "Una predizione è un <b>vero positivo</b> (TP) se ha IoU ≥ 0,5 con un oggetto vero della stessa classe non "
    "ancora abbinato. Le predizioni si considerano in ordine di score: le più sicure scelgono per prime.",
    "Altrimenti è un <b>falso positivo</b> (FP): box sbagliata, classe sbagliata o duplicato di un oggetto già trovato.",
    "Gli oggetti mai abbinati sono <b>falsi negativi</b>.",
    "<b>Regioni ignorate</b> (regole di COCO): non contano una predizione abbinata a un oggetto sotto soglia, una "
    "predizione contenuta per almeno metà in una box crowd della sua classe, e una predizione non abbinata a sua "
    "volta sotto soglia.",
])
formula("f_metrics", 11)
H2("10.2 La trappola della soglia fissa")
P("Precision e recall cambiano con la soglia di confidenza: alzandola si scartano box, la precision sale e il recall "
  "scende. Confrontare due modelli a una soglia fissa è ingannevole, perché modelli diversi sono calibrati in modo "
  "diverso. In una valutazione preliminare lo stesso modello aveva <b>F1 = 0,145</b> alla soglia 0,25 e "
  "<b>F1 = 0,249</b> alla soglia 0,65: confrontato a 0,25 sarebbe sembrato molto peggiore di quanto era.")
H2("10.3 Average Precision e mAP")
P("Si raccolgono tutte le predizioni (da confidenza 0,01 in su), si ordinano per score e si scende lungo la lista: "
  "a ogni passo si ricalcolano precision e recall. Si ottiene una curva precision-recall; la precision viene "
  "<b>interpolata</b> (a ogni recall si prende la precision massima ottenibile a recall uguale o maggiore) e l'area "
  "sotto la curva è l'<b>AP</b> della classe. La <b>mAP</b> è la media sulle classi.")
formula("f_ap", 11.5)
pe = extra["pr_example"]
img(HERE / "pr_curve.png", 10,
    f"Esempio: 12 predizioni ordinate per score su {pe['n_gt']} oggetti (sequenza TP/FP: "
    + " ".join("TP" if t else "FP" for t in pe["ranked"]) + f"). AP = {dec(pe['ap'], 3)}; il recall si ferma a "
    f"{dec(pe['final_recall'], 2)} perché 2 oggetti non sono mai trovati.")
bullets([
    "La mAP@0.5 <b>non dipende dalla soglia</b>: è la metrica con cui si scelgono il checkpoint e la variante migliore.",
    "Per usare il detector serve comunque una soglia: si sceglie quella che massimizza l'F1 sul validation, e viene "
    "salvata nel checkpoint e usata da <font face='Consolas'>detect.py</font>.",
    "Il report divide il recall per fascia di dimensione (medi: lato minore tra 1/20 e 3/20; grandi: oltre) per "
    "capire dove il modello fatica.",
])

# ================================================================== 11
H1("11. Il training")
table([
    ["Aspetto", "Scelta", "Motivo"],
    ["Pre-elaborazione", "ridimensionamento a 416×416, normalizzazione ImageNet", "stesse statistiche con cui ResNet18 è stata pre-addestrata"],
    ["Augmentation", "flip orizzontale (p = 0,5), luminosità e contrasto, zoom e traslazione ±30%", "più varietà senza nuove etichette; il flip ribalta anche le box (c<sub>x</sub> → 1 − c<sub>x</sub>)"],
    ["Ottimizzatore", "Adam, learning rate 10<super>-4</super> con decadimento coseno, weight decay 5·10<super>-4</super>", "learning rate basso per non rovinare il backbone pre-addestrato"],
    ["Batch / epoche", "16 / 30", "circa 720 passi per epoca"],
    ["Selezione", "mAP@0.5 sul validation a ogni epoca", "la loss può scendere senza che le detection migliorino"],
    ["Riproducibilità", "configurazione salvata nel checkpoint e verificata al caricamento", "un checkpoint caricato con parametri diversi darebbe risultati sbagliati in silenzio"],
], [3.2, 6.6, 6.8])
P("L'augmentation geometrica opzionale (esperimento e4) ingrandisce o rimpicciolisce l'immagine fino al 30% e la "
  "sposta, trasformando le box di conseguenza. Una box che dopo il taglio resta visibile per meno del 40% diventa "
  "una regione ignorata, per lo stesso motivo delle box crowd.")
P("Per gli esperimenti ogni variante si lancia senza modificare il codice, passando i parametri in una variabile "
  "d'ambiente. Il codice viene copiato nella cartella dell'esperimento, che salva checkpoint e storico delle "
  "metriche per epoca: ogni risultato resta riproducibile.")

# ================================================================== 12
H1("12. Gli esperimenti di ottimizzazione")
P("Invece di modificare molte cose insieme, ogni esperimento cambia <b>una sola</b> scelta rispetto alla baseline "
  "(<i>ablation</i>). Così l'effetto di ogni scelta è misurato separatamente e si può motivare. Alla fine si "
  "combinano le modifiche che migliorano la mAP@0.5 sul validation.")
rows = [["Esperimento", "Base", "Cosa cambia", "mAP@0.5 val."]]
for name, base, change, hyp in EXPERIMENTS:
    stt = statuses[name]
    res = f"<b>{dec(stt['best']['map50'], 3)}</b>" if stt else "—"
    rows.append([name.replace("final_s16_jitter_r34_cosine_e30", "finale"), base, change, res])
table(rows, [4.2, 2.2, 7.4, 2.8])
img(HERE / "confronto_esperimenti.png", 16,
    "mAP@0.5 sul validation. Le barre blu scuro sono le modifiche adottate; e5, e6 ed e7 partono tutte da e4, "
    "non si sommano fra loro.")
H2("12.1 Cosa ha funzionato e cosa no")
bullets([
    "<b>Stride 16: +0,121</b> (0,221 → 0,342), il salto più grande. Conferma la diagnosi del campo recettivo: con "
    "stride 8 le classi grandi (Table 0,03, Cabinet 0,08) erano quasi invisibili, con stride 16 salgono a 0,13 e 0,20. "
    "Stride 32 peggiora (0,294): il campo recettivo è ancora più ampio, ma celle da 32 px non bastano a separare gli "
    "oggetti medi, che sono la maggioranza.",
    "<b>Augmentation: +0,043</b> (0,342 → 0,385) ed elimina l'overfitting: senza, la loss sul validation risaliva da "
    "2,88 a 3,75 nelle ultime epoche; con, resta al minimo fino alla fine.",
    "<b>ResNet34: +0,052</b> (0,385 → 0,437), il secondo contributo. Un backbone più profondo migliora tutte le "
    "classi, in particolare quelle difficili.",
    "<b>Learning rate coseno: +0,008</b>, dentro il rumore come valore, ma la curva finale è più stabile: da "
    "0,376/0,385/0,376 a una salita regolare. Adottato per questo motivo, non per il numero.",
    "<b>8 anchor: −0,004</b>. Gli anchor coprono meglio le forme (box mal coperte dal 33% al 13%), ma il risultato "
    "non cambia: <b>non erano il collo di bottiglia</b>. Scartato.",
    "<b>Backbone congelato per 3 epoche: +0,011</b> sul massimo, ma l'ultima epoca è più bassa della variante senza "
    "congelamento (0,379 contro 0,383): il massimo è un picco isolato. Scartato.",
])
box("Attenzione al rumore", (
    "Ogni variante è addestrata una volta sola: differenze di mAP molto piccole (per esempio sotto 0,005) possono "
    "dipendere dal caso (ordine dei dati, inizializzazione della testa) più che dalla modifica. Si considerano "
    "significativi solo miglioramenti netti."), LIGHT_ORANGE)

# ================================================================== 13
H1("13. Risultati")
H2("13.1 La configurazione finale")
fb = risultati["final_best"]
table([
    ["Scelta", "Valore", "Perché"],
    ["Stride", "16 (griglia 26×26)", "il salto più grande: +0,121 di mAP"],
    ["Augmentation", "zoom e traslazione ±30%", "+0,043 e nessun overfitting"],
    ["Backbone", "ResNet34 pre-addestrata", "+0,052"],
    ["Learning rate", "10<super>-4</super> con decadimento coseno", "curva finale più stabile"],
    ["Epoche", "30", "a 20 epoche il mAP saliva ancora"],
    ["Anchor", "i 5 originali", "gli 8 anchor non hanno migliorato nulla"],
], [3.4, 5.4, 7.8])
img(HERE / "curve_finale.png", 16,
    f"Run finale. A sinistra le loss: quella di validation smette di scendere ma non risale, quindi nessun "
    f"overfitting. A destra le metriche sul validation; il checkpoint scelto è l'epoca {fb['epoch']}.")
P(f"Il training è durato circa {int(risultati['seconds_per_epoch'] * 30 / 60)} minuti "
  f"({risultati['seconds_per_epoch']} secondi per epoca su una RTX 4070 Ti). Il checkpoint migliore è quello "
  f"dell'epoca {fb['epoch']}, con <b>mAP@0.5 = {dec(fb['map50'], 3)}</b> sul validation e F1 massimo "
  f"{dec(fb['f1_max'], 3)} alla soglia di confidenza {dec(fb['best_conf'], 2)}.")

H2("13.2 Il risultato sul test set")
P(f"Il modello scelto è stato valutato <b>una sola volta</b> sulle {it(facts['splits']['test'])} immagini del test "
  f"({it(TEST['n_gt'])} oggetti del compito), con la soglia di confidenza scelta sul validation.")
table([
    ["Metrica", "Validation", "Test"],
    ["mAP@0.5", f"<b>{dec(fb['map50'], 3)}</b>", f"<b>{dec(TEST['map50'], 3)}</b>"],
    ["F1 (a confidenza > 0,9)", dec(fb["f1_max"], 3), dec(TEST["f1"], 3)],
    ["Precision", "—", dec(TEST["precision"], 3)],
    ["Recall", "—", dec(TEST["recall"], 3)],
], [5.4, 5.6, 5.6])
P("Il calo tra validation e test è piccolo e atteso: sul validation sono stati scelti checkpoint, soglia e "
  "configurazione, quindi quel numero è leggermente ottimistico. <b>Il risultato da riportare è quello sul test.</b>")
rows = [["Classe", "Precision", "Recall", "F1", "AP@0.5", "Oggetti"]]
for c in facts["per_class"]:
    p, r, f1c, ap, n = TEST["per_class"][c]
    rows.append([c, dec(p, 3), dec(r, 3), dec(f1c, 3), f"<b>{dec(ap, 3)}</b>", it(n)])
rows.append(["<b>totale</b>", f"<b>{dec(TEST['precision'], 3)}</b>", f"<b>{dec(TEST['recall'], 3)}</b>",
             f"<b>{dec(TEST['f1'], 3)}</b>", f"<b>{dec(TEST['map50'], 3)}</b>", f"<b>{it(TEST['n_gt'])}</b>"])
table(rows, [3, 2.6, 2.4, 2.2, 2.6, 2.6])
img(HERE / "ap_per_classe.png", 16, "AP per classe: baseline, modello finale sul validation e modello finale sul test.")
bullets([
    f"<b>Person</b> è la classe migliore (AP {dec(TEST['per_class']['Person'][3], 2)}): è anche la più frequente, "
    "con quasi la metà degli oggetti.",
    f"<b>Table</b> resta la più difficile ({dec(TEST['per_class']['Table'][3], 2)}): i tavoli sono spesso coperti da "
    "oggetti, tagliati dal bordo e con confini ambigui.",
    f"<b>Il campo recettivo non è più un problema</b>: sul test il recall sugli oggetti grandi "
    f"({dec(TEST['recall_grandi'], 2)}) supera quello sui medi ({dec(TEST['recall_medi'], 2)}), mentre nella "
    "baseline era il contrario (0,38 contro 0,56).",
])

H2("13.3 Dove sbaglia il modello", need_cm=9)
P("Analizzando le predizioni sul validation alla soglia d'uso si vede che il problema principale non è "
  "riconoscere l'oggetto, ma delimitarlo.")
img(HERE / "errori.png", 16, "Scomposizione degli errori del modello finale (validation, soglia 0,9).")
fp_tot = sum(errori["fp"].values())
fn_tot = sum(errori["fn"].values())
bullets([
    f"<b>Box imprecise</b>: {dec(100 * errori['fp']['box imprecisa (IoU 0,1-0,5)'] / fp_tot, 0)}% dei falsi positivi "
    f"e {dec(100 * errori['fn']['trovato ma box imprecisa'] / fn_tot, 0)}% degli oggetti mancati. Il modello trova "
    "l'oggetto ma la box non arriva a IoU 0,5, quindi lo stesso errore conta due volte.",
    f"<b>Confidenza poco separata</b>: il {dec(100 * errori['fn']['visto ma con confidenza sotto soglia'] / fn_tot, 0)}% "
    "degli oggetti mancati era stato predetto correttamente, ma sotto la soglia. Alzare la soglia perde oggetti, "
    "abbassarla aggiunge falsi positivi.",
    f"<b>Classificazione corretta</b>: solo il {dec(100 * (errori['fp']['classe sbagliata']) / fp_tot, 1)}% dei falsi "
    "positivi è un errore di classe. Le confusioni sono quasi tutte fra Chair e Table.",
    "<b>Etichette mancanti</b>: controllando a occhio le predizioni “sullo sfondo” più sicure, molte sono oggetti "
    "reali non annotati (un'auto, un mobile, alcune sedie). Una parte di questi falsi positivi non è un errore del "
    "modello, e il valore reale è quindi un po' più alto di quello misurato.",
])

H2("13.4 Il riconoscitore su una foto")
P("La consegna chiede un riconoscitore per una foto scattata con una fotocamera. Questa è una foto di prova, mai "
  "vista dal modello, elaborata con <font face='Consolas'>detect.py</font> alla soglia salvata nel checkpoint.")
img(PHOTO, 13, "Detection sulla foto: entrambe le persone trovate con confidenza 0,96 e 0,98, nessun falso positivo. "
    "Le due figure parziali sul bordo sinistro, piccole e coperte dai fiori, restano sotto la soglia.")
H2("13.5 Un secondo esempio: una scena d'ufficio", need_cm=10)
P("La seconda immagine (una scena d'ufficio con più classi insieme) mostra bene sia i pregi sia i limiti. Il "
  "modello trova <b>tre sedie su quattro</b>, con box precise, e un tavolo sullo sfondo; tutte le predizioni sono "
  "corrette, quindi la precision è perfetta.")
img(PHOTO2, 11, "Quattro oggetti rilevati, tutti corretti. Restano fuori le due scrivanie in primo piano, i monitor "
    "e le lampade.")
bullets([
    "<b>Le scrivanie in primo piano non vengono trovate</b>, pur essendo gli oggetti più grandi: è la debolezza "
    f"misurata sul test (Table ha AP {dec(TEST['per_class']['Table'][3], 2)}, la più bassa). Qui sono anche tagliate "
    "dal bordo, a forma di L e coperte da oggetti.",
    "<b>Monitor e lampade restano sotto soglia</b>: uno schermo nero spento su sfondo chiaro e una lampada snodata "
    "sono lontani dagli esempi tipici del dataset.",
    "<b>Precision alta, recall basso</b>: è il comportamento atteso alla soglia 0,9, scelta per privilegiare la "
    "correttezza delle box rispetto al numero di oggetti trovati. Abbassando la soglia si trovano più oggetti, ma "
    "arrivano anche i falsi positivi.",
])
box("Il risultato in una frase", (
    f"Su foto mai viste il detector trova circa la metà degli oggetti richiesti ({dec(TEST['recall'], 2)} di recall) "
    f"e due box su tre sono corrette ({dec(TEST['precision'], 2)} di precision). Funziona bene sulle persone, "
    "discretamente su auto, quadri e monitor, male su tavoli e mobili."))

# ================================================================== 14
H1("14. Come è organizzato il codice")
table([
    ["File", "Cosa fa"],
    ["<font face='Consolas'>config.py</font>", "Classi, geometria, anchor, soglie, iperparametri, override per gli esperimenti."],
    ["<font face='Consolas'>prepare_data.py</font>", "Scarica gli shard, filtra le classi, marca le box crowd, divide in train/val/test."],
    ["<font face='Consolas'>analyze_sizes.py</font>", "Statistiche sulle dimensioni: griglia, soglie, anchor."],
    ["<font face='Consolas'>dataset.py</font>", "Dataset PyTorch: lettura di immagini ed etichette, augmentation."],
    ["<font face='Consolas'>model.py</font>", "Backbone (ResNet18/34 o da zero), collo e testa."],
    ["<font face='Consolas'>loss.py</font>", "Costruzione dei target, regioni ignorate, loss YOLO, decodifica."],
    ["<font face='Consolas'>utils.py</font>", "IoU, conversioni tra formati di box, NMS, tutto scritto a mano."],
    ["<font face='Consolas'>train.py</font>", "Training, valutazione a ogni epoca, checkpoint e storico."],
    ["<font face='Consolas'>evaluate.py</font>", "Abbinamento predizioni-oggetti, precision/recall/F1, AP e mAP."],
    ["<font face='Consolas'>detect.py</font>", "Detection su una foto e disegno delle box."],
], [3.8, 12.8])
story.append(Preformatted(
    "python src/prepare_data.py                       # dati (15 shard)\n"
    "python src/train.py                              # training\n"
    "python src/evaluate.py --split test              # valutazione finale\n"
    "python src/detect.py foto.jpg --out risultato.jpg\n"
    "\n"
    "# una variante, senza modificare il codice (PowerShell)\n"
    "$env:ES1_CONFIG = '{\"EXPERIMENT\": \"e2_stride16\", \"STRIDE\": 16}'\n"
    "python src/train.py", code))

# ================================================================== 15
H1("15. Limiti e possibili miglioramenti")
table([
    ["Limite", "Possibile intervento"],
    ["Una sola scala di predizione: compromesso tra oggetti piccoli e grandi", "Predizioni su più scale (feature pyramid, come YOLOv3)"],
    ["Ridimensionamento a quadrato: le proporzioni vengono deformate", "Letterbox: ridimensionare mantenendo le proporzioni e riempire i bordi"],
    ["Un oggetto per coppia cella-anchor", "Assegnare un oggetto a più anchor, o griglie più fini"],
    ["Etichette rumorose (circa un ritaglio su dieci sbagliato)", "Pulizia delle etichette, o loss più robuste"],
    ["Classi sbilanciate (Person oltre il 50%)", "Campionamento bilanciato o pesi per classe nella loss"],
    ["Poche immagini rispetto a Object365 completo", "Più shard: è la leva più forte, a costo di training più lunghi"],
    ["Valutazione solo a IoU 0,5", "mAP media su IoU da 0,5 a 0,95 (metrica COCO), più severa sulla precisione delle box"],
], [7.8, 8.8])

# ================================================================== 16
H1("16. Domande d'esame probabili")
qa = [
    ("Perché YOLO e non una sliding window?",
     "La sliding window classifica migliaia di ritagli separatamente. YOLO predice tutte le box in un solo passaggio "
     "convoluzionale sull'immagine intera, condividendo i calcoli: è molto più veloce."),
    ("A quale cella viene assegnato un oggetto?",
     "A quella che contiene il suo centro. Solo quella cella, con l'anchor di forma più simile, ha l'obiettivo di predirlo."),
    ("A cosa servono gli anchor?",
     "Permettono a una cella di predire più oggetti e fanno specializzare le predizioni su forme tipiche (alte, "
     "larghe, piccole, grandi). La rete predice solo una correzione dell'anchor."),
    ("Come si sceglie l'anchor responsabile?",
     "Con l'IoU tra le sole forme (larghezza e altezza, box centrate nello stesso punto): vince l'anchor con IoU massima."),
    ("Perché la sigmoide sul centro e l'esponenziale sulle dimensioni?",
     "La sigmoide tiene il centro dentro la cella responsabile e stabilizza il training; l'esponenziale garantisce "
     "dimensioni positive e scala l'anchor in modo moltiplicativo."),
    ("Cosa succede se il peso del no-object è troppo basso? E troppo alto?",
     "Troppo basso: la rete predice oggetti ovunque e la precision crolla. Troppo alto: impara che non c'è mai "
     "nulla e il recall crolla. Il progetto usa la media sulle celle vuote con λ = 10."),
    ("Calcola l'IoU di due box.",
     f"IoU = intersezione / (area A + area B − intersezione). Nell'esempio: {dec(ie['inter'], 2)} / "
     f"({dec(ie['areaA'], 2)} + {dec(ie['areaB'], 2)} − {dec(ie['inter'], 2)}) = {dec(ie['iou'], 3)}."),
    ("Come funziona la NMS e perché per classe?",
     "Si tiene la box con score massimo e si scartano quelle che la sovrappongono con IoU sopra soglia, poi si "
     "ripete. Per classe, perché oggetti di classi diverse possono sovrapporsi davvero."),
    ("Cos'è il campo recettivo e perché conta qui?",
     "La porzione di immagine che influenza un'unità. Con stride 8 e rete troncata è ~131 px su 416: gli oggetti "
     "grandi non sono visti per intero da una cella, e infatti sono riconosciuti peggio."),
    ("Perché una ResNet pre-addestrata?",
     "Transfer learning: i filtri appresi su ImageNet riconoscono già bordi, texture e parti di oggetti. Con circa "
     "11.000 immagini di training partire da pesi casuali darebbe risultati molto peggiori."),
    ("Cosa sono le box crowd e come le tratti?",
     "Box attorno a gruppi di oggetti non annotati singolarmente. Sono regioni ignorate: niente obiettivi, niente "
     "penalità no-object sulle celle coperte, e in valutazione le predizioni al loro interno non contano."),
    ("Perché non cancellare semplicemente gli oggetti piccoli?",
     "Resterebbero nell'immagine: la loss insegnerebbe che in quel punto non c'è nulla, penalizzando anche oggetti "
     "simili più grandi. Ignorarli evita il segnale contraddittorio."),
    ("Perché la mAP e non l'F1?",
     "L'F1 dipende dalla soglia di confidenza e modelli diversi sono calibrati diversamente. La mAP integra tutte "
     "le soglie. L'F1 massimo serve poi per scegliere la soglia d'uso."),
    ("Come si calcola l'AP?",
     "Si ordinano le predizioni per score, si calcolano precision e recall cumulati, si interpola la precision "
     "(massimo a recall maggiore o uguale) e si somma l'area: Σ (R<sub>k</sub> − R<sub>k−1</sub>) · P<sub>k</sub>."),
    ("Perché validation e test separati?",
     "Il validation si usa per scegliere checkpoint e varianti, quindi il suo punteggio è ottimistico. Il test non "
     "influenza nessuna scelta e dà una stima onesta."),
    ("Perché un esperimento alla volta?",
     "Per attribuire ogni miglioramento alla sua causa. Cambiando più cose insieme non si sa quale ha aiutato e "
     "quale ha peggiorato."),
    ("Perché le categorie simili sono unite?",
     "Un SUV etichettato come sfondo accanto a un'auto etichettata come oggetto dà segnali contraddittori. Unirle "
     "rende coerente la supervisione e aumenta gli esempi."),
    ("Cosa miglioreresti?",
     "Predizioni multi-scala, letterbox al posto della deformazione, più dati, bilanciamento delle classi, "
     "valutazione anche a IoU più severe."),
]
for i, (q, a) in enumerate(qa, 1):
    story.append(KeepTogether([
        Paragraph(f"<b>{i}. {q}</b>", ParagraphStyle("q", parent=body, spaceAfter=2, textColor=BLUE)),
        Paragraph(a, ParagraphStyle("a", parent=body, leftIndent=12, spaceAfter=8)),
    ]))

# ================================================================== 17
H1("17. Glossario")
glossary = [
    ("Anchor box", "Forma di box predefinita associata a ogni cella; la rete ne predice una correzione."),
    ("AP / mAP", "Area sotto la curva precision-recall di una classe / media sulle classi."),
    ("Backbone", "Parte della rete che estrae le feature (qui ResNet18)."),
    ("Box crowd", "Box attorno a un gruppo di oggetti non annotati singolarmente; regione ignorata."),
    ("Campo recettivo", "Porzione di immagine che influenza un'unità della rete."),
    ("Cella responsabile", "Cella che contiene il centro di un oggetto e ha il compito di predirlo."),
    ("Collo (neck)", "Strati tra backbone e testa."),
    ("IoU", "Intersection over Union: area comune diviso area unione di due box."),
    ("Logit", "Valore grezzo prima di sigmoide o softmax."),
    ("NMS", "Non-max suppression: elimina le box duplicate sullo stesso oggetto."),
    ("Objectness", "Probabilità predetta che una coppia cella-anchor contenga un oggetto."),
    ("Regione ignorata", "Zona che non produce né obiettivi né penalità."),
    ("Stride", "Fattore di riduzione della risoluzione tra immagine e feature map."),
    ("Testa (head)", "Strato finale che produce le predizioni per ogni cella e anchor."),
    ("Transfer learning", "Riutilizzo di una rete pre-addestrata su un altro compito."),
]
table([["Termine", "Significato"]] + [[f"<b>{t}</b>", d] for t, d in glossary], [4.2, 12.4])

doc = Doc(str(OUT_PDF))
doc.multiBuild(story)
print("scritto", OUT_PDF, "| esperimenti completati:", len(completed))
