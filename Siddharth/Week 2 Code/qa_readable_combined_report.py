from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tmp/pdf120_render_deps'))
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw
p=R/'Report/BTC_All_Horizons_MAE_Models_and_Features_READABLE_2026-11-14.pdf'
reader=PdfReader(p);d=pdfium.PdfDocument(str(p));out=R/'tmp/pdfs/combined_readable'
matrixpages=len(PdfReader(out/'matrix.pdf').pages)
indices=sorted(set([0,1,matrixpages-1,matrixpages,matrixpages+1,matrixpages+3,matrixpages+5,len(d)-2]+list(range(0,len(d),15))))
for i in indices:
 im=d[i].render(scale=1.25).to_pil();im.save(out/f'qa_{i+1:03d}.png')
print(json.dumps({'pages':len(d),'matrix_pages':matrixpages,'review_pages':[i+1 for i in indices]}))
