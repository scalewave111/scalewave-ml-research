from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/short_horizon_pdf_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
out=ROOT/'tmp/pdfs/excursion_48h_report';out.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(ROOT/'Report/output/pdf/BTC_48h_Excursion_Models_Features_Rankings_2026-11-13.pdf'))
thumbs=[]
for i in range(len(doc)):
    im=doc[i].render(scale=1.5).to_pil().convert('RGB');im.save(out/f'page_{i+1:02d}.png')
    im.thumbnail((840,595));thumbs.append(im)
for start in range(0,len(thumbs),4):
    sheet=Image.new('RGB',(1720,1250),'#dce3e8');draw=ImageDraw.Draw(sheet)
    for k,im in enumerate(thumbs[start:start+4]):
        x=10+(k%2)*860;y=28+(k//2)*620;sheet.paste(im,(x,y));draw.text((x,y-18),f'Page{start+k+1}',fill='black')
    sheet.save(out/f'contact_{start//4+1:02d}.png')
print(json.dumps(dict(pages=len(doc),render_dir=str(out))))
