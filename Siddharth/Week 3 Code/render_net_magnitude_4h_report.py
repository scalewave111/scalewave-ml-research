"""Render every page for visual QA; no model execution."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/report_render_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
PDF=ROOT/'Report/BTC_4h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
OUT=ROOT/'tmp/pdfs/net_magnitude_4h_report'
OUT.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(PDF));thumbs=[]
for i in range(len(doc)):
    im=doc[i].render(scale=1.5).to_pil().convert('RGB')
    im.save(OUT/f'page_{i+1:02}.png');im.thumbnail((842,595));thumbs.append(im)
for start in range(0,len(thumbs),4):
    sheet=Image.new('RGB',(1720,1240),'#dce3e8');draw=ImageDraw.Draw(sheet)
    for k,im in enumerate(thumbs[start:start+4]):
        x=10+(k%2)*860;y=25+(k//2)*615
        sheet.paste(im,(x,y));draw.text((x,y-17),f'Page {start+k+1}',fill='black')
    sheet.save(OUT/f'contact_{start//4+1:02}.png')
print(json.dumps(dict(pages=len(doc),directory=str(OUT))))
