from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/pdf120_render_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
OUT=ROOT/'tmp/pdfs/report_200h'
OUT.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(ROOT/'Report/BTC_200h_Models_Features_Rankings_2026-11-14.pdf'))
thumbs=[]
for i in range(len(doc)):
    im=doc[i].render(scale=1.5).to_pil().convert('RGB')
    im.save(OUT/f'page_{i+1:02d}.png');im.thumbnail((842,595));thumbs.append(im)
for start in range(0,len(thumbs),4):
    sheet=Image.new('RGB',(1720,1240),'#dce3e8');draw=ImageDraw.Draw(sheet)
    for k,im in enumerate(thumbs[start:start+4]):
        x=10+(k%2)*860;y=25+(k//2)*615
        sheet.paste(im,(x,y));draw.text((x,y-17),f'Page {start+k+1}',fill='black')
    sheet.save(OUT/f'contact_{start//4+1:02d}.png')
print(json.dumps(dict(pages=len(doc),directory=str(OUT))))
