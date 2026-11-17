from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/short_horizon_pdf_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
out=ROOT/'tmp/pdfs/short_horizon_reports'
out.mkdir(parents=True,exist_ok=True)
result=[]
for h in [1,4]:
    path=ROOT/f'Report/output/pdf/ScaleWave_{h}h_Models_Ranked_MAE_IC_2026-11-11.pdf'
    doc=pdfium.PdfDocument(str(path));thumbs=[]
    for i in range(len(doc)):
        im=doc[i].render(scale=1.5).to_pil().convert('RGB')
        im.save(out/f'{h}h_page_{i+1}.png')
        im.thumbnail((720,510));thumbs.append(im)
    for start in range(0,len(thumbs),4):
        sheet=Image.new('RGB',(1480,1080),'#dce3e8');draw=ImageDraw.Draw(sheet)
        for k,im in enumerate(thumbs[start:start+4]):
            x=10+(k%2)*740;y=25+(k//2)*540
            sheet.paste(im,(x,y));draw.text((x,y-18),f'{h}h | page {start+k+1}',fill='black')
        sheet.save(out/f'{h}h_contact_{start//4+1}.png')
    result.append(dict(horizon=h,pages=len(doc)))
print(json.dumps(result))
