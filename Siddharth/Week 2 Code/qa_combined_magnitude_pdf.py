from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tmp/pdf120_render_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
d=pdfium.PdfDocument(str(R/'Report/BTC_All_Horizons_MAE_Models_and_Features_2026-11-14.pdf'));out=R/'tmp/pdfs/combined';thumbs=[]
for i in range(len(d)):
 im=d[i].render(scale=.17 if i==0 else 1).to_pil().convert('RGB');im.save(out/f'page_{i+1:02d}.png');im.thumbnail((600,430));thumbs.append(im)
for start in range(0,len(thumbs),12):
 sheet=Image.new('RGB',(1840,1810),'#dde4eb');dr=ImageDraw.Draw(sheet)
 for k,im in enumerate(thumbs[start:start+12]):
  x=10+(k%3)*610;y=25+(k//3)*450;sheet.paste(im,(x,y));dr.text((x,y-17),str(start+k+1),fill='black')
 sheet.save(out/f'contact_{start//12+1}.png')
print('Rendered',len(d),'pages')
