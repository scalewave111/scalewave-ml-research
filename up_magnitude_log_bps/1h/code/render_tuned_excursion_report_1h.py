from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/short_horizon_pdf_deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
out=ROOT/'tmp/pdfs/tuned_excursion_report_1h'
out.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(str(ROOT/'Report/output/pdf/ScaleWave_1h_Tuned_Models_Report.pdf'))
thumbs=[];texts=[]
for i in range(len(doc)):
    page=doc[i]
    texts.append(page.get_textpage().get_text_range())
    im=page.render(scale=1.5).to_pil().convert('RGB')
    im.save(out/f'page_{i+1}.png')
    im.thumbnail((447,632));thumbs.append(im)
for start in range(0,len(thumbs),4):
    sheet=Image.new('RGB',(934,1320),'#dce3e8');draw=ImageDraw.Draw(sheet)
    for j,im in enumerate(thumbs[start:start+4]):
        x=10+(j%2)*467;y=25+(j//2)*660
        sheet.paste(im,(x,y));draw.text((x,y-18),f'Page {start+j+1}',fill='black')
    sheet.save(out/f'contact_{start//4+1}.png')
(out/'extracted_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print(json.dumps({'pages':len(doc),'characters_per_page':[len(t) for t in texts]}))

