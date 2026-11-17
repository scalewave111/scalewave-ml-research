"""Build a small text-based PDF from the summary; no data/network/training."""
from pathlib import Path
import html
import re
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'reports/research_summary.md').read_text(encoding='utf-8')
blocks=[]
for block in source.strip().split('\n\n'):
    if block.startswith('# '):blocks.append('<h1>'+html.escape(block[2:])+'</h1>')
    elif block.startswith('## '):blocks.append('<h2>'+html.escape(block[3:])+'</h2>')
    else:blocks.append('<p>'+html.escape(block)+'</p>')
css='''body {font-family: sans-serif; font-size: 10.5pt; color: #233047; line-height: 1.3;}
h1 {font-size: 24pt; color: #123b62;} h2 {font-size: 13pt; color: #123b62; margin-top: 12pt;}
p {margin-bottom: 8pt;}'''
story=pymupdf.Story(html=''.join(blocks),user_css=css)
out=ROOT/'reports/research_summary.pdf'
writer=pymupdf.DocumentWriter(str(out))
page=pymupdf.paper_rect('a4'); rect=page+(45,45,-45,-45)
more=True
while more:
    device=writer.begin_page(page)
    more,_=story.place(rect)
    story.draw(device)
    writer.end_page()
writer.close()
with pymupdf.open(out) as pdf:
    assert len(pdf)>=1
    extracted=' '.join(p.get_text() for p in pdf)
    assert '180.798518' in extracted and 'What should happen next?' in extracted
    print(f'{out.name}: {len(pdf)} pages, {out.stat().st_size} bytes; text extraction passed')
