"""Reformat the audited combined report; retain all evidence and full names."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'Code/render_combined_magnitude_report.py').read_text()
start=s.index('# Wide matrix');end=s.index('W,H=landscape(A4)')
matrix='''# Paginated matrix: one horizon per column, continued on ordinary A4 pages.
from reportlab.platypus import KeepTogether
ms=ParagraphStyle('matrix',fontName='Helvetica',fontSize=12,leading=16,splitLongWords=True,spaceAfter=7)
mhstyle=ParagraphStyle('mh',fontName='Helvetica-Bold',fontSize=22,leading=27,textColor=navy,spaceAfter=12)
matrix_story=[]
for horizon in HS:
 up=next(w for w in WS if w['horizon']==horizon and side(w)=='UP')
 down=next(w for w in WS if w['horizon']==horizon and side(w)=='DOWN')
 chunks=max(1,math.ceil(max(len(up['features']),len(down['features']))/8))
 for chunk in range(chunks):
  if matrix_story:matrix_story.append(PageBreak())
  matrix_story.append(Paragraph('Full feature matrix | '+horizon+' | '+str(chunk+1)+' / '+str(chunks),mhstyle))
  matrix_story.append(Paragraph('Quarter labels: Jan, Apr, Jul, Oct 2024. Full feature names continue across pages. October models also cover January 2025.',ms))
  data=[[Paragraph('<b>Target</b>',ms),Paragraph('<b>'+horizon+' - full feature names</b>',ms)]]
  for w in [up,down]:
   entries=[]
   for f in w['features'][chunk*8:(chunk+1)*8]:
    months=membership(w,f).replace('01','Jan').replace('04','Apr').replace('07','Jul').replace('10','Oct')
    entries.append(Paragraph(html.escape(f)+' <font color="#087f8c">['+months+']</font>',ms))
   if not entries:entries=[Paragraph('Feature list completed on preceding pages.',ms)]
   data.append([Paragraph('<b>'+side(w)+'</b><br/>MAE<br/>'+f"{w['MAE']:.6f}",ms),entries])
  t=Table(data,colWidths=[90,685],hAlign='LEFT')
  t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f7'),colors.white]),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#becbd4')),('INNERGRID',(0,0),(-1,-1),.3,colors.HexColor('#becbd4')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
  matrix_story.append(t)
SimpleDocTemplate(str(TMP/'matrix.pdf'),pagesize=landscape(A4),leftMargin=32,rightMargin=32,topMargin=25,bottomMargin=25).build(matrix_story)
'''
s=s[:start]+matrix+s[end:]
s=s.replace("TMP=ROOT/'tmp/pdfs/combined'","TMP=ROOT/'tmp/pdfs/combined_readable'")
s=s.replace("Report/BTC_All_Horizons_MAE_Models_and_Features_2026-11-14.pdf","Report/BTC_All_Horizons_MAE_Models_and_Features_READABLE_2026-11-14.pdf")
s=s.replace("('head','Helvetica-Bold',12,16)","('head','Helvetica-Bold',15,20)")
s=s.replace("('body','Helvetica',9,12)","('body','Helvetica',12,17)")
s=s.replace("('small','Helvetica',7.6,10)","('small','Helvetica',12,16)")
s=s.replace("('cell','Helvetica',7.7,10)","('cell','Helvetica',12,16)")
s=s.replace("('th','Helvetica-Bold',7.8,10)","('th','Helvetica-Bold',12,16)")
s=s.replace("for q in w['quarters']:\n  text(","for q in w['quarters']:\n  page(w['horizon']+' | '+side(w)+' | '+q['issue'])\n  text(")
s=s.replace("c.setFont('Helvetica',8)","c.setFont('Helvetica',10)")
s=s.replace("['Horizon','Model MAE'","['Hours','Model MAE'")
s=s.replace('str(d.page+1)',"str(d.page+len(PdfReader(TMP/'matrix.pdf').pages))")
s=s.replace("writer.write(OUT)","\nfor i, pg in enumerate(writer.pages):\n lines=(pg.extract_text() or '').splitlines()\n if lines and (('Full feature matrix' in lines[0] and '| 1 /' in lines[0]) or 'Best observed MAE model' in lines[0] or 'MAE winners' in lines[0] or 'definitions and timing' in lines[0] or 'formula dictionary' in lines[0]):writer.add_outline_item(lines[0],i)\nwriter.write(OUT)")
s=s.replace('The first page is the requested full-name feature matrix.','The opening section is the full-name feature matrix, split across standard A4 pages with 12-point text.')
s=s.replace('opening matrix','opening paginated matrix')
s=s.replace("mt=reader.pages[0].extract_text()","mt='\\n'.join(page.extract_text() for page in PdfReader(TMP/'matrix.pdf').pages)")
# Summarize model settings in readable entries instead of dense inline paragraphs.
s=s.replace("if isinstance(x,dict):return '; '.join(str(k)+'='+compact(v) for k,v in x.items())", "if isinstance(x,dict):return '; '.join(str(k)+' = '+compact(v) for k,v in x.items() if k not in ['window_mae','window_rmse','window_mse','features','baseline_window_errors','validation_fraction','verbose','warm_start','categorical_features','interaction_cst','monotonic_cst','n_iter_no_change','scoring','tol','quantile'])")
exec(compile(s,str(ROOT/'Code/render_combined_magnitude_report.py'),'exec'))
