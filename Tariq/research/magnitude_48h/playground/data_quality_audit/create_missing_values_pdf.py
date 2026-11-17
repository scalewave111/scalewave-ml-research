from pathlib import Path
import fitz
import pandas as pd

OUT = Path(__file__).resolve().parent
DEST = OUT / 'TEAM_48hr_Missing_Values_Report.pdf'
# User authorized updating the inspected report in place.
date_rows = []
date_indexes = []
for split, label in [('train', 'Training'), ('development', 'Development / validation')]:
    idx = pd.read_parquet(OUT.parent / f'data/y_{split}.parquet').index
    date_indexes.append(idx)
    date_rows.append([label, idx.min().strftime('%d %b %Y %H:%M'), idx.max().strftime('%d %b %Y %H:%M')])
date_rows.append(['Combined', min(i.min() for i in date_indexes).strftime('%d %b %Y %H:%M'), max(i.max() for i in date_indexes).strftime('%d %b %Y %H:%M')])
d = pd.read_csv(OUT / 'feature_missingness.csv')
d['group'] = d.lookback.str.split(' / ').str[-1]
order = ['5m', '15m', '1h', '4h', '12h', '1d']
labels = ['5 minutes', '15 minutes', '1 hour', '4 hours', '12 hours', '1 day']
g = d.groupby('group').agg(columns=('feature','size'), train=('train_missing','sum'),
                         dev=('development_missing','sum'), total=('combined_missing','sum')).loc[order]
assert int(g.train.sum()) == 7893934 and int(g.dev.sum()) == 2678637
doc = fitz.open()
navy = (0.08, 0.18, 0.26)
teal = (0.0, 0.38, 0.40)
gray = (0.28, 0.33, 0.37)
W, H = 595, 842
left, right = 42, 553

def new_page(section):
    global page, y
    page = doc.new_page(width=W, height=H)
    page.draw_rect(fitz.Rect(0,0,W,9),color=teal,fill=teal)
    page.insert_text((left,35),'TEAM 48hr DATA ANALYSIS',fontsize=10,fontname='hebo',color=teal)
    page.insert_text((left,66),section,fontsize=22,fontname='hebo',color=navy)
    y=88

def text(s, size=10.5, bold=False, color=gray, gap=9):
    global y
    font='hebo' if bold else 'helv'
    words=s.split(); lines=[]; line=''
    for word in words:
        proposed=(line+' '+word).strip()
        if fitz.get_text_length(proposed,fontname=font,fontsize=size)>right-left:
            lines.append(line);line=word
        else:line=proposed
    if line:lines.append(line)
    for line in lines:
        page.insert_text((left,y+size),line,fontsize=size,fontname=font,color=color)
        y+=size*1.43
    y+=gap
    assert y < 780, (s,y)

def heading(s):
    global y
    y+=6;text(s,13,True,navy,7)

def table(headers,rows,widths,rowh=26):
    global y
    for ri,row in enumerate([headers]+rows):
        fill=navy if ri==0 else ((0.93,0.96,0.96) if ri%2 else (0.98,0.98,0.98))
        page.draw_rect(fitz.Rect(left,y,right,y+rowh),color=None,fill=fill)
        x=left
        for ci,(value,width) in enumerate(zip(row,widths)):
            value=str(value);font='hebo' if ri==0 else 'helv';size=9
            assert fitz.get_text_length(value,fontname=font,fontsize=size)<width-12, value
            tx=x+7 if ci==0 else x+width-7-fitz.get_text_length(value,fontname=font,fontsize=size)
            page.insert_text((tx,y+18),value,fontsize=size,fontname=font,color=(1,1,1) if ri==0 else navy)
            x+=width
        y+=rowh
    y+=12

fmt=lambda n:f'{int(n):,}'
pct=lambda n:f'{n:.2f}%'
new_page('Missing Values Report')
text('Prepared 11 November 2026 | Scope: supplied Team 48hr repository data',9)
text('10,572,571 missing feature values',22,True,teal,13)
text('42.11% of all feature cells are missing. Most missing values are concentrated in features involving 1-day and 12-hour lookbacks. Together, these groups account for 82.51% of all missing feature values.')
heading('1. Dataset overview')
table(['Dataset','Rows','Features','Missing values','Missing %'],[
 ['Training','29,615','522','7,893,934','51.06%'],
 ['Development','18,481','522','2,678,637','27.77%'],
 ['Combined','48,096','522','10,572,571','42.11%']], [106,72,69,156,108])
table(['Dataset','Total feature cells','Present values'],[
 ['Training','15,459,030','7,565,096'],['Development','9,647,082','6,968,445'],
 ['Combined','25,106,112','14,533,541']], [145,183,183])
table(['Dataset period','First observation (UTC)','Last observation (UTC)'],date_rows,[155,178,178],rowh=22)
text('Inclusive decision times, not 48-hour outcome end times. Ranges include the documented gaps.',9,gap=5)
heading('2. How to read this report')
text('A feature cell is one feature value for one observation. For example, 100 observations with 10 features contain 1,000 feature cells.')
text('A missing value is stored as NaN: no usable reading is supplied. Missing does not mean zero. The repository states that invalid readings were masked as missing; the prepared files do not identify the reason for every missing reading.')
text('Development data is used for evaluation and model comparison, similar to validation data. Its results have repeatedly informed model choices, so it is not an untouched final test set.')
text('All 522 expected feature columns are present. The target files contain no missing values.',11,True,teal)

new_page('Missingness by lookback')
text('A lookback describes how much past market activity a feature summarizes. It is different from our prediction horizon, which remains 48 hours.')
text('Features comparing two lookbacks are grouped under the longer one: for example, 5m / 1d belongs to 1 day. Each feature is counted once. Column counts mean separate features, not column positions.')
heading('3. Training data')
rows=[]
for label,(_,r) in zip(labels,g.iterrows()):
    rows.append([label,fmt(r['columns']),fmt(r.train),pct(r.train/(r['columns']*29615)*100)])
rows.append(['Total','522','7,893,934','51.06%'])
table(['Lookback','Columns','Missing values','Missing %'],rows,[145,85,161,120])
text('All 122 columns involving a 1-day lookback are completely empty across all 29,615 training observations.',11,True,teal)
heading('4. Development / validation data')
rows=[]
for label,(_,r) in zip(labels,g.iterrows()):
    rows.append([label,fmt(r['columns']),fmt(r.dev),pct(r.dev/(r['columns']*18481)*100)])
rows.append(['Total','522','2,678,637','27.77%'])
table(['Lookback','Columns','Missing values','Missing %'],rows,[145,85,161,120])
text('The 12-hour group is substantially more complete in development than in training. The 1-day group remains almost entirely empty.',10)

new_page('Where missing values concentrate')
heading('5. Both splits combined')
rows=[]
for label,(_,r) in zip(labels,g.iterrows()):
    rows.append([label,fmt(r.total),pct(r.total/(r['columns']*48096)*100),pct(r.total/g.total.sum()*100)])
rows.append(['Total','10,572,571','42.11%','100.00%'])
table(['Lookback','Missing values','Within group','Share of missing'],rows,[119,137,116,139])
text('Within group: the percentage of cells missing in that lookback group. Share of missing: the percentage of all missing cells contributed by that group. Percentages are rounded.',9)
text('1-day features contribute 5,844,528 missing values (55.28% of all missing values). 12-hour features contribute 2,878,934 (27.23%). Combined, they contribute 8,723,462 missing values (82.51%).',11,True,teal)
heading('6. Entirely empty versus absent columns')
text('An absent column is not in the file. An entirely empty column exists, but has no usable values. No expected feature columns are absent.')
table(['Scope','Entirely empty','Partially missing','Complete'],[
 ['Training','135','387','0'],['Development','42','284','196'],['Combined','42','480','0']], [145,122,134,110])
text('Of the 135 entirely empty training columns, 122 belong to the 1-day group. Across both splits, 42 columns never contain a usable value: 9 funding, 9 liquidation and 24 volatility features.')
text('Every training observation has at least one missing feature. Dropping every row with any missing value would remove the entire training dataset.',11,True,teal)
text('The 15-minute and 1-hour groups have the lowest training missingness, at approximately 10-12%. Shorter-lookback features are substantially more complete than the 1-day group.')

new_page('Checks and implications')
heading('7. Target completeness and other checks')
text('Both target files have zero missing values across their four columns. Feature and target timestamps align within each split; feature indexes contain no missing or duplicate timestamps. Neither feature matrix contains infinite values.')
text('All six files in the data directory match the supplied archive and recorded manifest. The audit covers the supplied prepared repository dataset, not the separate raw source archives.')
text('Separate timestamp gap: 96 half-hour observations are absent from March 3, 2024, 20:00 through March 5, 2024, 19:30 UTC. This is different from missing cells and from the documented 96-row training-boundary purge. The cause of this gap was not established.')
heading('8. Implications for the current magnitude model')
text('The current magnitude model uses seven selected features with 5-minute, 15-minute and 1-hour lookbacks. It does not use the heavily missing 12-hour or 1-day groups.')
text('The overall 51.06% training missingness is not the missingness rate of the current model\'s seven inputs.',11,True,teal)
text('1. Preserve original files. Perform feature selection and missing-value handling within the modeling pipeline.')
text('2. Exclude entirely empty features in the relevant training fold. They provide no observed values from which to learn.')
text('3. Handle partially missing inputs using training-only rules. Never use future observations to fill earlier missing values.')
text('4. Check feature eligibility in every chronological fold. Availability changes substantially over time. Retain required evaluation rows rather than silently deleting rows with missing inputs.')
text('The original admission rule requires at least 1,000 finite training readings and at least two distinct values. Under that rule, 384 features qualify on the full original training split. Nonempty does not automatically mean eligible.')
heading('9. Conclusion and supporting files')
text('Missingness is concentrated in long-lookback features, particularly the 1-day group. It limits available information, but this audit does not establish that missingness is the main reason further model improvements have been difficult.')
text('Supporting files (same directory): feature_missingness.csv; entirely_missing_features.csv; file_summary.csv; checks.json. Definitions and policies: repository README.md and data/feature_dictionary_and_admission.csv.',9)

for i,p in enumerate(doc):
    p.draw_line((left,800),(right,800),color=(0.78,0.83,0.85),width=0.5)
    p.insert_text((left,817),'ScaleWave | Team 48hr | Prepared dataset audit',fontsize=8,color=gray)
    p.insert_text((right-55,817),f'{i+1} / {len(doc)}',fontsize=8,color=gray)
doc.set_metadata({'title':'TEAM 48hr DATA ANALYSIS - Missing Values Report','author':'ScaleWave','subject':'Missing feature values by lookback, split and feature completeness'})
pdf_bytes = doc.tobytes(garbage=4,deflate=True)
DEST.write_bytes(pdf_bytes)
doc.close()
with fitz.open(DEST) as check:
    content='\n'.join(p.get_text() for p in check)
    for token in ['10,572,571','3,613,030','2,231,498','8,723,462','384','22 Apr 2022 00:30','29 Dec 2023 23:30','01 Jan 2024 00:00','22 Jan 2025 00:00']:
        assert token in content,token
    print(f'Created {DEST}\nPages: {len(check)}\nBytes: {DEST.stat().st_size}')
    check[0].get_pixmap(matrix=fitz.Matrix(1,1)).save(OUT/'missing_values_pdf_preview.png')
