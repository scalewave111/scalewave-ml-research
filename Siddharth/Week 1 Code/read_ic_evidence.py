from pathlib import Path
root = Path(__file__).resolve().parents[1]
for p in (root/'Report/ic_rf_sources').glob('*.txt'):
    print('\nSOURCE', p.name)
    content = p.read_text(encoding='utf-8')
    if p.name.startswith('Top 20'):
        print(content.encode('ascii', 'replace').decode())
    else:
        lines = content.splitlines()
        selected = set()
        for i,s in enumerate(lines):
            if any(k in s.lower() for k in ['spearman','stationar','magnitude','feature engineering','random forest','rank ic','delta ic']):
                selected.update(range(max(0,i-2),min(len(lines),i+5)))
        print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected))[:23000].encode('ascii','replace').decode())
