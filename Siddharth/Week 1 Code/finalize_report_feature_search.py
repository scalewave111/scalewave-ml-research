"""Wait for the registered run, audit it, and write a factual final record."""
from pathlib import Path
from datetime import datetime,timezone
import json,py_compile,subprocess,sys,time
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'experiments/report38_subset_search_v2'


def main():
    py_compile.compile(str(HERE/'audit_report_feature_search.py'),doraise=True)
    print('WAITING FOR ALL MODEL DIAGNOSTICS',flush=True)
    while not (OUT/'COMPLETE.json').exists():time.sleep(15)
    subprocess.run([sys.executable,str(HERE/'audit_report_feature_search.py')],cwd=HERE,check=True)
    audit=json.loads((OUT/'INDEPENDENT_AUDIT.json').read_text());assert audit['status']=='PASS'
    scores=pd.read_csv(OUT/'scores.csv')
    variants=pd.read_csv(OUT/'removal_variant_scores.csv').sort_values('mae')
    contributions=pd.read_csv(OUT/'feature_contributions.csv')
    selected=pd.read_csv(OUT/'selected_features.csv')
    admission=pd.read_csv(OUT/'feature_admission.csv')
    base=float(scores.set_index('model').loc['baseline','mae'])
    outcomes={name:bool(float(scores.set_index('model').loc[name,'mae'])<base) for name in ['rf','mlp']}
    primary_win=any(outcomes.values())
    best_variant=variants.iloc[0].to_dict() if len(variants) else None
    clock=datetime.now(timezone.utc).isoformat()
    text='Completed feature-subset search - '+clock+'\n\n'
    text+=scores.to_string(index=False)+'\n\n'
    text+='Primary strategy beats original baseline: '+str(outcomes)+'\n'
    text+='All comparisons use the same18481 historical observations. The38 unique requested columns were inspected;32 had usable training variation in every window.\n'
    text+='Up to200 screening subsets per model/window; final candidates use the full previous model recipes. Screening is approximate, not exhaustive enumeration.\n'
    text+='Full diagnostics are retained for compact models. The32-input MLP receives all-feature validation block permutations and six full removals; coverage is explicit in selected_features.csv.\n'
    text+='The historical evaluation was already exposed. Removal variants are post-hoc diagnostic comparisons, not primary validation-selected strategies or independent confirmation.\n\n'
    text+='Fully covered removal variants (best point estimates first):\n'+variants.head(10).to_string(index=False)+'\n\n'
    text+='Independent audit: PASS. Primary model hashes and forecasts unchanged after provisional reporting. Raw inputs unchanged.\n'
    (OUT/'SUMMARY.txt').write_text(text,encoding='utf-8')
    record=dict(status='RESULTS_EXECUTED_AND_INDEPENDENTLY_AUDITED',finished_utc=clock,primary_beats_baseline=outcomes,
        rows=18481,primary_scores=scores.to_dict('records'),best_exploratory_removal_variant=best_variant,
        source_summary=str(OUT/'SUMMARY.txt'),index_refresh_pending=True)
    (OUT/'FINALIZED.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (OUT/'FILE_INDEX_REFRESH.json').write_text('{}\n')
    refresh=subprocess.run([sys.executable,str(ROOT/'refresh_file_index.py')],cwd=ROOT,capture_output=True,text=True)
    index=json.loads((ROOT/'FILE_INDEX_SUMMARY.json').read_text())
    (OUT/'FILE_INDEX_REFRESH.json').write_text(json.dumps(dict(returncode=refresh.returncode,
        file_count=index.get('file_count'),traversal_errors=index.get('traversal_errors',[]),
        scope='Accessible paths indexed; denied bundled-dependency directories are explicitly reported.'),indent=2)+'\n')
    record['index_refresh_pending']=False;record['index_partial']=bool(index.get('traversal_errors'))
    (OUT/'FINALIZED.json').write_text(json.dumps(record,indent=2)+'\n')
    context='\n\n## Final feature-search result - 2026-11-05\n\n'
    context+='Supersedes the provisional execution notes above. Executed on real data and independently audited; all eight model/window runs completed. '
    context+='Primary outcomes: '+str(outcomes)+'.\n\n'
    context+=scores[['model','mae','rank_ic']].to_string(index=False)+'\n\n'
    if best_variant:
        context+='Best fully covered removal diagnostic: '+json.dumps(best_variant)+'. This is an exploratory post-hoc diagnostic, not an independently confirmed or primary-selected strategy.\n\n'
    context+='Sources: Code/experiments/report38_subset_search_v2/{FINALIZED.json,SUMMARY.txt,scores.csv,SELECTIONS.json,selected_features.csv,feature_admission.csv,feature_contributions.csv,removal_variant_scores.csv,uncertainty.csv,INDEPENDENT_AUDIT.json}. '
    context+='All primary forecasts match the earlier preview and saved model hashes. Raw data and previous results remain unchanged. Thirty-two of38 requested unique columns were usable; no exhaustive2^38 search was claimed. Diagnostic scope and operational amendments remain in the run directory. No additional search round or deployment was started. '
    context+='Root index refreshed; bundled-dependency access errors mean the inventory remains partial.\n'
    with (ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as handle:handle.write(context)
    print('FINAL MODEL SCORES\n'+scores.to_string(index=False),flush=True)
    print('REMOVAL VARIANTS\n'+variants.head(12).to_string(index=False),flush=True)
    print('FEATURE CONTRIBUTIONS\n'+contributions.sort_values('full_period_mae_increase_when_removed',ascending=False).head(16).to_string(index=False),flush=True)
    print('ALL RESULTS AUDITED AND RECORDED',flush=True)


if __name__=='__main__':main()
