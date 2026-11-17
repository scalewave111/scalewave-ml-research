"""Separate method-labelled summary; never overwrite the original tree reports."""
import io
import csv


def csv_data(rows):
    out=io.StringIO(newline='')
    if rows:
        writer=csv.DictWriter(out,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def make(config,baseline,states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    summary=[]; ranks=[]; selected=[]
    for h in config['horizons']:
        s=states.get(str(h),{}); base=baseline['completed'][str(h)]['pooled']
        best=min(v['MAE_minutes'] for v in base.values()); folds=s.get('policy_folds',{})
        complete=len(folds)==5
        mae=sum(v['score']['absolute_error_sum'] for v in folds.values())/sum(v['validation_rows'] for v in folds.values()) if complete else ''
        summary.append({'horizon_minutes':h,'screened_features':len(s.get('screen',{})),'candidate_features':len(config['candidate_features']),
                        'policy_folds':len(folds),'status':'COMPLETE' if complete else ('PARTIAL' if s else 'PENDING'),
                        'method':'16-bin screen -> top3 inner-only -> six-setting tree -> one selected feature per fold',
                        'baseline_MAE_minutes':best,'policy_MAE_minutes':mae,
                        'policy_MAE_percent_of_horizon':100*mae/h if complete else '',
                        'improvement_percent':100*(best-mae)/best if complete else '',
                        'limitations':'Provisional development; proxy may discard useful features; not exhaustive or fresh-test evidence'})
        ordered=sorted(s.get('screen',{}),key=lambda f:(s['screen'][f]['pooled_MAE_minutes'],f))
        rank={f:i+1 for i,f in enumerate(ordered)}
        for f in config['candidate_features']:
            score=s.get('screen',{}).get(f)
            ranks.append({'target':config['target'],'horizon_minutes':h,'feature':f,'model':'Fixed 16-quantile-bin median',
                          'status':'COMPLETE' if score else 'PENDING','rank_among_screened':rank.get(f,''),
                          'MAE_minutes':score['pooled_MAE_minutes'] if score else '',
                          'MAE_percent_of_horizon':100*score['pooled_MAE_minutes']/h if score else '',
                          'best_baseline_MAE_minutes':best,
                          'improvement_percent':100*(best-score['pooled_MAE_minutes'])/best if score else '',
                          'fallback_folds':sum(v['outer']['status']!='PASS' for v in score['folds']) if score else '',
                          'observed_validation_percent':100*sum(v['outer']['validation_observed'] for v in score['folds'])/sum(v['outer']['validation_rows'] for v in score['folds']) if score else ''})
        for fold,v in folds.items():
            selected.append({'horizon_minutes':h,'fold':int(fold)+1,'selected_feature':v['selected_feature'],
                             'settings':str(v['selected_settings']),'MAE_minutes':v['score']['MAE_minutes'],
                             'validation_rows':v['validation_rows'],'status':v['score']['status'],
                             'origin':v['origin'],'shortlist':'; '.join(v['shortlist'])})
    buf=io.BytesIO()
    with PdfPages(buf) as pdf:
        fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.92,'BTCUSDT | time-to-high staged research',fontsize=23,weight='bold')
        fig.text(.06,.86,'A different method from the original exhaustive single-feature tree screen',fontsize=12)
        ax=fig.add_axes([.06,.34,.88,.44]); ax.axis('off')
        data=[['15m' if r['horizon_minutes']==15 else str(r['horizon_minutes']//60)+'h',f'{r["screened_features"]}/{r["candidate_features"]}',f'{r["policy_folds"]}/5',f'{r["baseline_MAE_minutes"]:.3f}',f'{r["policy_MAE_minutes"]:.3f}' if r['status']=='COMPLETE' else 'Pending',r['status']] for r in summary]
        tab=ax.table(cellText=data,colLabels=['Horizon','Cheap screen','Policy folds','Baseline MAE','Policy MAE','Status'],loc='center'); tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1,1.8)
        notes=['Errors are minutes, not predicted values. Lower is better.',
               'Shortlists/settings use earlier inner validation only. Outer periods assess the selection policy.',
               'Original 15m tree results remain separate; applicable verified fits are reused.',
               'All baseline comparisons use identical evaluation rows. Reserve stays excluded.',
               'Three-feature proxy shortlists can miss useful features. No significance/profitability claim.',
               'No clock-based cutoff; Colab interruption requires manual rerun. Missing work stays marked pending.']
        for i,note in enumerate(notes): fig.text(.06,.27-i*.034,note,fontsize=10)
        pdf.savefig(fig); plt.close(fig)
        for h in config['horizons']:
            rows=[r for r in ranks if r['horizon_minutes']==h and r['status']=='COMPLETE']
            if not rows: continue
            rows=sorted(rows,key=lambda r:r['rank_among_screened'])[:10]
            fig,ax=plt.subplots(figsize=(11.7,8.3))
            labels=[r['feature'].replace('F4Z__','').replace('__value','').replace('__',' / ') for r in rows]
            values=[r['improvement_percent'] for r in rows]
            ax.barh(labels,values,color=['#147d74' if v>0 else '#b85b35' for v in values]); ax.invert_yaxis(); ax.axvline(0,color='black',lw=.6)
            ax.set_title(f'{h}m forecast | top cheap-screen features\nNot a ranking of exhaustively tuned trees')
            ax.set_xlabel('Error reduction versus best pooled baseline (%) — positive is better')
            ax.tick_params(axis='y',labelsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return {'staged_summary.pdf':buf.getvalue(),'horizon_matrix.csv':csv_data(summary),
            'cheap_feature_rankings.csv':csv_data(ranks),'selected_policy_folds.csv':csv_data(selected)}
