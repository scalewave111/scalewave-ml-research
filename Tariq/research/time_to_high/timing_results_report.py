"""Static scientific report and complete CSV tables; no website or raw data."""
import csv
import io
import json
import numpy as np


def tables(config, baseline, jobs, horizon=15):
    base = baseline['completed'][str(horizon)]
    policy = min(base['pooled'], key=lambda k:base['pooled'][k]['MAE_minutes'])
    best = base['pooled'][policy]['MAE_minutes']
    rows, folds = [], []
    for feature in config['candidate_features']:
        job = jobs.get(feature+'|'+str(horizon), {})
        entries = [job.get('folds',{}).get(f'{feature}|{horizon}|{i}') for i in range(len(config['folds']))]
        row = {'rank':'','feature':feature,'family':feature.split('__')[1],
               'feature_lookback':feature.split('__')[3:-1], 'target':config['target'],
               'forecast_horizon_minutes':horizon,'status':'PENDING',
               'completed_folds':sum(e is not None for e in entries),
               'MAE_minutes':'','MAE_percent_of_horizon':'','improvement_minutes':'','improvement_percent':'',
               'folds_beating_reference_policy':'','observed_validation_percent':'','fallback_folds':'',
               'baseline_policy_for_comparison':policy,'baseline_MAE_minutes':best,
               'model':'DecisionTreeRegressor absolute_error seed48',
               'transformation':'Raw feature + missing indicator; train-only median imputation',
               'selected_settings_by_fold':'','verification':'Pending',
               'limitations':'Provisional development; upstream lineage unresolved; ranking is not significance or trading evidence'}
        for name,v in base['pooled'].items(): row['baseline_'+name+'_MAE_minutes']=v['MAE_minutes']
        if row['completed_folds']: row['status']='PARTIAL'
        for i,e in enumerate(entries):
            if e is None: continue
            score = job['fits'][f'{feature}|{horizon}|{i}|outer']
            coverage = score.get('coverage',{})
            folds.append({'feature':feature,'horizon_minutes':horizon,'fold':i+1,
                          'period_start':config['folds'][i][0], 'period_end_exclusive':config['folds'][i][1],
                          'MAE_minutes':e['MAE_minutes'],'validation_rows':e['validation_rows'],
                          'baseline_policy':policy,'baseline_MAE_minutes':base['folds'][i]['scores'][policy]['MAE_minutes'],
                          'selected_settings':e['selected'], 'status':e['status'],
                          'observed_validation_rows':coverage.get('validation_observed',''),
                          'observed_training_rows':coverage.get('train_observed',''),
                          'latest_training_label_end':e['outer_latest_training_label_end']})
        if all(e is not None for e in entries):
            count=sum(e['validation_rows'] for e in entries)
            mae=sum(e['absolute_error_sum'] for e in entries)/count
            assert count==next(iter(base['pooled'].values()))['rows']
            cover=[job['fits'][f'{feature}|{horizon}|{i}|outer'].get('coverage',{}).get('validation_observed') for i in range(len(entries))]
            row.update(status='COMPLETE',MAE_minutes=mae,MAE_percent_of_horizon=100*mae/horizon,
                       improvement_minutes=best-mae,improvement_percent=100*(best-mae)/best,
                       folds_beating_reference_policy=sum(e['MAE_minutes']<base['folds'][i]['scores'][policy]['MAE_minutes'] for i,e in enumerate(entries)),
                       observed_validation_percent=100*sum(cover)/count if all(v is not None for v in cover) else '',
                       fallback_folds=sum(e['status']!='PASS' for e in entries),
                       selected_settings_by_fold=[e['selected'] for e in entries],
                       verification='Receipt checks; tree traversal reported for fitted models; no full raw-data replay')
        rows.append(row)
    ranked=sorted((r for r in rows if r['status']=='COMPLETE'),key=lambda r:(r['MAE_minutes'],r['feature']))
    for i,row in enumerate(ranked,1): row['rank']=i
    rows=ranked+[r for r in rows if r['status']!='COMPLETE']
    return rows,folds,policy


def csv_bytes(rows):
    buffer=io.StringIO(newline='')
    if rows:
        writer=csv.DictWriter(buffer,fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return buffer.getvalue().encode('utf-8-sig')


def artifacts(config,baseline,jobs,horizon=15):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows,folds,policy=tables(config,baseline,jobs,horizon)
    ranked=[r for r in rows if r['status']=='COMPLETE']
    buf=io.BytesIO()
    with PdfPages(buf) as pdf:
        fig=plt.figure(figsize=(11.7,8.3)); fig.patch.set_facecolor('#f5f7fa')
        fig.text(.07,.91,'BTCUSDT | Timing research',fontsize=25,weight='bold',color='#15304c')
        fig.text(.07,.85,f'Target: minutes until the earliest high | Current forecast: {horizon} minutes',fontsize=13)
        fig.text(.07,.77,f'{len(ranked)} / {len(rows)} features complete',fontsize=23,weight='bold')
        fig.text(.07,.72,f'{sum(r["status"]=="PARTIAL" for r in rows)} partial | {sum(r["status"]=="PENDING" for r in rows)} pending',fontsize=13)
        summary=[]
        for h in config['horizons']:
            count=sum(len(j.get('folds',{}))==len(config['folds']) for j in jobs.values() if j['horizon_minutes']==h)
            summary.append(['15m' if h==15 else f'{h//60}h',f'{count} / {len(rows)}', 'Current screen' if h==horizon else ('Pilot only' if count else 'Pending')])
        ax=fig.add_axes([.07,.24,.86,.41]); ax.axis('off')
        table=ax.table(cellText=summary,colLabels=['Forecast horizon','Completed features','Stage'],loc='center',cellLoc='left'); table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1,1.6)
        fig.text(.07,.14,'MAE = average absolute timing error; lower is better. It is not the predicted time.',fontsize=11)
        fig.text(.07,.10,'Feature lookback describes past data; forecast horizon describes the future window.',fontsize=11)
        fig.text(.07,.06,'Development results only. Reserved six months excluded. No significance or profitability claim.',fontsize=10,color='#6b4d00')
        pdf.savefig(fig); plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(11.7,10),gridspec_kw={'height_ratios':[1.2,1]})
        fig.suptitle(f'{horizon}-minute forecast | completed features only',fontsize=19,weight='bold')
        top=ranked[:15]
        if top:
            labels=[r['feature'].replace('F4Z__','').replace('__value','').replace('__',' / ') for r in top]
            values=[r['improvement_minutes'] for r in top]
            axes[0].barh(labels,values,color=['#127c70' if v>0 else '#bd5b32' for v in values]); axes[0].invert_yaxis(); axes[0].axvline(0,color='#172c45',lw=1)
            axes[0].set_xlabel('Minutes of error reduced versus best pooled baseline (positive = better)')
            axes[0].set_title('Top ranked features — a top rank may still lose to the baseline',fontsize=11)
            matrix=np.array([[baseline['completed'][str(horizon)]['folds'][i]['scores'][policy]['MAE_minutes']-jobs[r['feature']+'|'+str(horizon)]['folds'][f'{r["feature"]}|{horizon}|{i}']['MAE_minutes'] for i in range(5)] for r in top])
            limit=max(abs(matrix).max(),.001)
            plot=axes[1].imshow(matrix,cmap='BrBG',vmin=-limit,vmax=limit,aspect='auto')
            axes[1].set_yticks(range(len(top)),labels=labels,fontsize=8)
            axes[1].set_xticks(range(5),labels=[v[0][:7] for v in config['folds']])
            axes[1].set_title('Consistency: improvement against the same baseline policy in each period',fontsize=11)
            for i in range(len(top)):
                for j in range(5): axes[1].text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=7)
            fig.colorbar(plot,ax=axes[1],label='Minutes of error reduced')
        else:
            for ax in axes: ax.text(.5,.5,'No completed feature results yet',ha='center',transform=ax.transAxes); ax.axis('off')
        fig.tight_layout(rect=[.02,.02,.98,.94]); pdf.savefig(fig); plt.close(fig)
        fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.93,'Coverage, interpretation and verification',fontsize=21,weight='bold')
        lines=[f'Comparison policy: {policy}',
               'The reference policy is chosen by pooled development MAE, not a deployable future selector.',
               'Green / positive means less error. Negative means the baseline performed better.',
               'Rankings include complete five-fold results only. Partial and pending features are in the CSV.',
               'All three baseline MAEs, fold settings and coverage are provided in the complete tables.',
               'Fallback rows remain in evaluation. Unknown legacy fallback coverage is left blank.',
               'Earlier-only training and inner selection; strict label-end purging; no reserved rows.',
               'Source availability gates are applied, but original F4Z publication/normalization lineage',
               'remains unresolved. Features and results are PROVISIONAL.',
               'Selecting among many features on development data can overfit. No fresh-test claim.',
               'A standalone feature screen does not test feature sets or connected prediction models.']
        for i,line in enumerate(lines): fig.text(.06,.85-i*.045,line,fontsize=11)
        ax=fig.add_axes([.06,.06,.88,.25]); ax.axis('off')
        view=[[r['rank'],r['feature'].replace('F4Z__','').replace('__value',''),f'{r["MAE_minutes"]:.4f}',f'{r["folds_beating_reference_policy"]}/5',f'{r["observed_validation_percent"]:.1f}%' if r['observed_validation_percent']!='' else 'Unknown',r['fallback_folds']] for r in ranked[:5]]
        if view:
            tab=ax.table(cellText=view,colLabels=['Rank','Feature','MAE min','Wins','Observed','Fallback folds'],loc='center',colWidths=[.05,.48,.12,.08,.12,.15]); tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1,1.8)
        pdf.savefig(fig); plt.close(fig)
    return {f'{horizon}m_summary.pdf':buf.getvalue(),f'{horizon}m_feature_rankings.csv':csv_bytes(rows),
            f'{horizon}m_fold_details.csv':csv_bytes(folds)}
