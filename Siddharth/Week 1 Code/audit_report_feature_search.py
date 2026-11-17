"""Independent saved-result checks; does not fit models or change selection."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'Code/experiments/report38_subset_search_v2'


def main():
    contract=json.loads((OUT/'CONTRACT.json').read_text())
    assert hashlib.sha256((OUT/'source_snapshot.py').read_bytes()).hexdigest()==contract['source_sha256']
    assert hashlib.sha256((ROOT/'Code/search_report_features.py').read_bytes()).hexdigest()==contract['source_sha256']
    p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    reference=pd.read_csv(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',index_col=0,parse_dates=True)
    assert len(p)==18481 and p.index.is_unique and p.index.equals(reference.index)
    np.testing.assert_allclose(p.truth,reference.truth,rtol=0,atol=1e-8)
    np.testing.assert_allclose(p.baseline,reference.report_hybrid,rtol=0,atol=1e-8)
    preview_path=OUT/'PRIMARY_PREVIEW.json'
    if preview_path.exists():
        preview=json.loads(preview_path.read_text())
        for path,digest in preview['outer_model_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        previous=pd.read_csv(OUT/'PRIMARY_PREVIEW_predictions.csv',index_col=0,parse_dates=True)
        assert previous.index.equals(p.index)
        for model in preview['models_ready']:
            np.testing.assert_allclose(previous[model],p[model],atol=1e-9,rtol=0)
    scores=pd.read_csv(OUT/'scores.csv').set_index('model')
    for column in p.columns.drop('truth'):
        assert np.isfinite(p[column]).all() and p[column].min()>=0
        np.testing.assert_allclose(abs(p[column]-p.truth).mean(),scores.loc[column,'mae'],atol=1e-9,rtol=0)
        ic=spearmanr(np.round(p[column],10),p.truth).statistic if np.ptp(p[column])>0 else 0.
        np.testing.assert_allclose(ic,scores.loc[column,'rank_ic'],atol=1e-10,rtol=0)
    allowed=set(contract['unique_requested_columns']+contract['optional_original_controls'])
    selection_rows=[];admission_rows=[];component_rows=[];permutation_rows=[]
    amendment_path=OUT/'DIAGNOSTIC_AMENDMENT.json'
    if amendment_path.exists():
        amendment=json.loads(amendment_path.read_text())
        assert hashlib.sha256((OUT/'focused_worker_snapshot.py').read_bytes()).hexdigest()==amendment['focused_worker_sha256']
    parallel_path=OUT/'RF_PARALLELISM_AMENDMENT.json'
    if parallel_path.exists():
        parallel=json.loads(parallel_path.read_text())
        assert parallel['replay_benchmark']['status']=='PASS'
        assert hashlib.sha256((OUT/'parallel_rf_worker_snapshot.py').read_bytes()).hexdigest()==parallel['parallel_worker_sha256']
    import joblib
    for folder in sorted(pth.parent for pth in OUT.glob('*/*/SELECTION.json')):
        selection=json.loads((folder/'SELECTION.json').read_text())
        rows=json.loads((folder/'FULL_VALIDATION.json').read_text())
        expected=min(rows,key=lambda r:(r['mae'],len(r['cols']),r['id']))
        assert expected==selection['selected']
        assert selection['selected_before_outer_predictions'] is True
        features=expected['cols'];assert set(features)<=allowed
        admission=json.loads((folder/'ADMISSION.json').read_text())
        admitted={r['feature'] for r in admission if r['admitted']}
        assert set(features)<=admitted|set(contract['optional_original_controls'])
        for r in admission:admission_rows.append(dict(method=selection['method'],issue=selection['issue'],**r))
        ablations=json.loads((folder/'ABLATIONS.json').read_text())
        tested={a['removed'] for a in ablations}
        if (folder/'DIAGNOSTIC_SCOPE.json').exists():
            scope=json.loads((folder/'DIAGNOSTIC_SCOPE.json').read_text())
            assert scope['selected_features']==features
            if len(features)>1:assert tested==set(scope['fully_refitted_removals'])
        for feature in features:selection_rows.append(dict(method=selection['method'],issue=selection['issue'],feature=feature,removal_refitted=feature in tested))
        if (folder/'VALIDATION_PERMUTATION.json').exists():
            for row in json.loads((folder/'VALIDATION_PERMUTATION.json').read_text()):
                permutation_rows.append(dict(method=selection['method'],issue=selection['issue'],**row))
        inner=joblib.load(folder/'full'/expected['id']/'inner/bundle.joblib')
        outer=joblib.load(folder/'full'/expected['id']/'outer/bundle.joblib')
        if selection['method']=='rf':
            assert inner['weight']==outer['weight']
            component_rows.append(dict(method='rf',issue=selection['issue'],rf_weight=outer['weight'],epochs=''))
        else:
            assert inner['epochs']==outer['epochs']
            component_rows.append(dict(method='mlp',issue=selection['issue'],rf_weight=None,epochs=str(outer['epochs'])))
        f=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
        assert f.index.min()>=pd.Timestamp(selection['issue'])
        np.testing.assert_allclose(f.selected,p.loc[f.index,selection['method']],atol=1e-9,rtol=0)
        for ablation in json.loads((folder/'ABLATIONS.json').read_text()):
            assert set(ablation['cols'])==set(features)-{ablation['removed']}
    assert len(component_rows)==8
    pd.DataFrame(selection_rows).to_csv(OUT/'selected_features.csv',index=False)
    pd.DataFrame(admission_rows).to_csv(OUT/'feature_admission.csv',index=False)
    pd.DataFrame(component_rows).to_csv(OUT/'selected_model_details.csv',index=False)
    pd.DataFrame(permutation_rows).to_csv(OUT/'validation_permutation.csv',index=False)
    selected_counts=pd.DataFrame(selection_rows).groupby(['method','feature']).size()
    ab=pd.read_csv(OUT/'feature_ablations.csv')
    summary=[];variant_rows=[];variant_predictions=pd.DataFrame(index=p.index)
    for (method,feature),group in ab.groupby(['method','feature']):
        weighted=(group.mae_increase_when_removed*group.rows).sum()
        selected_count=int(selected_counts.loc[method,feature])
        summary.append(dict(method=method,feature=feature,selected_windows=selected_count,removal_tested_windows=len(group),
            tested_rows=int(group.rows.sum()),full_period_mae_increase_when_removed=float(weighted/len(p)) if len(group)==selected_count else None,
            conditional_mae_increase_when_removed=float(weighted/group.rows.sum()),
            validation_increase_weighted_by_test_window_rows=float(np.average(group.validation_mae_increase_when_removed,weights=group.rows)),
            windows_positive=int((group.mae_increase_when_removed>0).sum())))
        if len(group)==selected_count:
            # A genuine full-period removal: keep the model unchanged only in
            # windows where the feature was never present in the selected set.
            candidate=p[method].copy()
            for _,row in group.iterrows():
                folder=OUT/method/f'{pd.Timestamp(row.issue):%Y%m%d}'
                records=json.loads((folder/'ABLATIONS.json').read_text())
                record=next(a for a in records if a['removed']==feature)
                frame=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
                candidate.loc[frame.index]=frame[record['prediction_column']]
            assert candidate.index.equals(p.index) and np.isfinite(candidate).all()
            name=method+'_drop_'+hashlib.sha256(feature.encode()).hexdigest()[:10]
            assert name not in variant_predictions
            variant_predictions[name]=candidate
            mae=float(abs(candidate-p.truth).mean())
            np.testing.assert_allclose(mae-float(scores.loc[method,'mae']),weighted/len(p),atol=1e-8,rtol=0)
            variant_rows.append(dict(model=name,method=method,removed_feature=feature,mae=mae,
                rank_ic=float(spearmanr(np.round(candidate,10),p.truth).statistic) if np.ptp(candidate)>0 else 0.,
                rows=len(p),removal_tested_windows=len(group),mae_difference_vs_baseline=mae-float(scores.loc['baseline','mae'])))
    pd.DataFrame(summary).sort_values('full_period_mae_increase_when_removed',ascending=False).to_csv(OUT/'feature_contributions.csv',index=False)
    pd.DataFrame(variant_rows,columns=['model','method','removed_feature','mae','rank_ic','rows','removal_tested_windows','mae_difference_vs_baseline']).to_csv(OUT/'removal_variant_scores.csv',index=False)
    variant_predictions.to_csv(OUT/'removal_variant_predictions.csv',index_label='decision_time')
    (OUT/'REMOVAL_VARIANT_INTERPRETATION.json').write_text(json.dumps(dict(
        role='Exploratory diagnostic comparisons, not the primary validation-selected strategies.',
        construction='Only features with a full refit in every selected window are eligible. Use the saved removal prediction in those windows; unchanged primary prediction elsewhere because that feature was absent.',
        selection_warning='Ranking these variants after seeing outer scores is post-hoc. Do not call the best such variant independently confirmed or validation-selected.',
        count=len(variant_rows)),indent=2)+'\n')
    result=dict(status='PASS',rows=len(p),windows=8,independent_mae_and_spearman_recalculation=True,
        exact_reference_alignment=True,validation_argmin_reproduced=True,requested_feature_scope_verified=True,
        training_only_admission_verified=True,outer_weights_and_epochs_equal_inner_choices=True,
        source_unchanged=True,ablation_features_verified=True,diagnostic_coverage_explicit=True,primary_fits_unchanged_after_preview=True)
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
