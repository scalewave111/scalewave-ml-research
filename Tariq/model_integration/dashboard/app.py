"""ScaleWave local, read-only integration dashboard."""
import json
import math
from datetime import datetime, timedelta
from urllib.parse import urlparse

import plotly.graph_objects as go
import streamlit as st

from state import snapshot, simulate, finite

st.set_page_config(page_title='ScaleWave | Model Operations', page_icon='📊', layout='wide')
st.markdown('''<style>
.block-container {max-width:1320px;padding-top:2rem;padding-bottom:3rem}
h1 {letter-spacing:-.04em} h2,h3 {letter-spacing:-.02em}
[data-testid="stMetric"] {border:1px solid #d7e5e1;border-radius:12px;padding:18px;background:#f4faf7}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color:#234c40 !important;opacity:1 !important}
[data-testid="stMetricLabel"] p {font-size:0.95rem;font-weight:600}
[data-testid="stMetricValue"] {color:#12614d}
[data-testid="stCaptionContainer"] {font-size:0.95rem;color:inherit;opacity:0.85}
[data-testid="stCaptionContainer"] p {color:inherit;font-size:inherit}
</style>''', unsafe_allow_html=True)
data = snapshot()
config, result = data['config'], data['result']


def chart_layout(fig, **kwargs):
    fig.update_layout(template='plotly_white', font={'family':'Arial', 'size':14},
                      margin={'l':20,'r':20,'t':40,'b':25}, **kwargs)
    return fig


def formatted(value, digits=6):
    return f'{value:,.{digits}f}' if finite(value) else 'Not available'


with st.sidebar:
    st.title('ScaleWave')
    st.caption('MODEL OPERATIONS / LOCAL')
    st.button('Refresh saved records', help='Rereads the files. Does not execute either model.')
    st.divider()
    st.write('**Mode:** development')
    st.write('**Trading:** disabled in this milestone')
    st.write('**Market feed:** not connected')
    st.caption('This dashboard reads local JSON records. It does not contact MLflow automatically or load model binaries.')
    if result:
        st.write('**Last recorded attempt (UTC)**')
        st.code(result.get('created_at_utc', 'Unknown'), language=None, wrap_lines=True)
    if data['age_hours'] is not None:
        st.caption(f'Record age: {data["age_hours"]:,.1f} hours. This is saved replay evidence, not a live heartbeat.')

st.caption('SCALEWAVE / INTEGRATION WORKSPACE')
st.title('From model handoff to a working system')
st.write('Inspect the selected models, their recorded verification, and what remains before real-data operation.')
st.info('Synthetic example replay only. No live market forecast, actual return or trading P&L is available here.')
for warning in data['warnings']:
    st.warning(warning)
tabs = st.tabs(['Overview', 'Models & evidence', 'Checks & history', 'Score simulator', 'Pipeline monitoring'])

with tabs[0]:
    st.subheader('Latest verified example')
    if data['verified_outputs']:
        st.success('Both frozen models reproduced their supplied example outputs. Selected configuration matches this record.')
    else:
        st.warning('No valid replay for the selected configuration is available. Prediction metrics are withheld.')
    cols = st.columns(4)
    cols[0].metric('Probability of UP', f'{data["p_up"]*100:.4f}%' if data['verified_outputs'] else '—')
    cols[1].metric('Magnitude · bps', formatted(data['magnitude']))
    cols[2].metric('Signed score · bps', formatted(data['score']))
    cols[3].metric('Recorded attempt · seconds', formatted(result.get('elapsed_seconds'), 3))
    st.caption('Magnitude estimates the conditional median absolute return. Signed score is a combined signal, not a calibrated expected return.')
    if data['verified_outputs']:
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown('**Direction probability**')
            p = data['p_up']
            fig = go.Figure(go.Bar(x=['UP', 'Non-UP'], y=[100*p,100*(1-p)],
                                  marker_color=['#168365','#6985a1'], text=[f'{100*p:.2f}%',f'{100*(1-p):.2f}%'], textposition='auto'))
            fig.update_yaxes(range=[0,100], title='Probability (%)')
            st.plotly_chart(chart_layout(fig, height=285), use_container_width=True)
            st.caption('Non-UP is 1 − P(UP); it includes unchanged outcomes under a positive-return classifier.')
        with right:
            st.markdown('**What time does this forecast refer to?**')
            try:
                origin = datetime.fromisoformat(result['forecast_time_utc'].replace('Z','+00:00'))
                fig = go.Figure()
                for role, model in config['models'].items():
                    window = model['target_window']
                    fig.add_trace(go.Scatter(x=[origin+timedelta(minutes=window['entry_offset_minutes']),origin+timedelta(minutes=window['exit_offset_minutes'])],
                                  y=[role,role], mode='lines+markers', name=role.capitalize(), line={'width':7}, marker={'size':12}))
                st.plotly_chart(chart_layout(fig, height=285, xaxis_title='Reference-price times (UTC)', showlegend=False), use_container_width=True)
                st.caption(f'Example forecast origin: {result["forecast_time_utc"]}. These are target windows, not actual fills.')
            except (KeyError, ValueError, TypeError):
                st.warning('Target timeline could not be read.')
    st.subheader('Readiness to progress')
    count = sum(c['Status']=='PASS' for c in data['checks'])
    st.dataframe([
        {'Stage':'Package replay','State':f'{count}/{len(data["checks"])} saved checks passed' if data['matches'] else 'Current verification unavailable','Next action':'Review errors if any'},
        {'Stage':'Target windows','State':result.get('target_alignment','Unknown') if data['matches'] else 'Unknown for selection','Next action':'Keep matching definitions on upgrades'},
        {'Stage':'Historical feature inputs','State':'Deferred — source data is on the company PC','Next action':'Connect the approved dataset and reproduce one real feature row'},
        {'Stage':'Combined-score evaluation','State':'Not performed','Next action':'Evaluate on real matched inputs and observed returns'},
        {'Stage':'Trading rule & risk controls','State':'Proposed, not implemented','Next action':'Validate score threshold, exits, costs and risk limits'},
        {'Stage':'Scheduled execution','State':'Not configured','Next action':'Add scheduling after real-data integration'},
    ], use_container_width=True, hide_index=True)
    st.caption('Readiness items describe this project milestone. They are not an automated health check of remote services.')

with tabs[1]:
    st.subheader('Selected model packages')
    for column, (role, model) in zip(st.columns(2), config.get('models', {}).items()):
        with column:
            st.markdown(f'### {role.capitalize()}')
            st.code(model.get('version','Unknown'), language=None)
            st.write('**Training cutoff:**', model.get('training_cutoff_utc','Unknown'))
            st.write('**Output:**', model.get('output_field','Unknown'))
            st.write(model.get('target_definition','Target definition not recorded.'))
            st.caption('MLflow run ID')
            st.code(model.get('run_id','Unknown'), language=None)
            uri = config.get('tracking_uri','')
            if urlparse(uri).scheme == 'https' and urlparse(uri).hostname == 'scalewave.tailf4ff21.ts.net':
                st.link_button('Open MLflow run', uri.rstrip('/')+'/#/experiments/2/runs/'+model['run_id'])
            st.caption('MLflow access requires company connectivity and your login. No credentials are stored in this dashboard.')
    st.divider()
    st.subheader('Owner-submitted evaluation evidence')
    st.warning('These scores are read from the submitted evaluation reports. This integration has not rerun the historical evaluation. The overall figures span seven quarterly checkpoints, not just the distributed checkpoint.')
    role = st.selectbox('Model to inspect', ['direction','magnitude'], format_func=str.capitalize)
    ev = data['evaluations'].get(role,{})
    if ev:
        st.write(f'**Metric:** {ev.get("primary_metric","Unknown")} · **Reported rows:** {ev.get("row_count","Unknown"):,}')
        st.caption(f'{ev.get("evaluation_start_utc","Unknown")} to {ev.get("evaluation_end_utc","Unknown")}')
        values = [ev.get('baseline_score'),ev.get('model_score')]
        if all(finite(v) for v in values):
            fig = go.Figure(go.Bar(x=['Baseline','Submitted model'], y=values, marker_color=['#9aaeb9','#168365'],
                                  text=[f'{v:.6f}' for v in values], textposition='auto'))
            fig.update_yaxes(title=ev.get('primary_metric','Score') + ' (lower is better)')
            st.plotly_chart(chart_layout(fig,height=300),use_container_width=True)
        st.caption('Compare only matching targets, metrics and evaluation rows. Direction and magnitude scores use different metrics.')
        with st.expander('Distributed checkpoint’s own reported evaluation'):
            st.json(ev.get('frozen_checkpoint_test',{}))
    else:
        st.info('No evaluation report is available.')
    st.caption('Original training Git commits are unavailable. The handoff commit identifies submitted packages, not original training history.')

with tabs[2]:
    st.subheader('Saved verification checks')
    st.write('**Recorded status:**', result.get('status','No result'))
    if result.get('error'):
        st.error(str(result['error']))
    only_failures = st.checkbox('Show only checks that did not pass')
    rows = [c for c in data['checks'] if not only_failures or c['Status']!='PASS']
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info('No checks match this view.')
    st.subheader('Compact attempt history')
    st.caption('These are integration attempts, not market-performance observations. Older versions may have been pruned locally; their records remain evidence of past checks.')
    history_rows=[]
    for h in data['history']:
        history_rows.append({'Recorded at (UTC)':h.get('created_at_utc','Unknown'), 'Status':h.get('status','Unknown'),
                             'Direction version':h.get('models',{}).get('direction',{}).get('version','Unknown'),
                             'Magnitude version':h.get('models',{}).get('magnitude',{}).get('version','Unknown'),
                             'Target alignment':h.get('target_alignment','Unknown'), 'Signed score (bps)':h.get('predicted_signed_score_bps'),
                             'Seconds':h.get('elapsed_seconds')})
    st.dataframe(history_rows, use_container_width=True, hide_index=True)
    st.caption('Missing duration means it was not recorded; it is not zero. No equity curve or P&L is fabricated from replay attempts.')
    with st.expander('Inspect the current result record'):
        st.json(result)
    st.download_button('Download current result JSON', json.dumps(result, indent=2), file_name='scalewave_current_result.json', mime='application/json')

with tabs[3]:
    st.subheader('Explore the score formula')
    st.info('SIMULATION ONLY — changing these controls does not run a model, update configuration, or place a trade.')
    a,b,c = st.columns(3)
    p_percent = a.slider('Probability of UP (%)',0.0,100.0,70.0,0.5)
    magnitude = b.number_input('Predicted magnitude (bps)',min_value=0.0,max_value=10000.0,value=20.0,step=1.0)
    threshold = c.number_input('Proposed absolute score threshold (bps)',min_value=0.1,max_value=10000.0,value=10.0,step=0.5)
    sim = simulate(p_percent/100,magnitude,threshold)
    st.metric('Simulated signed score · bps',f'{sim["score"]:+.6f}')
    st.write('**Filter interpretation:**',sim['interpretation'])
    st.code(f'(2 × {p_percent/100:.4f} − 1) × {magnitude:.4f} = {sim["score"]:+.6f} bps',language=None)
    probabilities = [i for i in range(101)]
    fig = go.Figure(go.Scatter(x=probabilities,y=[(2*i/100-1)*magnitude for i in probabilities],name='Signed score',line={'color':'#168365','width':3}))
    fig.add_hline(y=threshold,line_dash='dash',line_color='#aa6c18',annotation_text='Long boundary')
    fig.add_hline(y=-threshold,line_dash='dash',line_color='#aa6c18',annotation_text='Short boundary')
    fig.add_trace(go.Scatter(x=[p_percent],y=[sim['score']],mode='markers',name='Selected inputs',marker={'size':13,'color':'#243d64'}))
    fig.update_xaxes(title='Probability of UP (%)')
    fig.update_yaxes(title='Signed score (bps)')
    st.plotly_chart(chart_layout(fig,height=350),use_container_width=True)
    if magnitude <= threshold:
        st.write('This magnitude cannot strictly exceed the threshold, even with 100% probability in either direction.')
    else:
        boundary=(1+threshold/magnitude)/2*100
        st.write(f'To strictly exceed this threshold: UP probability > {boundary:.3f}% for long, or UP probability < {100-boundary:.3f}% for short.')
    st.caption('At exactly the threshold, the proposed strict rule says no trade. A score threshold is not a guaranteed return or profit after costs.')

@st.fragment(run_every=3)
def monitoring_panel():
    current = snapshot()
    record = current['monitoring']
    status = current['monitor_status']
    st.subheader('Pipeline progress')
    st.caption('Refreshes every 3 seconds while the page is open. This monitors the local replay process, not market data or remote services.')
    if status == 'NOT_RECORDED':
        st.info('No monitored attempt yet. Run the replay command below.')
    elif status == 'POSSIBLY_INTERRUPTED':
        st.error('Possibly interrupted: no recent heartbeat. Inspect the terminal and runner lock before retrying. This does not prove the process has stopped.')
    elif status in ('FAILED','INTERRUPTED'):
        st.error('Latest attempt: ' + status + '. Dependent steps were skipped.')
    elif status == 'RUNNING':
        st.info('Running. A heartbeat is written every 5 seconds.')
    else:
        st.success('Attempt completed. This is development replay evidence, not trading readiness.')
    if record:
        st.caption(f'Attempt: {record.get("attempt_id","Unknown")} | Mode: {record.get("mode","Unknown")} | Updated: {record.get("heartbeat_at_utc","Unknown")}')
        rows=[{'Step':s['name'], 'Status':s['status'], 'Seconds':s.get('duration_seconds'),
               'Started (UTC)':s.get('started_at_utc'), 'Finished (UTC)':s.get('finished_at_utc'),
               'Detail':s.get('detail','')} for s in record.get('steps',[])]
        st.dataframe(rows,hide_index=True,use_container_width=True)
        timed=[s for s in record.get('steps',[]) if finite(s.get('duration_seconds'))]
        if timed:
            fig=go.Figure(go.Bar(x=[s['duration_seconds'] for s in timed],y=[s['name'] for s in timed],orientation='h',
                                marker_color=['#ba4a3e' if s['status']=='FAILED' else '#168365' for s in timed]))
            fig.update_xaxes(title='Measured seconds')
            st.plotly_chart(chart_layout(fig,height=330),use_container_width=True,key='monitor_durations')
        st.caption('Model-step duration includes environment verification, inference and output validation. Save runs even after a failure to record it.')
    st.markdown('**Run a monitored replay from the integration folder in PowerShell:**')
    st.code("& '.\\.venv-model\\Scripts\\python.exe' -B pipeline.py --mode replay --model-python '.\\.venv-model\\Scripts\\python.exe'",language='powershell')
    st.caption('This page never starts a model or sends a trade. Alerts are displayed here only; no external notifications are configured.')

with tabs[4]:
    monitoring_panel()
