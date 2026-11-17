from pathlib import Path
p=Path('lstm_walk_forward.py')
s=p.read_text(encoding='utf-8-sig')
old="   error=max(error,float(np.max(abs(pred-p.loc[te,name]))));np.testing.assert_allclose(pred,p.loc[te,name],rtol=0,atol=1e-5)"
new="   error=max(error,float(np.max(abs(pred-p.loc[te,name]))))\n   if pred.dtype==np.float32:\n    np.testing.assert_array_equal(pred,p.loc[te,name].to_numpy(dtype=np.float32))\n   else:\n    np.testing.assert_allclose(pred,p.loc[te,name],rtol=0,atol=1e-10)"
assert old in s
s=s.replace(old,new)
s=s.replace("source_sha256=sha(__file__),limitation='Previously exposed historical evaluation; not fresh holdout'","training_source_sha256=json.loads((OUT/'CONTRACT.json').read_text())['source_sha256'],audit_source_sha256=sha(__file__),limitation='Previously exposed historical evaluation; not fresh holdout'")
p.write_text(s,encoding='utf-8')
