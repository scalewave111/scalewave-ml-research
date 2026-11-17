"""Offline fault injection; no network, credentials or real sleeps."""
import io
import json
import runpy
import sys
import types
from pathlib import Path

guard=runpy.run_path(str(Path(__file__).with_name('drive_rate_guard.py')))
class Clock:
    def __init__(self): self.now=0; self.waits=[]
    def clock(self): return self.now
    def sleep(self,value): self.waits.append(value); self.now+=value
clock=Clock(); gate=guard['Gate'](3,clock.clock,clock.sleep)
limited=json.dumps({'error':{'errors':[{'reason':'rateLimitExceeded'}]}}).encode()
permission=json.dumps({'error':{'errors':[{'reason':'insufficientPermissions'}]}}).encode()
def decode(value): return *value,lambda:None
body=io.BytesIO(b'unchanged checkpoint'); calls=[]
def send():
    calls.append((clock.now,body.read()))
    return (403,limited,{}) if len(calls)<3 else (200,b'ok',{})
result=guard['run_request'](send,decode,gate,body,max_retries=3)
assert result[0]==200 and len(calls)==3
assert all(v[1]==b'unchanged checkpoint' for v in calls)
assert all(calls[i+1][0]-calls[i][0]>=3 for i in range(2))
for status,content in [(403,permission),(500,b'ambiguous create failure')]:
    count=[]
    def no_retry(): count.append(1); return status,content,{}
    assert guard['run_request'](no_retry,decode,gate)[0]==status and len(count)==1
count=[]
def exhausted(): count.append(1); return 403,limited,{}
assert guard['run_request'](exhausted,decode,gate,max_retries=2)[0]==403 and len(count)==3
assert guard['retry_delay'](0,{'Retry-After':'120'})>=120
assert guard['is_drive']('https://www.googleapis.com/upload/drive/v3/files')
assert not guard['is_drive']('https://oauth2.googleapis.com/token')

# Exercise both actual wrapper paths using fake transports; success streams unread.
class HTTP:
    def request(self,uri,method='GET',body=None,headers=None,*args,**kwargs):
        self.calls=getattr(self,'calls',0)+1
        return (types.SimpleNamespace(status=200),b'ok')
class Response:
    status_code=206; headers={}
    @property
    def content(self): raise AssertionError('Successful range stream was consumed')
    def close(self): raise AssertionError('Successful range stream was closed')
class Session:
    def request(self,method,url,*args,**kwargs): return Response()
names=['google_auth_httplib2','google.auth.transport.requests']
old={k:sys.modules.get(k) for k in names}
sys.modules[names[0]]=types.SimpleNamespace(AuthorizedHttp=HTTP)
sys.modules[names[1]]=types.SimpleNamespace(AuthorizedSession=Session)
try:
    guard['install'](interval=0)
    assert HTTP().request('https://www.googleapis.com/drive/v3/files')[1]==b'ok'
    assert Session().request('GET','https://www.googleapis.com/drive/v3/files/id').status_code==206
    installed=HTTP.request; guard['install'](interval=0); assert HTTP.request is installed
finally:
    for name in names:
        if old[name] is None: sys.modules.pop(name,None)
        else: sys.modules[name]=old[name]
print('PASS: pacing, rejected-upload retries with identical bodies, stream rewind, bounded exhaustion, permission/ambiguous-error propagation, Retry-After and untouched success streams.')
