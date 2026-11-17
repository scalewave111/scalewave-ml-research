"""Session-local Drive transport pacing; no model/config/checkpoint changes."""
import json
import random
import threading
import time
from urllib.parse import urlsplit
from email.utils import parsedate_to_datetime


def is_drive(url):
    parsed=urlsplit(str(url))
    return parsed.hostname=='drive.googleapis.com' or (
        parsed.hostname=='www.googleapis.com' and parsed.path.startswith(('/drive/','/upload/drive/')))


def limited(status, content):
    if int(status)==429: return True
    if int(status)!=403: return False
    try:
        value=json.loads(content)
        reasons={e.get('reason') for e in value.get('error',{}).get('errors',[])}
        return bool(reasons & {'rateLimitExceeded','userRateLimitExceeded'})
    except (ValueError,TypeError,AttributeError):
        return False


class Gate:
    def __init__(self, interval=3.0, clock=time.monotonic, sleep=time.sleep):
        self.interval,self.clock,self.sleep=interval,clock,sleep
        self.next_allowed=0.0; self.lock=threading.Lock()
    def take(self):
        while True:
            with self.lock:
                now=self.clock(); wait=self.next_allowed-now
                if wait<=0:
                    self.next_allowed=now+self.interval
                    return
            self.sleep(min(wait,60))
    def defer(self, seconds):
        with self.lock: self.next_allowed=max(self.next_allowed,self.clock()+seconds)


def retry_delay(attempt, headers):
    delay=min(2**(attempt+1)+random.random(),60)
    value=headers.get('retry-after') or headers.get('Retry-After')
    if value:
        try: delay=max(delay,float(value))
        except (ValueError,TypeError):
            try: delay=max(delay,parsedate_to_datetime(value).timestamp()-time.time())
            except (ValueError,TypeError,OverflowError): pass
    return max(0,delay)


def run_request(send, decode, gate, body=None, max_retries=8):
    """Retry explicit rejection only. Ambiguous POST failures are never replayed."""
    offset=None
    if hasattr(body,'read'):
        try: offset=body.tell()
        except (AttributeError,OSError): pass
    for attempt in range(max_retries+1):
        if attempt and hasattr(body,'read'):
            if offset is None: raise RuntimeError('Cannot safely replay a streaming Drive request')
            body.seek(offset)
        gate.take()
        response=send()  # Exceptions and ambiguous network failures propagate unchanged.
        status,content,headers,close=decode(response)
        if not limited(status,content) or attempt==max_retries:
            if limited(status,content): print('Drive rate limit persisted after bounded retries; saved checkpoints remain reusable.',flush=True)
            return response
        delay=retry_delay(attempt,headers)
        close()
        gate.defer(delay)
        print(f'Drive rate limit: retry {attempt+1}/{max_retries}, waiting at least {delay:.1f}s. Fit settings unchanged.',flush=True)


def install(interval=3.0):
    import google_auth_httplib2
    from google.auth.transport.requests import AuthorizedSession
    marker='_scalewave_drive_rate_guard_v1'
    cls=google_auth_httplib2.AuthorizedHttp
    if getattr(cls,marker,False):
        print('Drive request guard already installed in this runtime.',flush=True)
        return
    gate=Gate(interval)
    original_http=cls.request
    def http_request(self, uri, method='GET', body=None, headers=None, *args, **kwargs):
        send=lambda:original_http(self,uri,method,body,headers,*args,**kwargs)
        if not is_drive(uri): return send()
        def decode(response):
            metadata,content=response
            return int(metadata.status),content,metadata,lambda:None
        return run_request(send,decode,gate,body=body)
    original_session=AuthorizedSession.request
    def session_request(self, method, url, *args, **kwargs):
        send=lambda:original_session(self,method,url,*args,**kwargs)
        if not is_drive(url): return send()
        def decode(response):
            # Do not consume successful streaming ZIP range responses.
            content=response.content if response.status_code in (403,429) else b''
            return response.status_code,content,response.headers,response.close
        return run_request(send,decode,gate,body=kwargs.get('data'))
    cls.request=http_request
    AuthorizedSession.request=session_request
    setattr(cls,marker,True)
    print(f'Drive I/O guard active: requests spaced by {interval:g}s per runtime; rate-limit retries enabled.',flush=True)
