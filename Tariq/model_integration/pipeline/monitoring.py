"""Bounded, atomic progress records for a single local pipeline attempt."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import re
import threading
import time
import uuid

STEPS = [('alignment', 'Check target and evaluation alignment'),
         ('files', 'Check model packages and hashes'), ('inputs', 'Validate example inputs'),
         ('direction', 'Run direction model'), ('magnitude', 'Run magnitude model'),
         ('combine', 'Calculate signed score'), ('save', 'Save result')]


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def error_text(error):
    # Subprocess exception strings may contain commands; do not persist those.
    if type(error).__name__ in ('CalledProcessError', 'TimeoutExpired'):
        return type(error).__name__ + ': child process failed or exceeded its time limit'
    text = str(error) or type(error).__name__
    return re.sub(r'(?i)(password|token|secret|api_key)\s*[:=]\s*\S+', r'\1=[REDACTED]', text)[:1200]


class Monitor:
    def __init__(self, folder, config, mode):
        self.path = folder / 'monitor.json'
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.record = {'attempt_id': uuid.uuid4().hex, 'mode': mode, 'status': 'RUNNING',
                       'started_at_utc': utc(), 'heartbeat_at_utc': utc(), 'finished_at_utc': None,
                       'models': {r: m['version'] for r, m in config['models'].items()},
                       'steps': [{'id': key, 'name': name, 'status': 'WAITING', 'duration_seconds': None,
                                  'started_at_utc': None, 'finished_at_utc': None, 'detail': ''} for key, name in STEPS]}
        self.flush()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def flush(self):
        with self.lock:
            self.record['heartbeat_at_utc'] = utc()
            atomic_json(self.path, self.record)

    def heartbeat(self):
        while not self.stop.wait(5):
            try:
                self.flush()
            except OSError:
                # An old heartbeat lets the reader flag missing progress if storage fails.
                return

    @contextmanager
    def step(self, key):
        item = next(s for s in self.record['steps'] if s['id'] == key)
        began = time.monotonic()
        with self.lock:
            item.update(status='RUNNING', started_at_utc=utc())
            self.flush()
        try:
            yield
        except BaseException as error:
            with self.lock:
                item.update(status='FAILED', detail=error_text(error))
            raise
        else:
            with self.lock:
                item['status'] = 'PASSED'
        finally:
            with self.lock:
                item.update(finished_at_utc=utc(), duration_seconds=round(time.monotonic()-began, 4))
                self.flush()

    def skip(self, key, detail):
        with self.lock:
            next(s for s in self.record['steps'] if s['id']==key).update(status='SKIPPED', detail=detail)
            self.flush()

    def finish(self, status):
        self.stop.set()
        self.thread.join(timeout=6)
        with self.lock:
            self.record.update(status=status, finished_at_utc=utc(),
                               duration_seconds=round(time.monotonic()-self.started,4))
            for item in self.record['steps']:
                if item['status'] == 'WAITING':
                    item.update(status='SKIPPED', detail='Not reached because an earlier step failed or was interrupted')
            self.flush()
            history_path = self.path.parent.parent / 'monitor_history.json'
            history = json.loads(history_path.read_text('utf-8')) if history_path.exists() else []
            history.append({k:self.record[k] for k in ('attempt_id','mode','status','started_at_utc','finished_at_utc','duration_seconds','models')}
                           | {'failed_steps':[s for s in self.record['steps'] if s['status']=='FAILED']})
            atomic_json(history_path, history[-100:])
