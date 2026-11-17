"""Run in Google Colab. Reads bounded ZIP ranges; never downloads whole archives.
Copies small audit documents into a dedicated Drive folder, leaving originals intact.
"""
import io
import json
import hashlib
import zipfile
from pathlib import PurePosixPath

ARCHIVES = {
    'canonical': '1y351d1ibxpAH-MxijvGDHPWEfqOJ70aM',
    'features': '1RXZDR_PaLkKMYtuswOBnl4_0eG9_LkNx',
}
MAX_TRANSFER = 96 * 1024**2
MAX_DOCUMENT = 2 * 1024**2
MAX_OUTPUT = 8 * 1024**2


class DriveRangeReader(io.RawIOBase):
    def __init__(self, session, file_id, size):
        self.session, self.file_id, self.size = session, file_id, size
        self.pos = self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if whence not in (0, 1, 2) or target < 0:
            raise ValueError('Invalid seek')
        self.pos = target
        return target

    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0: return b''
        if n > 32*1024**2 or self.transferred+n > MAX_TRANSFER:
            raise RuntimeError('Remote read budget reached; no full download attempted.')
        start, end = self.pos, self.pos+n-1
        url = f'https://www.googleapis.com/drive/v3/files/{self.file_id}'
        with self.session.get(url, params={'alt':'media'},
                              headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=90) as response:
            if response.status_code != 206:
                raise RuntimeError(f'Expected partial response 206, got {response.status_code}; stopped without reading full body.')
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise RuntimeError('Unexpected range response')
            data = response.raw.read(n+1)
            if len(data) != n: raise RuntimeError('Incomplete or oversized range')
        self.pos += n
        self.transferred += n
        return data


def run():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    session = AuthorizedSession(credentials)
    drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    bundle = io.BytesIO()
    inventory, receipts = [], []
    total = 0
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as output:
        for tag, file_id in ARCHIVES.items():
            meta = drive.files().get(fileId=file_id, fields='id,name,size,md5Checksum,modifiedTime').execute()
            print('Inspecting:', meta['name'], flush=True)
            remote = DriveRangeReader(session, file_id, int(meta['size']))
            with zipfile.ZipFile(remote) as archive:
                infos = archive.infolist()
                candidates = []
                for info in infos:
                    path = PurePosixPath(info.filename)
                    lower = info.filename.lower()
                    suffix = path.suffix.lower()
                    # Metadata/code only; never read market data, model weights, nested ZIPs or outcome tables.
                    eligible = (suffix in {'.md','.txt','.json','.yaml','.yml','.py','.html'}
                        and any(word in lower for word in ['readme','read_me','manifest','contract','formula',
                            'binding','availability','timing','normaliz','state_rule','label','build_y','price_verification','source_day_receipt',
                            'team_tools/','start_here','feature_schema','feature_dictionary','f4z_status','qa_report','handoff_2_qa','cuts_historical']))
                    safe = not path.is_absolute() and '..' not in path.parts
                    if eligible and safe and not info.is_dir():
                        inventory.append({'archive':tag,'path':info.filename,'bytes':info.file_size,
                                          'compressed_bytes':info.compress_size,'eligible_size':info.file_size<=MAX_DOCUMENT})
                        if info.file_size<=MAX_DOCUMENT and info.compress_size<=MAX_DOCUMENT:
                            candidates.append(info)
                candidates.sort(key=lambda x:(0 if any(w in x.filename.lower() for w in ['readme','read_me','contract','formula','build_y']) else 1,x.file_size,x.filename))
                selected = []
                for info in candidates:
                    if len(selected)>=60 or total+info.file_size>MAX_OUTPUT: continue
                    data = archive.read(info)  # zipfile validates CRC; extracted code is never executed.
                    total += len(data)
                    output.writestr(f'{tag}/{info.filename}',data)
                    selected.append({'path':info.filename,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
                receipts.append({'source':meta,'archive_tag':tag,'entries':len(infos),
                                 'remote_bytes_read':remote.transferred,'selected':selected})
                print(f'  Read {remote.transferred/1024**2:.2f} MiB; copied {len(selected)} audit documents.',flush=True)
        output.writestr('audit_inventory.json',json.dumps(inventory,indent=2))
        output.writestr('audit_receipt.json',json.dumps(receipts,indent=2))
    # Resume by reusing a uniquely named destination; never touch source ZIPs or sharing permissions.
    folder_name = 'ScaleWave_Archive_Audit'
    matches = drive.files().list(q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",fields='files(id)').execute()['files']
    if len(matches)>1: raise RuntimeError('Multiple audit folders found; resolve duplicate destinations first.')
    folder = matches[0]['id'] if matches else drive.files().create(body={'name':folder_name,'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
    name = 'ScaleWave_Audit_Documents.zip'
    existing = drive.files().list(q=f"'{folder}' in parents and name='{name}' and trashed=false",fields='files(id)').execute()['files']
    if len(existing)>1: raise RuntimeError('Multiple audit bundles found; refusing ambiguous overwrite.')
    bundle.seek(0)
    media = MediaIoBaseUpload(bundle,mimetype='application/zip',resumable=True)
    if existing:
        drive.files().update(fileId=existing[0]['id'],media_body=media,fields='id').execute()
    else:
        drive.files().create(body={'name':name,'parents':[folder]},media_body=media,fields='id').execute()
    print('DONE. Send this folder link to Codex:')
    print(f'https://drive.google.com/drive/folders/{folder}')
    print('Original archives unchanged. No market datasets read. No files downloaded to your PC.')


if __name__ == '__main__':
    run()
