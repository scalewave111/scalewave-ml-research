"""Narrow, recorded compatibility patch for Drive modifiedTime-only changes.

Original runner sources and checkpoint identities stay immutable. Only their two
archive-metadata assertions are replaced; executable patch hashes are recorded.
"""
import hashlib
import json

APPROVED = {
    '15m':'38d9944751c3ad2815a39024b40e3427d113a3fd7ada1acf521d72fdcc2f9777',
    '1h':'017f7326c9c5a43b71a3aede7b2a10fe2428c08efdd7d18aa1fa86f7241202eb',
}
OLD = "assert service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()==config['audit_sources'][k]"
NEW = "assert SOURCE_METADATA_CHECK(service,config,k,file_id)"


def compare(actual, expected):
    assert set(actual)==set(expected), 'Source metadata fields changed'
    assert actual.get('md5Checksum') and expected.get('md5Checksum'), 'Source checksum missing'
    different=[key for key in expected if actual[key]!=expected[key]]
    assert set(different)<= {'modifiedTime'}, 'Source identity/content changed: '+', '.join(different)
    return different


def compile_runner(source, namespace, horizon, patch_hash):
    original_hash=hashlib.sha256(source.encode()).hexdigest()
    assert original_hash==APPROVED[horizon], 'Runner not approved for metadata-only compatibility'
    assert source.count(OLD)==2, 'Expected exactly the startup and final source checks'
    patched=source.replace(OLD,NEW)
    evidence={'policy':'Accept modifiedTime-only difference; all other returned fields, including size and MD5, must match.',
              'original_runner_sha256':original_hash,
              'executed_runner_sha256':hashlib.sha256(patched.encode()).hexdigest(),
              'compat_module_sha256':patch_hash,'horizon':horizon,'sources':{}}
    def verify(service,config,key,file_id):
        actual=service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()
        expected=config['audit_sources'][key]
        differences=compare(actual,expected)
        evidence['sources'][key]={'expected':expected,'observed':actual,'accepted_differences':differences,
                                  'id_name_size_md5_match':True}
        # Separate audit receipt; never rewrite a source archive, frozen contract,
        # old model receipt, or cache identity to conceal its original metadata.
        store=namespace['REF']['DriveStore'](service,config['folder'])
        store.json(f'SOURCE_METADATA_COMPAT_{horizon}.json',evidence)
        if differences:
            print(f'{key}: Drive modification time changed; file ID, name, size and MD5 still match. Recorded metadata-only compatibility.',flush=True)
        return True
    namespace['SOURCE_METADATA_CHECK']=verify
    exec(compile(patched,f'screen_{horizon}_runner.py','exec'),namespace)
    return evidence
