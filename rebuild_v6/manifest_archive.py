"""Read the SHA-bound split archive; splitting changes transport, not ZIP bytes."""
import hashlib,io,json,zipfile
from pathlib import Path

def open_manifest_archive(data_dir=None):
    directory=Path(data_dir) if data_dir else Path(__file__).resolve().parents[1]/'data'
    index=json.loads((directory/'MANIFEST_ARCHIVE.json').read_text())
    chunks=[]
    for part in index['parts']:
        name=part['path']
        if Path(name).name!=name:raise ValueError('Unsafe archive part path')
        b=(directory/name).read_bytes()
        if len(b)!=part['bytes'] or hashlib.sha256(b).hexdigest()!=part['sha256']:
            raise ValueError('Archive part missing or corrupted: '+name)
        chunks.append(b)
    full=b''.join(chunks)
    if len(full)!=index['bytes'] or hashlib.sha256(full).hexdigest()!=index['sha256']:
        raise ValueError('Reassembled archive hash mismatch')
    return zipfile.ZipFile(io.BytesIO(full))
