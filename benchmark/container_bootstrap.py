"""Load an exact private benchmark bundle, then serve TLS. AGPL-3.0-or-later."""
import hashlib
import os
from pathlib import Path
import runpy
import sys
import tempfile
import urllib.request
import zipfile


def main():
    with tempfile.TemporaryDirectory(prefix='homr-benchmark-') as folder:
        root = Path(folder)
        archive = root / 'bundle.zip'
        with urllib.request.urlopen(os.environ['HOMR_BUNDLE_URL'], timeout=90) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != os.environ['HOMR_BUNDLE_SHA256']:
            raise ValueError('Benchmark bundle identity mismatch')
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe bundle path')
            source.extractall(root)
        archive.unlink()
        adapter = os.environ['HOMR_CONTAINER_SERVER'].encode()
        if hashlib.sha256(adapter).hexdigest() != os.environ['HOMR_CONTAINER_SERVER_SHA256']:
            raise ValueError('Container adapter identity mismatch')
        (root / 'container_server.py').write_bytes(adapter)
        sys.path[:0] = [str(root / '.python_packages/lib/site-packages'), str(root)]
        os.chdir(root)
        runpy.run_path(str(root / 'container_server.py'), run_name='__main__')


if __name__ == '__main__':
    main()
