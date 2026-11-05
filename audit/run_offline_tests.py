"""Run only the reviewed synthetic suite. Python guards are not an OS sandbox."""
from pathlib import Path
import datetime
import importlib.metadata
import json
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.audit-work' / uuid.uuid4().hex
WORK.mkdir(parents=True)
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = str(WORK)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest

READ_ROOTS = [ROOT, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
if os.environ.get('SystemRoot'):
    READ_ROOTS.append(Path(os.environ['SystemRoot']).resolve())

def within(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)

def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind',
                 'subprocess.Popen', 'os.system', 'os.startfile'}:
        raise RuntimeError('OFFLINE_AUDIT_BLOCKED_NETWORK_OR_PROCESS')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        if os.fsdecode(args[0]).lower() == os.devnull.lower():
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ''
        flags = args[2] or 0
        writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and not within(path, [WORK]):
            raise RuntimeError('OFFLINE_AUDIT_WRITE_OUTSIDE_SCRATCH')
        if not writing and not within(path, READ_ROOTS):
            raise RuntimeError('OFFLINE_AUDIT_READ_OUTSIDE_SNAPSHOT_OR_RUNTIME')
    if event == 'sqlite3.connect' and args[0] != ':memory:':
        if not within(Path(args[0]).resolve(), [WORK]):
            raise RuntimeError('OFFLINE_AUDIT_DATABASE_OUTSIDE_SCRATCH')

sys.addaudithook(guard)
started = time.perf_counter()
xml = WORK / 'junit.xml'
code = pytest.main(['tests', '-q', '-p', 'no:cacheprovider',
                    '--rootdir', str(ROOT), '--confcutdir', str(ROOT),
                    '--log-file', str(WORK / 'pytest.log'),
                    '--basetemp', str(WORK / 'pytest'), '--junitxml', str(xml)])
counts = dict(tests=0, failures=0, errors=0, skipped=0)
if xml.exists():
    for suite in ET.parse(xml).getroot().iter('testsuite'):
        for field in counts:
            counts[field] += int(suite.get(field, 0))
versions = {}
for name in ['numpy', 'pandas', 'pyarrow', 'pytest', 'fastapi', 'httpx', 'lightgbm', 'scipy']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = 'not installed'
result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              python=sys.version.split()[0], packages=versions, exit_code=int(code),
              elapsed_seconds=round(time.perf_counter()-started, 3), **counts,
              scope='Reviewed synthetic tests only; no live APIs, original datasets or historical acceptance suite',
              safeguards='Python audit hooks deny networking, subprocesses, writes outside scratch, and reads outside snapshot/runtime; not an OS sandbox')
(WORK / 'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
print(json.dumps(result, indent=2))
raise SystemExit(int(code))
