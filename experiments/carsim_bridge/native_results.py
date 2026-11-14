"""Publish preserved native solver outputs to the source run's GUI LastRun."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil


def publish_last_run(simfile, native_directory):
    source = Path(native_directory).resolve()
    text = Path(simfile).read_text(encoding='utf-8')
    macros = dict(re.findall(r'^SET_MACRO\s+\$\(([^)]+)\)\$\s+(.+)$', text, re.M))
    match = re.search(r'^FILEBASE\s+(.+)$', text, re.M)
    if not match:
        raise ValueError('Source simfile has no FILEBASE')
    prefix = match.group(1).strip()
    for _ in range(10):
        expanded = re.sub(r'\$\(([^)]+)\)\$', lambda m: macros.get(m[1], m[0]).strip(), prefix)
        if expanded == prefix:
            break
        prefix = expanded
    target = Path(prefix)
    if '$(' in prefix or not target.is_absolute() or target.name != 'LastRun' or not target.parent.is_dir():
        raise ValueError('Refuse ambiguous or non-LastRun destination: ' + prefix)
    binary = source / 'run.vsb'
    if not binary.exists() or binary.stat().st_size <= 24:
        raise ValueError('No populated native time-series binary to publish')
    # Preserve old versions beside this immutable evaluation, before replacing.
    backup = source / ('previous_LastRun_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    files = sorted(p for p in source.iterdir() if p.is_file() and
                   (p.name.startswith('run.') or p.name.startswith('run_')))
    pairs = [(p, target.parent / ('LastRun' + p.name[3:])) for p in files]
    for _, dest in pairs:
        if dest.exists():
            shutil.copy2(dest, backup / dest.name)
    # Metadata last: readers should not observe new metadata before its binary.
    pairs.sort(key=lambda pair: pair[0].suffix == '.vs')
    for src, dest in pairs:
        if src.suffix == '.vs':
            data = json.loads(src.read_text(encoding='utf-8-sig'))
            data['VsChannelGroup']['Input File'] = str(dest)
            dest.write_text(json.dumps(data, indent=2), encoding='utf-8')
        else:
            shutil.copy2(src, dest)
    src_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    dest_binary = target.with_suffix('.vsb')
    if hashlib.sha256(dest_binary.read_bytes()).hexdigest() != src_hash:
        raise RuntimeError('Published native binary differs from preserved source')
    manifest = dict(source=str(source),last_run=str(target),backup=str(backup),
                    binary_bytes=binary.stat().st_size,binary_sha256=src_hash,
                    files=[str(dest) for _,dest in pairs])
    (source/'last_run_publication.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
