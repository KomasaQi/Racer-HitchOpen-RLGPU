"""Collect frozen overnight evidence without ranking or promoting policies."""
from pathlib import Path
import argparse,json,csv,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
rows=[]
for f in sorted((root/'evaluations').glob('*/ep*/*/*/summary.json')):
 try:
  d=json.loads(f.read_text());rel=f.relative_to(root/'evaluations');row=dict(case=rel.parts[0],epoch=rel.parts[1],suite=rel.parts[2],fixture=rel.parts[3],file=str(f))
  for k,v in d.items():
   if v is None or isinstance(v,(str,int,float,bool)):row[k]=v
  rows.append(row)
 except Exception as exc:rows.append(dict(file=str(f),collection_error=str(exc)))
out=root/'collected';out.mkdir(exist_ok=True)
(out/'all_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
fields=sorted({k for r in rows for k in r})
with (out/'all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['runs','evaluations','logs'] for f in (root/folder).rglob('*') if f.is_file() and (f.suffix in ['.json','.pth'] or f.name=='worker.log')}
(out/'artifacts_sha256.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(collected_summaries=len(rows),hashed_artifacts=len(files),output=str(out))))
