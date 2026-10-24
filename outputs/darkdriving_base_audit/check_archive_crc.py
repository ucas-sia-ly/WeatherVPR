import json,zlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
out=Path('outputs/darkdriving_base_audit'); report=json.loads((out/'archive_inventory.json').read_text()); root=Path('data/DarkDriving_Base'); start=time.monotonic()
def check(row):
 p=root/row['path']; crc=0
 try:
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''): crc=zlib.crc32(block,crc)
  actual=f'{crc:08x}'
  return None if actual==row['archive_crc32'] else {'path':row['path'],'expected':row['archive_crc32'],'actual':actual}
 except Exception as exc: return {'path':row['path'],'error':str(exc)}
errors=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for i,result in enumerate(pool.map(check,report['inventory']),1):
  if result: errors.append(result)
  if i%3000==0: print(f'{i}/{len(report["inventory"])} CRC checks; errors={len(errors)}',flush=True)
report['crc_comparison']={'checked_files':len(report['inventory']),'mismatches':errors,'seconds':time.monotonic()-start}
(out/'archive_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report['crc_comparison']),flush=True)
