import re,subprocess,os,sys
from scrapling.fetchers import Fetcher
BASE='/Volumes/Tashon/opencv/smoke-rescue/assets_cache/vfx_candidates'
items={'jangafx_SmallCampfireVDB':'https://www.mediafire.com/file/9xx9icyomy2txv7/SmallCampfireVDB.zip/file',
       'jangafx_Fire_01_vdb':'https://www.mediafire.com/file/j9yvua10nfvhkvu/Fire_01.rar/file'}
for name,u in items.items():
    p=Fetcher.get(u)
    link=None
    for a in p.css('a'):
        h=a.attrib.get('href') or ''
        if re.search(r'download\d+\.mediafire',h): link=h;break
    d=f'{BASE}/{name}'; os.makedirs(d,exist_ok=True)
    out=f"{d}/{u.split('/')[-2]}"
    r=subprocess.run(['curl','-sfL','-m','900','-o',out,link]); print(name,r.returncode,os.path.getsize(out))
