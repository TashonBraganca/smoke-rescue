import sys,re
from scrapling.fetchers import DynamicFetcher
url=sys.argv[1]
p=DynamicFetcher.fetch(url, real_chrome=True, headless=True, network_idle=True, wait=3000)
print("STATUS",p.status,len(p.body))
t=p.get_all_text()
a=t.find('Download')
print(t[max(0,a-300):a+900])
for a in p.css('a'):
    h=a.attrib.get('href') or ''
    if re.search(r'download|\.zip|\.png|\.exr|r2\.|cdn|storage|s3|blob',h,re.I): print('A',h,'|',a.get_all_text()[:50])
for b in p.css('button'): print('B',b.get_all_text()[:80])
for i in p.css('img'): print('IMG',(i.attrib.get('src') or '')[:200])
