import sys,re
from scrapling.fetchers import Fetcher, StealthyFetcher, DynamicFetcher
url=sys.argv[1]; mode=sys.argv[2]; pat=sys.argv[3] if len(sys.argv)>3 else '.'
p={'f':Fetcher.get,'s':StealthyFetcher.fetch,'d':DynamicFetcher.fetch}[mode](url)
print("STATUS",p.status,"LEN",len(p.body))
seen=set()
for a in p.css('a'):
    h=a.attrib.get('href')
    if h and h not in seen and re.search(pat,h+(a.get_all_text() or ''),re.I):
        seen.add(h); print(h,'|',(a.get_all_text() or '')[:80].strip())
for a in p.css('img'):
    h=a.attrib.get('src') or a.attrib.get('data-src')
    if h and re.search(pat,h,re.I): print('IMG',h)
