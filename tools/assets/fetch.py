import sys
from scrapling.fetchers import Fetcher, StealthyFetcher
url=sys.argv[1]; mode=sys.argv[2] if len(sys.argv)>2 else 'f'
p = Fetcher.get(url) if mode=='f' else StealthyFetcher.fetch(url)
print("STATUS",p.status, "LEN", len(p.body))
txt=p.get_all_text() if hasattr(p,'get_all_text') else ''
print(txt[:int(sys.argv[3]) if len(sys.argv)>3 else 3000])
print("LINKS")
seen=set()
for a in p.css('a'):
    h=a.attrib.get('href')
    if h and h not in seen:
        seen.add(h); print(h, '|', (a.get_all_text() or '')[:60].strip())
