import sys,re
from scrapling.fetchers import Fetcher
for u in sys.argv[1:]:
    p=Fetcher.get(u)
    for a in p.css('a'):
        h=a.attrib.get('href') or ''
        if 'download' in h and 'mediafire.com/' in h and 'download_repair' not in h and 'customer' not in h and re.search(r'download\d+\.mediafire',h):
            print(h,'|',a.get_all_text().strip()); break
    else: print("NOLINK",u)
