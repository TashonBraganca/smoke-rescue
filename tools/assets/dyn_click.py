import sys,re
from scrapling.fetchers import DynamicFetcher
url=sys.argv[1]
log=[]
def act(page):
    page.on('response', lambda r: log.append((r.status, r.url, r.headers.get('content-type'))))
    page.on('request', lambda r: log.append(('REQ', r.method, r.url)))
    try:
        page.get_by_role('button', name=re.compile('Download Asset')).first.click(timeout=5000)
    except Exception as e: log.append(('clickfail',str(e)[:200]))
    page.wait_for_timeout(5000)
    log.append(('BODY', page.inner_text('body')[:1500]))
    return page
p=DynamicFetcher.fetch(url, real_chrome=True, headless=True, network_idle=True, page_action=act)
for l in log:
    s=str(l)
    if 'BODY' in s or 'clickfail' in s or re.search(r'api|download|supabase|r2|zip|signed|cdn|storage',s,re.I): print(s[:600])
