# Static files from S3 (files over 5 MB redirect to a 1 h presigned HTTPS URL: Lambda responses cap at 6 MB),
# and three public OpenCV routes proxied to the ALB. Nothing else reaches the agent service.
# /api/mission* is the "Run your own rescue" API in ryor_api.py (ECS Fargate tasks, runs bucket, DynamoDB).
import base64, mimetypes, os, re, urllib.request, urllib.error
import boto3
s3 = boto3.client('s3')
B, ALB = os.environ['BUCKET'], os.environ['ALB'].rstrip('/')
API = {('GET', '/api/health'), ('POST', '/api/perceive/people')}
VISION = re.compile(r'/api/vision/[a-z_]{1,40}\Z')
SEC = {'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer'}
def out(code, body=b'', ctype='text/plain', extra=None):
    h = {'Content-Type': ctype, **SEC, **(extra or {})}
    return {'statusCode': code, 'headers': h, 'isBase64Encoded': True, 'body': base64.b64encode(body).decode()}
def handler(event, context):
    if 'requestContext' not in event:  # EventBridge schedule: stop ryor tasks past the hard limit
        import ryor_api
        return ryor_api.sweep()
    method = event['requestContext']['http']['method']
    path = event.get('rawPath') or '/'
    if path == '/api/mission' or path.startswith('/api/mission/'):  # "Run your own rescue", tools/ryor/CONTRACT.txt
        import ryor_api
        code, data = ryor_api.route(method, path, event)
        return out(code, data, 'application/json', {'Cache-Control': 'no-store'})
    if path.startswith('/api/'):
        if (method, path) not in API and not (method == 'POST' and VISION.match(path)):
            return out(404, b'not a public route')
        body = event.get('body') or ''
        body = base64.b64decode(body) if event.get('isBase64Encoded') else body.encode()
        hdr = {k: v for k, v in event.get('headers', {}).items() if k.lower() == 'content-type'}
        req = urllib.request.Request(ALB + path[4:], data=body if method == 'POST' else None, method=method, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=65) as r:
                return out(r.status, r.read(), r.headers.get('Content-Type', 'application/json'))
        except urllib.error.HTTPError as e:
            return out(e.code, e.read(), e.headers.get('Content-Type', 'application/json'))
        except Exception as e:
            print('ALB error', type(e).__name__, e)
            return out(502, b'{"detail": "OpenCV service unreachable"}', 'application/json')
    if method not in ('GET', 'HEAD'):
        return out(405, b'method not allowed')
    key = path.lstrip('/') or 'index.html'
    if key.endswith('/'):
        key += 'index.html'
    if '..' in key.split('/') or '\\' in key:
        return out(400, b'bad path')
    try:
        head = s3.head_object(Bucket=B, Key=key)
    except Exception:
        return out(404, b'not found')
    ctype = mimetypes.guess_type(key)[0] or 'application/octet-stream'
    if head['ContentLength'] > 5 * 1024 * 1024:
        url = s3.generate_presigned_url('get_object', Params={'Bucket': B, 'Key': key}, ExpiresIn=3600)
        return out(302, b'', extra={'Location': url, 'Cache-Control': 'no-store'})
    data = s3.get_object(Bucket=B, Key=key)['Body'].read()
    cache = 'no-store' if key.endswith('.json') or key.endswith('.jsonl') else 'public, max-age=300'
    return out(200, data if method == 'GET' else b'', ctype, {'Cache-Control': cache})
