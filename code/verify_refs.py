"""Verify every entry of refs.bib against DBLP and Crossref (title similarity, first author, year, volume/pages, venue)."""
import re, sys, json, time, difflib, urllib.parse, urllib.request
BIB = '/home/user/Diabetic-Retinopathy/paper/refs.bib'
def parse(path):
    txt = open(path).read(); out = []
    for chunk in re.split(r'\n(?=@)', txt):
        m = re.match(r'@(\w+)\{([^,]+),', chunk.strip())
        if not m: continue
        f = {}
        for k, v in re.findall(r'(\w+)\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\d+)', chunk): f[k.lower()] = v[1:-1] if v.startswith('{') else v
        f['key'] = m.group(2); f['type'] = m.group(1); out.append(f)
    return out
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', re.sub(r'\\[a-z]+|[{}$\\]', '', s.lower().replace('-', ' ').replace('&', ' and ')))
def get(url, tries=3):
    for t in range(tries):
        try:
            r = urllib.request.Request(url, headers={'User-Agent': 'refcheck/1.0 (mailto:refcheck@example.org)'})
            return json.load(urllib.request.urlopen(r, timeout=30))
        except Exception as e:
            time.sleep(3 * (t + 1)); err = e
    return None
def arxiv(title):
    import xml.etree.ElementTree as ET
    q = urllib.parse.quote('ti:"' + re.sub(r'[{}\\$]', '', title)[:120] + '"')
    try:
        r = urllib.request.Request('https://export.arxiv.org/api/query?max_results=3&search_query=' + q, headers={'User-Agent': 'refcheck/1.0'})
        root = ET.fromstring(urllib.request.urlopen(r, timeout=30).read())
    except Exception: return []
    ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}; out = []
    for e in root.findall('a:entry', ns):
        out.append(dict(title=' '.join((e.findtext('a:title', '', ns) or '').split()), year=(e.findtext('a:published', '', ns) or '')[:4], authors=[a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)],
                        jref=e.findtext('x:journal_ref', '', ns), id=e.findtext('a:id', '', ns)))
    return out
def crossref(title, author):
    d = get('https://api.crossref.org/works?rows=5&select=title,author,issued,volume,issue,page,container-title,DOI,type&query.bibliographic=' + urllib.parse.quote(re.sub(r'[{}\\$]', '', title) + ' ' + author))
    return ((d or {}).get('message') or {}).get('items') or []
res = []
for e in parse(BIB):
    t = e.get('title', e.get('howpublished', '')); au = e.get('author', '').split(' and ')[0]; fam = norm(au.split(',')[0]) if ',' in au else norm(au.split()[-1] if au else '')
    best = None
    for c in crossref(t, au):
        ct = (c.get('title') or [''])[0]; r = difflib.SequenceMatcher(None, norm(t), norm(ct)).ratio()
        if best is None or r > best[0]: best = (r, 'crossref', c)
    time.sleep(0.6)
    if best is None or best[0] < 0.9:
        for h in arxiv(t):
            r = difflib.SequenceMatcher(None, norm(t), norm(h['title'])).ratio()
            if best is None or r > best[0]: best = (r, 'arxiv', h)
        time.sleep(3)
    row = dict(key=e['key'], bib_title=t, bib_year=e.get('year'), bib_vol=e.get('volume'), bib_pages=e.get('pages'), bib_first=fam, bib_journal=e.get('journal') or e.get('booktitle') or e.get('publisher'))
    if best:
        r, src, h = best
        if src == 'arxiv':
            f0 = h['authors'][0] if h['authors'] else ''
            row.update(src='arxiv', ratio=round(r, 3), found_title=h['title'], found_year=h['year'], found_vol=None, found_pages=None, found_first=norm(f0.split()[-1]) if f0 else '', found_venue=h['jref'] or h['id'], doi=None)
        else:
            a = h.get('author') or []; f0 = a[0].get('family', '') if a else ''
            row.update(src='crossref', ratio=round(r, 3), found_title=(h.get('title') or [''])[0], found_year=((h.get('issued') or {}).get('date-parts') or [[None]])[0][0], found_vol=h.get('volume'), found_pages=h.get('page'), found_first=norm(f0), found_venue=(h.get('container-title') or [''])[0], doi=h.get('DOI'))
    res.append(row); print(e['key'], row.get('src'), row.get('ratio'), flush=True)
json.dump(res, open('/home/user/work/refcheck.json', 'w'), indent=1)
