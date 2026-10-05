import sys, json, time, urllib.parse, urllib.request
for q in sys.argv[1:]:
    url = 'https://api.crossref.org/works?rows=3&select=title,author,issued,volume,issue,page,container-title,DOI,type,article-number&query.bibliographic=' + urllib.parse.quote(q)
    try: d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'refcheck/1.0 (mailto:refcheck@example.org)'}), timeout=40))['message']['items']
    except Exception as e: print('ERR', q, e); continue
    print('\n## ' + q)
    for c in d:
        au = '; '.join(f"{a.get('family','')}, {a.get('given','')}" for a in (c.get('author') or [])[:12])
        print(f"- {(c.get('title') or [''])[0][:120]} | {au} | {((c.get('issued') or {}).get('date-parts') or [['?']])[0][0]} | {(c.get('container-title') or [''])[0][:70]} vol {c.get('volume','')} iss {c.get('issue','')} pp {c.get('page','')} art {c.get('article-number','')} | {c.get('DOI')}")
    time.sleep(1.2)
