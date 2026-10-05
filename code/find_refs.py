import sys, json, time, urllib.parse, urllib.request
Q = sys.argv[1:]
for q in Q:
    url = 'https://api.crossref.org/works?rows=8&select=title,author,issued,volume,issue,page,container-title,DOI,type,is-referenced-by-count&filter=from-pub-date:2023-01-01,type:journal-article&query.bibliographic=' + urllib.parse.quote(q)
    try: d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'refcheck/1.0 (mailto:refcheck@example.org)'}), timeout=40))['message']['items']
    except Exception as e: print('ERR', q, e); continue
    print('\n## ' + q)
    for c in d:
        a = c.get('author') or []; f = (a[0].get('family', '') + ' ' + (a[0].get('given', '') or '')[:1]) if a else ''
        print(f"- {(c.get('title') or [''])[0][:110]} | {f} et al ({len(a)} au) | {((c.get('issued') or {}).get('date-parts') or [['?']])[0][0]} | {(c.get('container-title') or [''])[0][:45]} {c.get('volume','')}({c.get('issue','')}) {c.get('page','')} | {c.get('DOI')} | cited {c.get('is-referenced-by-count')}")
    time.sleep(1.5)
