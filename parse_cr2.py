import json
d = json.load(open('cr.json', encoding='utf-8'))
for cr in d.get('check_runs', []):
    print('NAME', cr['name'], 'ID', cr['id'], 'ann_count', cr['output'].get('annotations_count'))
    out = cr['output']
    print('  title:', out.get('title'))
    print('  summary:', (out.get('summary') or '')[:800])
    print('  text:', (out.get('text') or '')[:800])