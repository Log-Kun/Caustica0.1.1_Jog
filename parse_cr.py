import json
d = json.load(open('cr.json', encoding='utf-8'))
print('total:', d.get('total_count'))
for cr in d.get('check_runs', []):
    print(cr['name'], cr['conclusion'], '| ann:', cr['output'].get('annotations_count'))
    for a in cr['output'].get('annotations', []):
        print('   ', a.get('path'), a.get('start_line'), a.get('message', '')[:800])