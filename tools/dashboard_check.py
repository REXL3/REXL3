#!/usr/bin/env python3
"""AGPL-3.0-only. Check exactly the authored browser panel series against REAL HTTP metrics."""
import argparse
import json
import math
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 4 << 20
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

def parse(text):
    rows, seen = {}, set()
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        m = SAMPLE.fullmatch(line)
        if not m:
            raise ValueError('invalid metrics sample')
        name, raw, v = m.groups()
        key = name + (raw or '')
        if key in seen:
            raise ValueError('duplicate metrics sample: ' + name)
        seen.add(key)
        value = float(v)
        if not math.isfinite(value):
            raise ValueError('nonfinite metrics sample: ' + name)
        labels = {}
        if raw:
            inner = raw[1:-1]
            offset = 0
            while offset < len(inner):
                label = LABEL.match(inner, offset)
                if label is None or label.group(1) in labels:
                    raise ValueError('invalid or duplicate metric label')
                labels[label.group(1)] = label.group(2)
                offset = label.end()
                if offset < len(inner):
                    if inner[offset] != ',' or offset + 1 == len(inner):
                        raise ValueError('invalid metric labels')
                    offset += 1
        rows.setdefault(name, []).append({'labels': labels, 'value': value})
    return rows

def fetch(url, headers=None):
    # Never inherit environment proxy settings (loopback exporter/dashboard).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(urllib.request.Request(url, headers=headers or {}), timeout=3) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('metrics response too large')
    return data.decode('utf-8')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def check(text):
    rows = parse(text)
    panels = json.loads((ROOT / 'dashboard/panels.json').read_text())
    required = set(['rexl3_engine_state'])
    optional = set()
    for p in panels:
        if p.get('source') == 'telemetry':
            continue
        names = [p['series']]
        if p.get('secondary'):
            names.append(p['secondary'])
        if p['kind'] == 'histogram':
            names = [p['series'] + suffix for suffix in ['_bucket', '_sum', '_count']]
        (optional if p.get('optional') else required).update(names)
    absent = sorted(required - rows.keys())
    for p in panels:
        if p['kind'] != 'histogram' or p['series'] + '_bucket' not in rows:
            continue
        name = p['series']
        buckets = sorted(rows[name + '_bucket'], key=lambda b: float(b['labels']['le']))
        values = [b['value'] for b in buckets]
        if len(buckets) != 25 or values != sorted(values) or values[-1] != rows[name + '_count'][0]['value']:
            raise ValueError('invalid histogram: ' + name)
    if 'rexl3_kv_pool_tokens_used' in rows and 'rexl3_kv_pool_tokens' in rows:
        if rows['rexl3_kv_pool_tokens_used'][0]['value'] > rows['rexl3_kv_pool_tokens'][0]['value']:
            raise ValueError('reserved KV exceeds capacity')
    return {'missing_required': absent, 'optional_absent': sorted(optional - rows.keys()), 'present_families': len(rows)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('engine_url')
    a = ap.parse_args()
    try:
        result = check(fetch(a.engine_url.rstrip('/') + '/metrics'))
    except Exception as e:
        print(json.dumps({'error': str(e)})); return 1
    print(json.dumps(result, sort_keys=True))
    return int(bool(result['missing_required']))

if __name__ == '__main__':
    raise SystemExit(main())
