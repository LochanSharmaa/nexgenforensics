import sys

log_path = r'C:\Users\hello\.gemini\antigravity-ide\brain\c34a4058-82a5-461c-8c74-1fd3837d83d0\.system_generated\tasks\task-628.log'
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total log lines: {len(lines)}')

# Parse per-probe lines
probe_lines = [l.strip() for l in lines if '| Status:' in l and ('HIT ' in l or 'MISS' in l)]
print(f'Total probe results parsed: {len(probe_lines)}')

stats = {}
errors = {}
for line in probe_lines:
    try:
        ds = line.split(']')[0].strip('[').strip()
        status = 'HIT' if '| Status: HIT ' in line else 'MISS'
        target = line.split('Target:')[1].split('|')[0].strip()
        probe = line.split('Probe:')[1].split('|')[0].strip()
        top = line.split('Top:')[1].split('|')[0].strip()
        score = float(line.split('Score:')[1].strip())

        if ds not in stats:
            stats[ds] = {'hits': 0, 'total': 0}
            errors[ds] = []
        stats[ds]['total'] += 1
        if status == 'HIT':
            stats[ds]['hits'] += 1
        else:
            errors[ds].append({'probe': probe, 'target': target, 'top': top, 'score': score})
    except Exception as e:
        pass

total_hits = sum(s['hits'] for s in stats.values())
total_probes = sum(s['total'] for s in stats.values())

print()
print('=' * 80)
print('PHASE 1 BENCHMARK - 200 IDENTITY MULTI-DATASET COMPLETE RESULTS')
print('=' * 80)
print()
print(f'OVERALL 1:N RANK-1 ACCURACY: {total_hits}/{total_probes} = {(total_hits/total_probes*100):.2f}%')
print()
print('DATASET BREAKDOWN:')
for ds, s in stats.items():
    acc = (s['hits'] / s['total'] * 100) if s['total'] > 0 else 0
    misses = s['total'] - s['hits']
    print(f'  {ds:12s}: {s["hits"]:3d} HIT / {s["total"]:3d} TOTAL = {acc:.2f}%  ({misses} failures)')

print()
print('ALL FAILURE ENTRIES PER DATASET:')
for ds, errs in errors.items():
    print(f'\n[{ds}] --- {len(errs)} failures ---')
    for e in errs:
        print(f'  Probe: {e["probe"]:40s} | Target: {e["target"]:30s} | Predicted: {e["top"]:30s} | Score: {e["score"]:.4f}')
