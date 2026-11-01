"""Builds static/Data/demo_plan.json, the ready-made load plan played on the home page.

Run from the project folder:  python tools/make_demo_plan.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import hga  # noqa: E402
import compaction  # noqa: E402

DATA = os.path.join(ROOT, 'static', 'Data')
PALETTE = ['#B9824A', '#4E79A6', '#D2A468', '#8C5B2E', '#7E95AE', '#C79A5B']  # same order as static/js/app.js
UPRIGHT = [1, 3]


def main():
    demos = json.load(open(os.path.join(DATA, 'demo_shipments.json'), encoding='utf-8'))
    trucks = json.load(open(os.path.join(DATA, 'camions.json'), encoding='utf-8'))
    demo = demos['truck']
    truck = next(t for t in trucks if t['Nom'] == demo['vehicle'])
    container = {'dc': truck['Longueur (m)'], 'wc': truck['Largeur (m)'], 'hc': truck['Hauteur (m)']}

    boxes, kinds = [], {}
    for idx, it in enumerate(demo['items']):
        kinds[str(idx)] = {'name': it['name'], 'color': PALETTE[idx % len(PALETTE)]}
        for _ in range(it['qty']):
            boxes.append({'id': len(boxes) + 1, 'name': it['name'], 'original_type_id': str(idx),
                          'db': it['d'], 'wb': it['w'], 'hb': it['h'], 'weight': it['weight'],
                          'lbs': it['lbs'] or 100000, 'allowed_orientations': UPRIGHT if it.get('upright') else list(range(1, 7))})
    id_to_kind = {b['id']: b['original_type_id'] for b in boxes}

    solution = hga.solve_loading_problem(container, boxes, {'npop': 20, 'nrep': 6, 'nmerge': 6, 'max_generations': 15, 'max_time_limit': 20})
    compaction.compact_solution(solution)
    placed = [b for layer in solution['visualization_data']['layers'] for b in layer['boxes']]
    m = solution['metrics']
    plan = {
        'vehicle': f"{truck['Nom']} · {container['dc']:.1f} × {container['wc']:.1f} × {container['hc']:.1f} m".replace('.', ','),
        'L': container['dc'], 'W': container['wc'], 'H': container['hc'],
        'total': len(boxes),
        'fill': round(m['total_packed_volume'] / m['container_volume'] * 100, 1),
        'kinds': kinds,
        'boxes': [{'k': id_to_kind[b['id']], 'x': round(b['ox'], 4), 'y': round(b['oy'], 4), 'z': round(b['oz'], 4),
                   'dx': round(b['dx'], 4), 'dy': round(b['dy'], 4), 'dz': round(b['dz'], 4)} for b in placed],
    }
    out = os.path.join(DATA, 'demo_plan.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False)
    print(f"{out}: {len(placed)}/{len(boxes)} boxes, {plan['fill']} %")


if __name__ == '__main__':
    main()
