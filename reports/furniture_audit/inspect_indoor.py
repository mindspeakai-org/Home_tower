"""Generate original-vs-prediction room examples and inspect dataset leakage."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from run_audit import ARCHIVE, COCO_TO_ORIGINAL, EXPORTED, OUT, ROOT, WEIGHT, dump, sha256


def draw_record(z, record, title, names):
    frame = cv2.imdecode(np.frombuffer(z.read(record['image']), np.uint8), cv2.IMREAD_COLOR)
    for row in record['ground_truth']:
        cls, *bbox = row
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, 'GT ' + names[str(int(cls))], (x1, max(y1 + 13, 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 220, 0), 1)
    for x1, y1, x2, y2, score, cls in record['predictions']:
        if score < .25 or int(cls) not in {56, 57, 59, 60, 69}:
            continue
        label = {56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table', 69: 'oven'}[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 150, 255), 1)
        cv2.putText(frame, f'{label} {score:.2f}', (int(x1), max(int(y2) - 4, 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 150, 255), 1)
    panel = np.full((470, 416, 3), 25, np.uint8)
    panel[:416] = frame
    cv2.putText(panel, title, (8, 435), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    cv2.putText(panel, 'GT green | model orange | conf >= 0.25', (8, 456),
                cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
    return panel


def main():
    records = [json.loads(line) for line in (OUT / 'test_predictions.jsonl').read_text().splitlines()]
    inventory = json.loads((OUT / 'inventory.json').read_text())
    names = inventory['classes']
    per_image = json.loads((OUT / 'strict_original_per_image.json').read_text())
    by_image = {r['image']: r for r in records}
    counts = {r['image']: r['classes'] for r in per_image}
    candidates, selected = [], set()
    # Prefer room context (multiple annotation types) rather than product cutouts.
    for identifier, label in ((0, 'bed'), (18, 'sofa'), (4, 'chair'), (8, 'dining table'), (20, 'generic table')):
        for criterion in ('tp', 'fn'):
            pool = [r for r in records if counts[r['image']].get(str(identifier), {}).get(criterion, 0) > 0
                    and r['image'] not in selected]
            pool.sort(key=lambda r: (len({int(g[0]) for g in r['ground_truth']}),
                                     len(r['ground_truth'])), reverse=True)
            for record in pool[:2]:
                selected.add(record['image'])
                candidates.append((f'{label} {criterion.upper()}', record))
    ovens = sorted([r for r in records if any(p[4] >= .25 and int(p[5]) == 69 for p in r['predictions'])],
                   key=lambda r: max(p[4] for p in r['predictions'] if int(p[5]) == 69), reverse=True)
    for record in ovens[:4]:
        if record['image'] not in selected:
            selected.add(record['image'])
            candidates.append(('kitchen / COCO oven proxy', record))
    # Empty GT with mapped predictions reveals incomplete annotations, if present.
    empty = [r for r in records if not r['ground_truth']
             and any(p[4] >= .5 and int(p[5]) in COCO_TO_ORIGINAL for p in r['predictions'])]
    for record in empty[:4]:
        candidates.append(('EMPTY GT / model furniture', record))
    for record in records:
        if ('/ikea-' in record['image'] and record['image'] not in selected
                and any(int(g[0]) == 0 for g in record['ground_truth'])):
            candidates.append(('IKEA room / Bed annotation', record))
    examples = []
    with zipfile.ZipFile(ARCHIVE) as z:
        panels = []
        example_dir = OUT / 'indoor_examples'
        example_dir.mkdir(exist_ok=True)
        for index, (description, record) in enumerate(candidates):
            key = f'{index:02d}'
            panel = draw_record(z, record, key + ' ' + description, names)
            cv2.imwrite(str(example_dir / f'{key}.jpg'), panel)
            panels.append(panel)
            examples.append({'index': key, 'description': description, **record})
        for start in range(0, len(panels), 12):
            group = panels[start:start + 12]
            group += [np.zeros_like(group[0])] * ((-len(group)) % 3)
            sheet = np.vstack([np.hstack(group[i:i + 3]) for i in range(0, len(group), 3)])
            cv2.imwrite(str(OUT / f'indoor_contact_sheet_{start // 12 + 1}.jpg'), sheet)
        dump('indoor_examples.json', examples)

        # Distinguish byte-identical duplicates, conflicting annotations, and
        # perceptual candidates. No near-duplicate is silently deleted.
        conflicts = []
        for group in inventory['duplicates']['exact_duplicate_groups']:
            labels = [z.read(str(Path(n.replace('/images/', '/labels/')).with_suffix('.txt'))).decode()
                      for n in group]
            if len(set(labels)) > 1:
                conflicts.append({'images': group, 'annotations': labels})
        hashes, thumbnails = {}, {}
        images = [n for n in z.namelist() if '/images/' in n and not n.endswith('/')]
        for index, name in enumerate(images):
            image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError('Invalid image: ' + name)
            small = cv2.resize(image, (32, 32)).astype(np.float32)
            dct = cv2.dct(small)[:8, :8].ravel()[1:]
            bits = dct > np.median(dct)
            hashes[name] = sum(int(b) << k for k, b in enumerate(bits))
            thumbnails[name] = cv2.resize(small, (16, 16))
            if (index + 1) % 2000 == 0:
                print(f'IMAGE INTEGRITY / PERCEPTUAL HASH {index + 1}/{len(images)}', flush=True)
        train = [(n, h) for n, h in hashes.items() if n.startswith('train/')]
        near = []
        for name, h in hashes.items():
            if not name.startswith(('test/', 'valid/')):
                continue
            for other, train_hash in train:
                distance = (h ^ train_hash).bit_count()
                if distance <= 4:
                    mae = float(np.abs(thumbnails[name] - thumbnails[other]).mean())
                    if mae <= 12:
                        near.append({'held_out': name, 'train': other, 'phash_hamming': distance,
                                     'grayscale_thumbnail_mae': mae})
        near.sort(key=lambda row: (row['grayscale_thumbnail_mae'], row['phash_hamming']))
        duplicates = {'exact_duplicate_annotation_conflicts': conflicts, 'phash_candidates': near,
                      'decoded_images': len(hashes), 'criterion': 'pHash Hamming <=4 and 16x16 grayscale MAE <=12/255',
                      'warning': 'Perceptual similarity is not proof of leakage; inspect candidate pairs'}
        dump('content_integrity.json', duplicates)
        print('CONTENT INTEGRITY', len(hashes), 'images decoded;', len(conflicts),
              'duplicate annotation conflicts;', len(near), 'perceptual candidates', flush=True)
        if near:
            pairs = []
            used = set()
            for row in near:
                if row['held_out'] in used:
                    continue
                used.add(row['held_out'])
                tiles = []
                for key in ('held_out', 'train'):
                    image = cv2.imdecode(np.frombuffer(z.read(row[key]), np.uint8), cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (320, 320))
                    panel = np.full((350, 320, 3), 25, np.uint8)
                    panel[:320] = image
                    cv2.putText(panel, key + f' MAE {row["grayscale_thumbnail_mae"]:.2f}', (5, 341),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                    tiles.append(panel)
                pairs.append(np.hstack(tiles))
                if len(pairs) == 4:
                    break
            cv2.imwrite(str(OUT / 'near_duplicate_pairs.jpg'), np.vstack(pairs))

        # Confirm that the deployed export is compatible with the PT checkpoint.
        model = YOLO(str(WEIGHT))
        model.predict(np.zeros((480, 480, 3), np.uint8), device='cpu', imgsz=480, rect=False, verbose=False)
        comparisons = []
        for index in np.linspace(0, len(records) - 1, 20, dtype=int):
            record = records[index]
            image = cv2.imdecode(np.frombuffer(z.read(record['image']), np.uint8), cv2.IMREAD_COLOR)
            result = model.predict(image, device='cpu', imgsz=480, rect=False, conf=.25, iou=.7,
                                   max_det=300, verbose=False)[0]
            pt = result.boxes.data.cpu().numpy()
            ov = np.array([p for p in record['predictions'] if p[4] >= .25]).reshape(-1, 6)
            similarities = []
            for p in pt:
                same = ov[ov[:, 5] == p[5]]
                if len(same):
                    lo = np.maximum(same[:, :2], p[:2])
                    hi = np.minimum(same[:, 2:4], p[2:4])
                    intersect = np.maximum(hi - lo, 0).prod(axis=1)
                    union = (same[:, 2:4] - same[:, :2]).prod(axis=1) + (p[2:4] - p[:2]).prod() - intersect
                    similarities.append(float((intersect / np.maximum(union, 1e-9)).max()))
                else:
                    similarities.append(0.0)
            comparisons.append({'image': record['image'], 'pt_count': len(pt), 'openvino_count': len(ov),
                                'pt_best_same_class_ious': similarities})
        exports = {str(p.relative_to(ROOT)): sha256(p.read_bytes())
                   for p in (ROOT / 'weights').rglob('*') if p.is_file()}
        dump('export_checkpoint_comparison.json', {'samples': comparisons, 'weight_file_sha256': exports})
        print('PT / OPENVINO comparison complete on 20 deterministic held-out samples', flush=True)


if __name__ == '__main__':
    main()
