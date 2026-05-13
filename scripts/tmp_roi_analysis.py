import json

with open('Data/BOLDMomentsDataset/targets/metadata.json') as f:
    meta = json.load(f)

sub = meta['sub-01']
rois = sub['roi_per_region_sizes']

categories = {
    'Early Visual':  ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v', 'V3ab', 'hV4'],
    'Scene/Place':   ['PPA', 'RSC', 'TOS'],
    'Body':          ['EBA'],
    'Face':          ['FFA', 'OFA'],
    'Object':        ['LOC'],
    'Motion':        ['MT'],
    'Temporal/STS':  ['STS'],
    'Parietal':      ['IPS0', 'IPS1-2-3', '7AL'],
    'Somatosensory': ['BA2', 'PFt', 'PFop'],
}

print('BMD ROI FUNCTIONAL GROUPING')
cat_counts = []
for cat, regions in categories.items():
    cat_total = 0
    for r in regions:
        l_name = f'l{r}'
        r_name = f'r{r}'
        cat_total += rois.get(l_name, 0) + rois.get(r_name, 0)
    cat_counts.append(cat_total)
    print(f'{cat:20s}: {cat_total:5d} voxels')

print(f'\nTotal: {sum(cat_counts)} voxels')
print(f'N categories: {len(cat_counts)}')
