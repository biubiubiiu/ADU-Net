import argparse

import mmcv
from pycocotools.coco import COCO
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ann_paths', type=str, required=True, nargs='+')
args = parser.parse_args()


all_anns = []


for ann_path in args.ann_paths:
    print(f'Processing {ann_path}')
    coco = COCO(ann_path)
    for img_id in tqdm(coco.imgs.keys()):
        img_info = coco.imgs[img_id]
        annotations = coco.loadAnns(coco.getAnnIds([img_id]))

        for ann in annotations:
            all_anns.append(
                {
                    'image_id': ann['image_id'] + '.png',
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id'],
                    'category_name': coco.loadCats([ann['category_id']])[0]['name'],
                }
            )

result_file = f'bboxes_out.json'
mmcv.dump(all_anns, result_file)
