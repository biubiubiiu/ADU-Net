import json

from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from metrics.Evaluator import Evaluator as MetricsMeter
from metrics.utils import BBFormat, BBType, CoordinatesType, MethodAveragePrecision

from mmdet.datasets.coco import CocoDataset


class Evaluator:

    CLASSES = CocoDataset.CLASSES

    def load(self, json_path):
        """Load json file into list of dict"""
        with open(json_path, 'r') as f:
            json_data = f.read()

        data = json.loads(json_data)
        return data

    def filter_by_categories(self, data, keeplist):
        return [it for it in data if it['category_name'] in keeplist]

    def filter_by_imageid(self, data, keeplist):
        return [it for it in data if it['image_id'] in keeplist]

    def convert_to_bboxes(self, data_list, is_gt):
        out_bboxes = []
        for data in data_list:
            bb = BoundingBox(
                imageName=data['image_id'],
                classId=data['category_name'],
                x=float(data['bbox'][0]),
                y=float(data['bbox'][1]),
                w=float(data['bbox'][2]),
                h=float(data['bbox'][3]),
                typeCoordinates=CoordinatesType.Absolute,
                classConfidence=None if is_gt else float(data['score']),
                bbType=BBType.GroundTruth if is_gt else BBType.Detected,
                format=BBFormat.XYWH,
            )
            out_bboxes.append(bb)
        return out_bboxes

    def evaluate(
        self, det_json_path, gt_json_path, iou_threshold=0.5, categories_keeplist=None
    ):
        det_data = self.load(det_json_path)
        gt_data = self.load(gt_json_path)

        # remove those images without GT annotation (used for RAW-NOD only)
        all_image_ids = set([it['image_id'] for it in gt_data])
        det_data = self.filter_by_imageid(det_data, all_image_ids)

        categories_keeplist = categories_keeplist or self.CLASSES
        det_data = self.filter_by_categories(det_data, categories_keeplist)

        all_bounding_boxes = BoundingBoxes()
        for bbox in self.convert_to_bboxes(gt_data, is_gt=True):
            all_bounding_boxes.addBoundingBox(bbox)
            classes_to_keep = all_bounding_boxes.getClasses()
        for bbox in self.convert_to_bboxes(det_data, is_gt=False):
            # Ignore those classes that's not presented in GTs
            if bbox.getClassId() in classes_to_keep:
                all_bounding_boxes.addBoundingBox(bbox)

        evaluator = MetricsMeter()
        # Get metrics
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_threshold,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
        )

        APs = [mc['AP'] for mc in metricsPerClass]
        mAP = sum(APs) / len(APs)
        return mAP


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--det_result', type=str, required=True)
    parser.add_argument('--gt_annotation', type=str, required=True)
    parser.add_argument('--categories_keeplist', type=str, nargs='+')
    args = parser.parse_args()

    results = {}

    evaluator = Evaluator()
    for iou_threshold in [it / 100 for it in range(50, 100, 5)]:
        mAP = evaluator.evaluate(
            args.det_result, args.gt_annotation, iou_threshold, args.categories_keeplist
        )
        results[iou_threshold] = mAP

    print(f'mAP[.5]: {results[0.5]:.2%}')
    print(f'mAP[.75]: {results[0.75]:.2%}')
    print(f'mAP[.5:.95]: {sum(results.values()) / len(results.values()):.2%}')
