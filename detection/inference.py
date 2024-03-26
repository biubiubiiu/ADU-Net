import os
import os.path as osp

import mmcv
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector


class Runner:

    # Choose to use a config and initialize the detector
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = {
        'url': 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth',
        'path': 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth',
    }
    # Set the device to be used for evaluation
    device = 'cuda:0'

    def __init__(self):
        # Load the config
        config = mmcv.Config.fromfile(self.config_file)
        # Set pretrained to be None since we do not need pretrained model here
        config.model.pretrained = None

        # Initialize the detector
        model = build_detector(config.model).to(self.device)

        # Download checkpoint file if not exists
        if not osp.exists(self.checkpoint['path']):
            os.makedirs(osp.dirname(self.checkpoint['path']), exist_ok=True)
            os.system(f'wget -c {self.checkpoint["url"]} -O {self.checkpoint["path"]}')

        # Load checkpoint
        checkpoint = load_checkpoint(
            model, self.checkpoint['path'], map_location=self.device
        )

        # Set the classes of models for inference
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.cfg = config

        # Convert the model into evaluation mode
        model.eval()

        self.config = config
        self.model = model
        self.categories = model.CLASSES

    def inference(self, data_dir, out_dir, jsonfile_prefix, score_thr=0, save_vis=True):
        """Inference the detector

        Args:
            data_dir (str): Input images to load from
            out_dir (str): Directory to save detection results
            jsonfile_prefix (str): The filename prefix of output json file
            score_thr (float): Minimum score of bboxes to keep. Default: 0.
            save_vis (bool): Save visualization results or not. Default: True.
        """
        input_fnames = sorted(os.listdir(data_dir))
        input_fpaths = [osp.join(data_dir, fname) for fname in input_fnames]

        # Use the detector to do inference
        results = []
        for fp in tqdm(input_fpaths, desc='Inferencing'):
            ret = inference_detector(self.model, fp)
            results.append(ret)

        # save detection results
        if save_vis:
            for fp, fn, result in tqdm(
                zip(input_fpaths, input_fnames, results),
                total=len(results),
                desc='Saving visualization result',
            ):
                show_result_pyplot(
                    self.model,
                    fp,
                    result,
                    score_thr=0.5,
                    out_file=osp.join(out_dir, 'visualization', fn),
                    thickness=6,
                    font_size=42,
                )

        formatter = Formatter(img_ids=input_fnames, category_names=self.categories)
        out_file = formatter.format_results(
            results, osp.join(out_dir, jsonfile_prefix), score_thr
        )
        return out_file


class Formatter:
    def __init__(self, img_ids, category_names):
        self.image_ids = img_ids
        self.category_names = category_names

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _det2json(self, results):
        """Convert detection results to COCO json style"""
        json_results = []
        for i, result in tqdm(enumerate(results), desc='Dumping json file'):
            img_id = self.image_ids[i]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = label
                    data['category_name'] = self.category_names[label]
                    json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json file. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_file = None
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_file = f'{outfile_prefix}_bbox.json'
            mmcv.dump(json_results, result_file)
        else:
            raise TypeError('invalid type of results')
        return result_file

    def format_results(self, results, jsonfile_prefix, score_thr=0):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            score_thr (float): Minimum score of bboxes to keep. Default: 0.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if score_thr > 0:
            for result in results:
                for label in range(len(result)):
                    bboxes = result[label]
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    result[label] = bboxes[inds, :]

        result_file = self.results2json(results, jsonfile_prefix)
        return result_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/SID/Sony/gt_srgb')
    parser.add_argument('--out_dir', type=str, default='det_results')
    parser.add_argument('--jsonfile_prefix', type=str)
    parser.add_argument('--score_thr', type=float, default=0)
    parser.add_argument('--no_vis', action='store_true')
    args = parser.parse_args()

    runner = Runner()
    out_file = runner.inference(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        jsonfile_prefix=args.jsonfile_prefix,
        score_thr=args.score_thr,
        save_vis=not args.no_vis,
    )
