## Detection

This folder hosts source code to evaluate object detection results of the enhanced images. The overall process is as follows:

1. Given a batch of sRGB images, we use the [mmdetection](https://github.com/open-mmlab/mmdetection) library to obtain detection results and dump them to a JSON file (in COCO format).
2. After obtaining the detection results, we use the well-organized [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics.git) library to compute the mean Average Precision (mAP) metrics.

### Installation

```sh
chmod +x setup.sh
./setup.sh
```

### Run Detection

```sh
python inference.py --data_dir $FOLDER --jsonfile_prefix $PREFIX [optional arguments]
```

* `--data_dir $FOLDER`: The directory containing sRGB images.
* `--jsonfile_prefix $PREFIX`: The prefix of the output JSON filename. The detection result will be saved to "./det_results/${PREFIX}_bbox.json".

Optional arguments are:

* `--out_dir $FOLDER`: The directory to save detection results. If not given, results will be saved to "./det_results".
* `--score_thr $VALUE`: Minimum score of bounding boxes to keep. This argument is used for exporting ground-truth object annotations on the SID dataset.
* `--no_vis`: By default, the codebase will save the visualization results to "./det_results". To disable this behavior, use `--no_vis`.

### Evaluation

```sh
python eval.py --det_result $DET_ANN --gt_annotation $GT_ANN
```

* `--det_result $DET_ANN`: The detection result in COCO format.
* `--gt_annotation $GT_ANN`: The ground truth object annotations, in COCO format.
