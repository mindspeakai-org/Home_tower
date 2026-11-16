# Local source data

Place the supplied Roboflow Furniture Detection v20 archive here:

```text
furniture detection.v20i.yolov11.zip
```

Source: <https://universe.roboflow.com/mokhamed-nagy-u69zl/furniture-detection-qiufc/dataset/20>.
The source metadata states CC BY 4.0; see the repository's third-party notices.
The archive is not redistributed in this Git repository. Obtain it separately
and preserve its original contents. No Open Images download is needed.

The audited archive SHA-256 is:

```text
7236062f0af92ec59a9f4cdb62743ebdcf1814b3170f4de95465e51ac175a728
```

After placing the archive, run from the project root:

```bash
python -m src.stage1_calibration.dataset_sanitizer
```
