# Local model weights

Weights and CPU exports are generated locally and excluded from Git.

```bash
python -c 'from src.stage1_calibration.download_stage1_data import download_weights; download_weights()'
python -m src.utils.export_cpu_models
```

Required checkpoints are `yolo11n.pt` and `yolo11n-pose.pt`. Their current heads
are COCO detection and COCO pose; neither was fine-tuned on the furniture data.
Exports prefer OpenVINO on x86_64 and fall back to ONNX on CPU.

The audited detector SHA-256 is
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
Bootstrap code validates this identity before reusing cached audit predictions.
See `reports/furniture_audit/checkpoint_evidence.json` for model provenance.

Model licensing is separate from this repository's source licensing; consult
`THIRD_PARTY_NOTICES.md` before redistribution.
