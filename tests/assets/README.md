# Generated integration fixture

`test_e2e_pipeline.py` generates `e2e_test_scene.mp4` locally: 360 frames at
640×480 and 30 FPS. The fixture and rendered videos are not committed.

Scripted observations drive deterministic alert phases. The test exercises real
video I/O, scheduling, HSV sensing, monitors, logging, and HUD rendering; it does
not measure neural detection accuracy. Real-model tests need local checkpoints
and, for Stage 1 integration, the prepared source dataset cache.
