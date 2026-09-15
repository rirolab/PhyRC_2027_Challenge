# Real teleop data example

This example comes from run `20260913T153054_259148Z_0e4bae`.
**1,142 action steps / 1,143 distinct observations / 20 Hz / 57.1 s of simulation time**.
Evaluation: pickup 5/50 points, dressing points 0/45, success=false. It is not a successful demonstration.

![RGB from the 5 cameras](../../videos/sample_cameras_rgb.gif)

[5-camera RGB MP4](../../videos/sample_all_cameras_rgb.mp4) · [Depth MP4](../../videos/sample_all_cameras_depth.mp4) ·
[Teleop viewport MP4](../../videos/sample_teleop_viewport.mp4)

On GitHub, use the GIF above and the CSV files.
After cloning, open **[index.html](index.html)** in this folder in a browser to see the videos and example numbers on one page.
GitHub's file view does not run HTML pages.

## Files

- [actions](demo_0_actions.csv): 18 normalized commands and timing for each of the 1,142 steps.
- [obs](demo_0_obs.csv) / [next_obs](demo_0_next_obs.csv): all non-image public observations just before / after each step.
- [applied targets](demo_0_applied_targets.csv): 3,426 rows of 60 Hz controller steps. Audit only; do not use as action labels.
- [evaluation](demo_0_evaluation.csv): for evaluation and selecting demonstrations; not a policy input.
- [columns](columns.csv): unit, index, meaning, and first-row example for all 356 CSV columns.
- [video frame index](video_frame_index.csv): maps preview frames to the original ticks and times.
- [contract](contract.json), [HDF5 schema](hdf5_schema.json), [validation](validation.json).

Every numeric row is kept. The videos are 5 fps previews with 287 frames: one frame per 4 observations, plus the last observation.
Because the last frame is shown, the MP4 is 57.4 s long while the actual simulation span is 57.1 s.
The human operator took about 595.7 s of wall-clock time. The RGB/depth preview GIFs cover 16–28 s of that.
The time at the top of each frame is the original observation's simulation_time_s.

The original policy.hdf5 (about 3.16 GB) and audit.hdf5 (about 182 MB) are not in Git.
The MP4/GIF and CSV files are here to explain the data layout; this is not a full training dataset.
Collect your own data with `./run.sh gui --training-record 1`.

See the [data guide](../../competition/DATA.md) for what each key means and the [rules](../../competition/RULES.md) for what you may use.
