# Setup and collection test — 2026-09-14

We cloned only the competition source from GitHub into a fresh folder and tested everything from installation to training data collection.
The tested code is part of the current `main` branch. No run folders, downloaded assets, or caches were copied from an existing development environment.
**This was a fresh install on the same PC, not a test on a different PC or a freshly installed operating system.**

## Test environment

Ubuntu Desktop 22.04.5, NVIDIA RTX 5080 16 GB (driver 580.178.04), Docker 29.1.3.
The Isaac Sim 6.0.1 image pinned by the repository was used, and the project's Docker build ran from scratch without cache.

## Results

On the fresh clone, the full sequence `build → prepare → smoke → gui --training-record 1 --evaluate 1` worked.

- All 8 required asset hashes verified; generated runtime code matched the GitHub source
- Driving both robots, gripper open/close, and saving/loading scene state (0 m cloth vertex error) worked
- Every physics tick was recorded; after ESC the training HDF5 was generated with no extra command (`READY`, `complete=true`)
- Synchronization checks passed for all 5 cameras (RGB, depth, valid mask), observations, actions, and camera poses
- Automatic evaluation without a button (`--evaluate 1`) started and saved correctly

No robot, manikin, or shirt settings were changed to make these checks pass.
With an empty cache, the first preparation can take several minutes.

## How to check on your own machine

After following the [setup guide](SETUP.md), run:

```bash
./run.sh smoke
./run.sh gui --training-record 1 --evaluate 1
```

Confirm `PHYRC-SMOKE-PASS`, then start the GUI. Click the 3D view, drive briefly, and press ESC.
**Keep the terminal open after the window closes and wait for `[Dataset export] READY`.**
You can check your own training files with the commands in the [data guide](DATA.md#retrying-the-export-and-checking-files).

This was a short functional test of the install, teleop, and saving path.
It does not show that a policy can be trained to complete the dressing, cover long recordings, or guarantee every GPU, driver, and OS combination.
