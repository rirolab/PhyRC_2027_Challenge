# Third-party components

The code in this repository is released under the [Apache License 2.0](LICENSE).
That license covers the source code written for this project. It does **not**
cover the third-party components below, which keep their own terms.

| Component | Where it comes from | How it is distributed here |
|---|---|---|
| NVIDIA Isaac Sim 6.0.1 | Official `nvcr.io/nvidia/isaac-sim` container; the base image digest is pinned in `Dockerfile` | Downloaded by `./run.sh build`. NVIDIA's terms apply |
| Simulation source (`src/PhyRC_Sim`) | Teleop scene and garment helpers adapted from an open-source garment-manipulation framework and ported to Isaac Sim 6.0.1 surface FEM, plus this project's policy, recording, and evaluation code | Included |
| Stretch4 robot | `assets/custom/robots/stretch_4/stretch_4.usd` (from RCareWorld-2.0), with visual, collision, and articulation data embedded | Included as a single USD file |
| Female manikin | `assets/custom/human/female2_c4-c5.usd` and `assets/custom/human/textures/c4-c5.jpg` | Included, used with permission from Cornell University |
| T-shirt | `assets/custom/garments/N_PR_2000_TShirt001.usd` | Included. The simulation geometry is generated from it locally during `prepare` |
| Floor texture | One JPEG from `Scene.zip` in a public Hugging Face garment-manipulation dataset (Apache-2.0), pinned to a fixed revision | Downloaded by `./run.sh prepare`; not committed |
| Initial cloth material | `linen_Pumpkin.usd` and its texture JPEGs from the upstream garment-manipulation framework, pinned to a fixed commit | Downloaded by `./run.sh prepare`; not committed |
| Python packages | termcolor, scipy, matplotlib (other libraries come with Isaac Sim) | Installed at pinned versions by `./run.sh build`; not committed |

Every downloaded file has its exact URL and SHA-256 hash recorded in
[`config/assets.lock.json`](config/assets.lock.json). `prepare` refuses files
whose hash does not match.

If you redistribute the assets under `assets/custom/` or any downloaded files,
check the terms of each component first. Questions: phyrc.challenge@gmail.com.
