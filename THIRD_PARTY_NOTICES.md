# Third-Party Notices

NovaSplat's own code is released under the MIT License (see [LICENSE](LICENSE)).
That license covers **only** code written by the NovaSplat authors. All
third-party components listed below keep their own licenses, which you must
comply with when using or redistributing them.

## Git submodules (`thirdparty/`)

Submodules are referenced by URL and commit. Their code is not copied into this
repository. Each submodule contains its own license file.

| Component | License | Modified fork | Use in NovaSplat | Upstream |
|---|---|---|---|---|
| bop_toolkit | MIT | Yes (POSE-Lab fork) | Imported (`bop_toolkit_lib`) | https://github.com/thodan/bop_toolkit |
| COLMAP | BSD-3-Clause | No | Built from source, used via `pycolmap` | https://github.com/colmap/colmap |
| DINOv3 | DINOv3 License (Meta) | No | Feature extraction | https://github.com/facebookresearch/dinov3 |
| SAM 3 | SAM License (Meta) | No | Imported (`sam3`) for segmentation | https://github.com/facebookresearch/sam3 |
| OpenMVS | **AGPL-3.0** | Yes (POSE-Lab fork) | Built from source by `install.sh` / `Dockerfile` | https://github.com/cdcseacave/openMVS |
| VCGlib | **GPL-3.0** | No | Built together with OpenMVS | https://github.com/cnr-isti-vclab/vcglib |
| GeDi | **No license published upstream** (see notes) | Yes (POSE-Lab fork) | Imported (`gedi`) for 3D descriptors | https://github.com/fabiopoiesi/gedi |

Notes:

- The POSE-Lab forks of bop_toolkit and OpenMVS keep the license of their
  upstream projects. Our modifications are distributed under those same
  licenses (for OpenMVS: AGPL-3.0).
- The upstream GeDi repository (https://github.com/fabiopoiesi/gedi) does not
  contain a license file, and GitHub reports no license for it. Without a
  license, the authors retain all rights, so no permission to use,
  modify or redistribute GeDi is granted by default. Our POSE-Lab fork
  adds no license of its own. Obtain permission from the GeDi authors before
  using or redistributing it. Its `backbones/pointnet2_ops_lib` directory comes
  from https://github.com/erikwijmans/Pointnet2_PyTorch (Unlicense).
- DINOv3 and SAM 3 (code **and** model weights) are distributed under Meta's
  custom licenses, which are not OSI open-source licenses. Their terms,
  including use restrictions, apply to you. Model weights are not included in
  this repository. Download them from the upstream sources and accept their
  licenses.
- OpenMVS and VCGlib are copyleft. They are built and used as separate
  programs and are not linked into NovaSplat's Python code. If you distribute
  binaries or container images containing them, you must also provide their
  corresponding source and license texts.

## Code vendored in `thirdparty/`

Only the files listed here were copied. The upstream projects are not included
in full. Each folder carries the upstream license text, original copyright
notices are retained in file headers, and modified files state that they were
changed.

| Folder | Derived from | License | Modified | Base commit |
|---|---|---|---|---|
| `thirdparty/cnos25` | https://github.com/Vision-Kek/cnos25 (itself derived from https://github.com/nv-nguyen/cnos) | MIT | Yes | d92e3d32f55202c634aa698a4435aae993089a59 |
| `thirdparty/cnos25/src/utils/segment_anything_transforms.py` | https://github.com/facebookresearch/segment-anything | Apache-2.0 | No | dca509fe793f601edb92606367a655c15ac00fdf |
| `thirdparty/gsplat` (example scripts only) | https://github.com/nerfstudio-project/gsplat (`examples/`) | Apache-2.0 | Yes | 937e29912570c372bed6747a5c9bf85fed877bae |
| `thirdparty/gsplat/datasets/{camera,image,rotation,scene_manager}.py` | https://github.com/trueprice/pycolmap (Copyright (c) 2018 True Price, UNC Chapel Hill) | MIT (see `thirdparty/gsplat/datasets/LICENSE-pycolmap`) | `camera.py` and `scene_manager.py`: Yes. `image.py` and `rotation.py`: No | fe7a7c45df803b6c391777e349f0d8d65d39d777 |
| `thirdparty/gsplat/datasets/traj.py` (partly) | https://github.com/google-research/multinerf (`internal/camera_utils.py`, Copyright 2022 Google LLC) | Apache-2.0 | Borrowed code, as in upstream gsplat | 5b4d4f64608ec8077222c52fdf814d40acc10bc1 |

Files in these folders that were written by the NovaSplat authors are covered
by the MIT license in the repository root.

## Model weights downloaded at runtime

Model weights are not included in this repository.

| Weights | License | Note |
|---|---|---|
| `mobileclip_blt.ts` (MobileCLIP text encoder, https://github.com/apple/ml-mobileclip) | Apple ML Research Model terms (see `LICENSE_MODELS` upstream): **research purposes only, no commercial use** | Downloaded automatically by `ultralytics` on first use of YOLOE text prompts, into the directory the code is run from (`thirdparty/cnos25/`). Redistribution requires including Apple's agreement and an attribution notice, so it is not included here. |
| `yoloe-11l-seg.pt` (YOLOE) | **AGPL-3.0** (Ultralytics / THU-MIG YOLOE) | Download from the Ultralytics sources. |
| DINOv3, SAM 3 checkpoints | Meta's custom licenses | See the submodule table above. |

## Images in `assets/`

The images in `assets/` are licensed under CC BY-SA 4.0 (see
`assets/LICENSE.md`), not under the MIT License. The qualitative images contain
frames from the HOPEv2 dataset (CC BY-SA 4.0, https://bop.felk.cvut.cz/datasets/),
which requires the adapted images to be shared under the same license.

## Notable Python dependencies

Installed via `requirements.txt` / `install.sh`, not included in this
repository. This is not an exhaustive list.

| Package | License | Note |
|---|---|---|
| `ultralytics` | **AGPL-3.0** | Imported by `thirdparty/cnos25` (YOLOE proposals). Using or deploying that code path carries AGPL-3.0 obligations. |
| `gsplat` | Apache-2.0 | Gaussian splatting rasterization library |
| `pycolmap` | BSD-3-Clause | Built with COLMAP |
| `open3d` | MIT | |

Please check the licenses of all other dependencies before redistributing or
deploying NovaSplat commercially.
