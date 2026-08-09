# XRF Group Localization

This repository contains the compact reference implementation accompanying
*Acquisition Geometry-Assisted Whole-Group Localization of X-ray Fluorescence
Maps in Optical Microscopy Images*. It is provided for inspection of the
methods described in the manuscript. It is research code, not a reproduction
package, production application, or stable software library.

The implementation treats related X-ray fluorescence (XRF) tiles as a group
whose relative scan positions are known. It includes the independent per-tile
reference, two single-scale group strategies, and two bridge-informed
multiscale strategies with the runtime selection rule reported in the paper.

## Methods

| Manuscript term | Python function |
| --- | --- |
| Independent per-tile localization | `localize_independent_tiles` |
| Anchor-verified group localization | `localize_anchor_verified_group` |
| Mosaic localization | `localize_group_mosaic` |
| Structural-bridge localization | `localize_structural_bridge` |
| Bridge-prior direct localization | `localize_with_bridge_prior` |
| Bridge runtime rule | `localize_with_bridge_rule` |
| GroupIoU | `group_iou` |
| TileIoU summaries | `tile_iou_summary` |

All reader-facing functions are exported by `groupmatch`. Localization returns
`None` when no candidate passes the runtime checks. Data types and the fixed
manuscript configurations are defined in `groupmatch.types`.

## Environment

The locked environment targets Linux x86-64 with Python 3.10 and requires
Pixi 0.75 or newer.

```bash
pixi install
pixi run test
```

The tests create synthetic TIFF, MAPS HDF5, and XML inputs. They check the
method paths without reproducing any experimental result from the manuscript.

## Scientific Inputs

`load_optical_image` reads optical TIFF images, including pixels-per-inch or
pixels-per-centimetre resolution metadata, and accepts an optional orientation
XML file. `load_xrf_tile` reads an explicitly named MAPS HDF5 dataset together
with `MAPS/channel_names`, `MAPS/x_axis`, and `MAPS/y_axis`. Physical centres
and pixel steps use micrometres. Image coordinates use `(x, y)`, rectangles use
`(x, y, width, height)`, angles use degrees, and transforms are homogeneous
source-to-destination matrices.

Experimental data are not included. They are available from the corresponding
authors upon reasonable request, subject to applicable institutional
conditions.

## Citation

Please cite the accompanying manuscript. Bibliographic identifiers will be
added after publication.

## License

The code is released under the BSD 3-Clause License.
