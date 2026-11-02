# mCellSeg: A Microscopy Cell Instance Segmentation Dataset

mCellSeg is an expert-annotated microscopy image dataset designed for benchmarking cell instance segmentation models. The dataset contains images from two human cell lines acquired using differential interference contrast (DIC) and fluorescence microscopy. It captures diverse cellular morphologies, varying confluency levels, and challenging imaging conditions commonly encountered in real laboratory environments.

The dataset is intended to support the development and evaluation of deep learning methods for robust cell segmentation in microscopy images.

---

## Dataset Overview

mCellSeg contains:

- **200 manually annotated microscopy images**
  - 100 HEK-293T images
  - 100 HUVEC images
- **100 additional unannotated images**
  - Suitable for semi-supervised learning
- **Instance segmentation masks** in TIFF format
- Expert-reviewed annotations for high-quality benchmarking

---

## Dataset Structure

```text
mCellSeg/
    labeled/
        images/     # 200 annotated microscopy images (.tif)
        masks/      # Corresponding instance segmentation masks (.tif)
    unlabeled/
        images/     # 100 unannotated microscopy images (.tif)
    README.md
```

---

## Dataset Statistics

| Property               | Details                          |
|------------------------|----------------------------------|
| Total annotated images | 200                              |
| Unannotated images     | 100                              |
| Cell types             | HEK-293T, HUVEC                  |
| Total annotated cells  | ~16,199                          |
| Imaging modalities     | DIC and fluorescence microscopy  |
| Annotation format      | TIFF instance masks              |

---

## Cell Types

### HEK-293T

Immortalized human embryonic kidney cells cultured in high-glucose Dulbecco's Modified Eagle Medium (DMEM; Sigma) supplemented with 10% fetal bovine serum (FBS) and 1% penicillin-streptomycin. Cells were maintained at 37°C in a humidified incubator with 5% CO₂. Cytoplasmic GFP expression was induced by transfecting cells with a GFP-TREX1(D18N) plasmid (Addgene #27220) using Lipofectamine 3000 (Thermo Fisher Scientific). For imaging, cells were seeded onto 18-well µ-slides or 96-well round-bottom plates (ibidi GmbH), fixed in 4% paraformaldehyde, and stained with Hoechst 33342 (1:1000, Thermo Fisher Scientific) for nuclear visualisation.

### HUVEC

Primary human umbilical vein endothelial cells (Lonza, CC-2529A) cultured in Endothelial Cell Growth Medium-2 (EGM™-2 BulletKit; Lonza) and maintained at 37°C in a humidified incubator with 5% CO₂. Cells were passaged between passages 2 and 8, with medium changes every 2–3 days.

---

## Image Acquisition

Both differential interference contrast (DIC) and fluorescence microscopy were used for image acquisition. Imaging was performed using either a Zeiss LSM 880 confocal microscope with a Plan-Apochromat 40×/1.3 oil immersion or 20×/0.8 air objective, or a Zeiss Cell Discoverer 7 system equipped with a Plan-Apochromat 20×/0.95 objective and a Zeiss Axiocam 506 mono camera. This combination of high-resolution DIC and fluorescence channels enabled accurate visualisation of cell morphology and subcellular structures.

---

## Annotation Protocol

Ground-truth masks were manually annotated using **ImageJ (v1.54p, NIH)** with the **Labkit** plugin.

### Annotation Process

- Each cell boundary was manually traced using the Draw tool for precise instance-level segmentation.
- Both nucleus (Hoechst) and cytoplasm (GFP) fluorescence channels were used to assist annotation.
- Semi-automated tools (e.g. AnnotatorJ with U-Net backbones) were evaluated, but manual annotation was preferred to avoid algorithmic bias.
- Initial annotations were performed by an expert researcher trained in cell annotation using Fiji.
- Final masks were reviewed and validated by a senior cell biologist to ensure accuracy and biological relevance.
- Masks were exported in TIFF format.

### Mask Format

- Single-channel 2D TIFF images.
- Background pixels are labelled as `0`.
- Each cell instance is assigned a unique integer ID.
- Instance IDs are consecutive starting from `1`.

| Pixel Value | Description     |
|-------------|-----------------|
| 0           | Background      |
| 1           | Cell instance 1 |
| 2           | Cell instance 2 |

---

## Key Challenges

mCellSeg includes several biologically and technically challenging scenarios for cell segmentation:

- High cell density and extensive cell-to-cell contact
- Indistinct intercellular boundaries caused by cell adhesion
- Weak contrast and blurred cell contours due to 3D cell structure and defocus
- Tiny and slender cellular structures easily missed by conventional models
- Background artefacts and imaging noise from staining debris
- Large variation in cell morphology, size, and confluency

These characteristics make the dataset suitable for evaluating robust and generalisable segmentation algorithms.

---

## Applications

The dataset can be used for:

- Cell instance segmentation
- Semantic segmentation
- Semi-supervised and weakly supervised learning
- Benchmarking microscopy segmentation models
- Training foundation-model-based segmentation methods

---

## Licence

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.

https://creativecommons.org/licenses/by/4.0/

---

## Citation

If you use mCellSeg in your research, please cite:

```bibtex
@article{alam2026mcellseg,
  title={Cell segmentation in microscopy images using a SAM-based U-Net architecture and a novel dataset},
  author={Alam, Md. Shariful and Jackson, Miriam and Lord, Megan and Meijering, Erik},
  journal={Computer Methods and Programs in Biomedicine},
  year={2026},
  note={Under review}
}
```

---

## Acknowledgements

This work was supported by the UNSW RNA Institute and the ARC Future Fellowship (FT220100092) awarded to M. Lord.
