# LUPA – Language Understanding Pathology Assistant

LUPA is a library for building a Language Understanding Pathology Assistant, focused on deep learning-based analysis of histopathology whole slide images (WSI).
The project aims to combine visual understanding of pathology scans with language-based reasoning to support diagnostic workflows.

## Project Milestones

- [x] Patch-level cancer type prediction
- [x] Whole slide analysis and prediction
- [x] Uncertain regions detection and spatial voting
- [ ] Language and visual feature merging - *(in progress)*
- [ ] Language assistant

## Example Usage

An example of how to use the library is provided in the `Example.ipynb` file.
The notebook demonstrates patch-level prediction, whole slide inference, and available utilities.

## Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

LUPA also requires OpenSlide for working with whole slide images.

Download and install OpenSlide from:
https://openslide.org/download/

Make sure OpenSlide is properly installed and available on your system.

## Requirements

- Python 3.x
- Packages listed in `requirements.txt`
- OpenSlide

## Contributing

Contributions, issues, and feature requests are welcome.

## Publication

This code is based mostly on a DiagSet experiments, if you find it useful, use can use:
```bibtex
@article{koziarski2024diagset,
  title={DiagSet: a dataset for prostate cancer histopathological image classification},
  author={Koziarski, Micha{\l} and Cyganek, Bogus{\l}aw and Niedziela, Przemys{\l}aw and Olborski, Bogus{\l}aw and Antosz, Zbigniew and {\.Z}ydak, Marcin and Kwolek, Bogdan and W{\k{a}}sowicz, Pawe{\l} and Buka{\l}a, Andrzej and Swad{\'z}ba, Jakub and others},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={6780},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## License

MIT License
