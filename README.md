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

## License

MIT License
