# GCA-Detection-TAB-Slides

Deep Neural Network to automate the detection of Giant Cell Arteritis from digital pathology slides of Temporal Artery Biopsy.

## Drive Directories

### ND-KD Shared Workspace

1. [GCADetection-TABSlides](https://drive.google.com/drive/folders/1_RQTxfbj7Awx1GhnnLKy6KeTRhale__2?usp=sharing)

### Shared by Naveena ma'am

1. [GCA_project](https://drive.google.com/drive/folders/1f4Iwodhixomwwb4sxPjJ3PCHW2382mNQ?usp=sharing)
2. [TAB_Opthananology](https://drive.google.com/drive/folders/1Oxh3VMHT2IRmN4J1q8ZTAE7CaUb1QOGj?usp=sharing)

## Working Notes

 Link to doc notes: [here](https://docs.google.com/document/d/1EI5U-VP_N0la0jteKMeJGxWmzn7jbAj8no2Khai4MwM/edit)      
 Link to sample ROIs: [here](https://drive.google.com/drive/folders/1w94vfqY0z4Gr2lwi8LnAQiKYCEJDIUHV?usp=sharing)
 
## Progress Tracking

1. IBM Code adapted:
    - [X] retain original file names post conversions
    - [X] perform 16x, 8x & w/o downscaling (update immediately)
    - [X] only retain the unflitered and roi (black background) files
2. Create a code to pick each ROI and filename_01, filename_02, ... (segmentation)
    - [X] extract roi
3. Labelling
    - [ ] Map from annotations file
    - [ ] Verify labelled dataset

## Procedure to Render .SVS files

Link to Render .svs files procedure: [here](./docs/01-render_svs_files.md)

## Procedure to Extract ROI

Link to ROI Extraction Procedure file: [here](./docs/02-roi_extraction_procedure.md)

## Procedure to Annotate Extracted ROIs

Link to Annotation Procedure file: [here](./docs/03-annotation_procedure.md)
