# GCA Detection from TAB Slides

A Deep Learning -based approach to automate the detection of Giant Cell Arteritis from digital pathology slides of Temporal Artery Biopsy.

## Drive Directories

### ND-KD Shared Workspace

1. Drive Root: [GCADetection-TABSlides](https://drive.google.com/drive/folders/1_RQTxfbj7Awx1GhnnLKy6KeTRhale__2?usp=sharing)
2. [Progress Tracking and Working Notes](https://docs.google.com/document/d/1EI5U-VP_N0la0jteKMeJGxWmzn7jbAj8no2Khai4MwM/edit)      
3. [Granularity verification ROI Samples to finalize downscale factor with ma'am](https://drive.google.com/drive/folders/1w94vfqY0z4Gr2lwi8LnAQiKYCEJDIUHV?usp=sharing)

### Shared by Naveena ma'am

1. [GCA_project](https://drive.google.com/drive/folders/1f4Iwodhixomwwb4sxPjJ3PCHW2382mNQ?usp=sharing) - Metadata and Labeling Spreadsheets 
2. [TAB_Opthananology](https://drive.google.com/drive/folders/1Oxh3VMHT2IRmN4J1q8ZTAE7CaUb1QOGj?usp=sharing) - Sample .SVS files for testing scripts 
3. [New Files](https://drive.google.com/drive/folders/1wNMkBg7kh8HdLntc05PEawvz-KLSkGQh?usp=sharing) - Erroneous ROI extracts on ma'am's system

## Documentation

1. [Rendering .SVS files](./docs/01-render_svs_files.md)
2. [ROI Extraction](./docs/02-roi_extraction_procedure.md)
3. [ROI Annotation](./docs/03-annotation_procedure.md)
4. [Progress Report (Updated: September '22)](https://docs.google.com/document/d/14vXHEkumhzXJesqY8LUNyQd2O7FVP_vWxGg40SNi9_Q/edit?usp=sharing)
5. [GradCAM Visualization Results - Slides](https://docs.google.com/presentation/d/1GQD0_uGlFD01MreCIrYuCU_YlGyblGtRPHYrLJ8xOL8/edit?usp=sharing)
6. [Manuscript Draft](https://docs.google.com/document/d/1Wkmae8R-DypWB5Hv9i0qM-ujm_lL0Q6_02oDFgkkQCk/edit?usp=sharing)

## Progress Tracking

1. Script for rendering .SVS files 
   - [X] setup the openslide environment
   - [X] modify the wsi viewer python script
   - [X] setup the configuration file paths for the .svs files rendering cmd tool
   - [X] documentation to use scripts
2. Conversion of .SVS to .TIFF files
   - [X] adapt the [IBM](https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/) code for file format conversion 
   - [X] retain original file names post conversions
   - [X] try both lossless and lossy conversions
   - [X] perform 16x, 8x & w/o downscaling and summarize the [results](https://docs.google.com/document/d/1EI5U-VP_N0la0jteKMeJGxWmzn7jbAj8no2Khai4MwM/edit#heading=h.awk4rggu1nqh) (update ma'am immediately)
   - [X] Granularity verification - choose `downscale factor=4`
   - [X] only retain the unflitered and roi (black background) files
   - [X] documentation to use scripts
3. Prepare scripts to pick each ROI and filename_01, filename_02, ... (segmentation)
   - [X] extract roi
   - [X] setup the configuration file paths for the .svs files rendering cmd tool
   - [X] documentation to use scripts
4. Labelling
   - [X] procedure practice as instructed by ma'am - `rotate right (once clockwise) and number top to bottom, left to right`
   - [X] track uploads and downloads on the drive
   - [X] create sheets for roi labelling and progress tracking
   - [X] documentation to use scripts
   - [X] complete manual for part 1 labelling using the annotation sheet generated in the previous step
   - [X] complete manual for part 2 labelling using the annotation sheet generated in the previous step
   - [X] check for duplicates between ND and KD
   - [X] sort out annotation-related issues with ma'am for part 1 annotations.
   - [ ] sort out annotation-related issues with ma'am for part 2 annotations.
   - [ ] upload annotations to the drive after fixing.  ([Drive location for fixed files](https://drive.google.com/drive/folders/184X-4lbfwLuAIAqPI7ZzVPlySRtxcwgP?usp=share_link))
   - [ ] update spreadsheet for fixed files. ([spreadsheet to track fixes](https://docs.google.com/spreadsheets/d/1KolmDlzGPSAhI4W-2FgJwvc56n7bZLdoJA1Y9zhH_Vg/edit#gid=997139375))
   - [X] collect statistics from ma'am on the data - total no. of files, lost, re-upload, etc.
   - [ ] merge the mapping excel into one sheet
   - [ ] find and list missing slides - send list to ma'am
   - [ ] verify labelled dataset with Naveena ma'am
   - [ ] store annotations on hardrive and send post to Naveena ma'am
5. Deep Neural Network
   - [X] Balancing dataset
   - [X] ResNet18 Model
   - [X] GradCAM
