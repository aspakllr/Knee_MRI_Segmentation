Diploma thesis, titled "Knee joint segmentation based on deep learning and label
fusion techniques" , developed for the Biomedical Engineering masters program, University of Patras.

The segmentation process is guided by three components:
- The first part includes a deformable registration process that maps the
images, to N different  subspaces, thus creating N
”different” datasets with reduced anatomical variability compared to the original dataset.

- The second part includes N segmentation networks that are trained on the N subspaces.
The process produces N different segmentations for each test image.

- The final part combines the N predictions together through a label fusion algorithm to
obtain a final segmentation map for each test image.
