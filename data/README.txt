data/preCropNet.h5: 
3499 sampled frames from 168 SWF415/ SWF702/ SWF1088 hermaphoridtes across prj_kfc, prj_5ht, prj_rim, prj_starvation and prj_aversion. 
Head positions are uncropped to match the 322x210 dimension. 
~30 frames are sampled from each animal
See notebooks/generate_preCropNet_data_from_data_dict.ipynb for julia code that generated preCropNet.h5.

data/immob_preCropNet.h5:
381 sampled frames from a subset of immobilized multichannel images across prj_kfc, prj_5ht, prj_rim, prj_starvation and prj_aversion.
Head positions are uncropped to match the 322x210 dimension. 
Only t=30 (i.e. 1 frame) is taken to avoid redundancy since all immobilized images are functionally identical.
See notebooks/generate_preCropNet_immobilized_data_from_data_dict.ipynb for julia code that generated immob_preCropNet.h5.

data/preCropNet_cropped.h5 and data/immob_preCropNet_cropped.h5: 
All images and labels from preCropNet.h5 and immob_preCropNet.h5 were passed through CropNet to yield in the standard dimension of (284,120).
See notebooks/CropNet_2D.ipynb for julia code that generated preCropNet_cropped.h5 and immob_preCropNet_cropped.h5.

====================================================

data/postCropNet.h5:
1675 sampled frames from 24 heterogeneous animals -- 5 SWF360 hermaphrodites, 1 SWF467 hermaphrodite, 3 SWF702 hermaphrodites, 7 SWF1212 hermaphrodites and 6 SWF1088/ SWF1212 males.
Among them, 13 animals were imaged with Zyla4.2P, and 11 animals were images with Hamamatsu Orca Fusion BT2.
~80 frames are sampled from each animal, with both positive and negative examples.
Labels come directly from ImageClick outputs.
Both images and labels are in standard dimension (284, 120) from the get-go.
See notebooks/generate_postCropNet_data_from_imageClick.ipynb for python code that generated postCropNet.h5.

data/immob_postCropNet.h5:
30 sampled frames from a subset of immobilized multichannel images, with the vast majority imaged with Hamamatsu Orca Fusion BT2. 
Only t=30 (i.e. 1 frame) is taken to avoid redundancy since all immobilized images are functionally identical.
See notebooks/generate_postCropNet_immobilized_data_from_data_dict.ipynb for julia code that generated immob_postCropNet.h5.

====================================================

Data that go into network training/ validation/ testing are all in the standard dimension of (284,120):
data/preCropNet_cropped.h5
data/immob_preCropNet_cropped.h5
data/postCropNet.h5
data/immob_postCropNet.h5

Total 5585 unique image-label pairs from 192 animals.