a structured Python script that detects colored circles in an image, extracts their color values in RGB/HSV format, compares them with reference colors, and applies enhancements if necessary using KMeans clustering and HSV adjustments.

pip **install** numpy opencv-python scikit-learn

### Explanation

* **Detecting Circles** : The `detect_colored_circles` function uses Hough Transform to find circles in the image and retrieves their colors in both RGB and HSV formats.
* **Comparing Colors** : The `compare_with_reference` function checks if the detected circle colors match the provided reference colors. If a match is not found, it applies KMeans to find the dominant color and adjusts it.
* **Enhancing the Image** : The `enhance_image` function modifies the original image by enhancing the colors of circles that didn't match the reference values and saves the new image.

### Usage

* Replace `'input_image.png'` with the path to your image.
* Update the `reference_colors` with your actual reference color values in RGB format.

### Note

* You may need to adjust the parameters for sensitivity and enhancement factors depending on the specifics of your use case.

P237C - #EC86D0

P218C - #E56DB1

P310C - #6AD1E3

P301C - #004B87

# Reference Colors list of Strips circular objects (Top to Bottom) and

# Mobile clicked cropped images paths

P237C - 35 - #FBD5FF
RGB -- (251, 213, 255)
HSV -- (294, 16.5, 100)
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P237C_35.png

P237C - 55 - #F8B9FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P237C_55.png

P237C - 70 - #F3A0FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P237C_70.png

P237C - 85 - #EC88FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P237C_85.png

P237C - 100 - #E578FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P237C_100.png

P218C - 70 - #F17AFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P218C_70.png

P218C - 80 - #EC6AFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P218C_80.png

P218C - 90 - #E75BFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P218C_90.png

P218C - 100 - #E14BFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P218C_100.png

P310C - 35 - #C3FFED
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P310C_35.png

P310C - 50 - #ACFFE8
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P310C_50.png

P310C - 65 - #98FFE1
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P310C_65.png

P298C - 65 - #88E9F1
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P298C_65.png

P298C - 85 - #6AE1EE
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P298C_85.png

P298C - 100 - #52DAEC
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P298C_100.png

P301C - 65 - #468FCD
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P301C_65.png

P301C - 80 - #216EC3
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P301C_80.png

P298C - 100 - #003E98
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/list_of_color_images/P301C_100.png



I have attached one folder, in that , 3 files are present. 2 reference images  (for complex task , as mention below, use ref_shadow_img.png image and then for image for normal use case)  & one .json file. The actual input image & refernce color values --> list of json data in .json file. For image, Only focus will be in middle black strip. It has middle black color band, on which we can see 18 circular shaped color objects and side by side the squared color objects (the sequence is --> Left Top-To-Bottom then Right Top-To-Bottom).

User can capture this strip using his/her mobile or tab camera. The task would be as follows,

1. Firstly detect the image position (whole black strip including circular & squared color objects) correctly or not (means , is that image rotated or tilted or any worngly captured image position) -- use two triangles (white-present black band & black-present just above of black band i.e. in white part) only present in image. Ensure it the follow step no 2 as below.
2. Detect all 18 circles objects with respective (aside) 18 squared color objects, all these combined 36 objects must be detected acurately according to its shape.
3. Then detect shadows (soft as well as hard / dark shadow area) present on strip or not.
4. Then detect any other object covering / cuting strips any part.

After that we need to develop fully functional python project to achive below objectives,

1. We need to detect all circular colored objects at first step (only which are along side on black strip only , dont consider any other circular object if available in an image). Get each circular objects' dominant color independently, store that value somewhere in variable / in-memory. Such process can be done for remaining 17 circular color objects as well.
2. Then We have refernce color values (squential color values as mention above strips color position) in .json file or using that .json file you can create new static python object as per your need. Using this reference color values , we must compair with extracted & stored dominant color values of each circular object's itself independently.
3. Suppose , if that extracted dominant color & respective circular objects reference color not matching (it may differ with higher or lower color values of HSV / RGB whatever) , then calculate scaling factor of that cicular object so that refernce color value must be applied (achieved) on that particular cirular object only , not any other object or globally of that entire image. So in such way, we need to use "each circular color objects scaling factor" by which we can use this "Scaling Factor value" to enhance or reduce-enhancement of "Respective Squared color object Only (The Squence of respective Circle-To-Square Objects are Left Top-To-Bottom & then Right Top-To-Bottom only)".
4. So as per step no 3 , we must follow same process endependently on each total 18 circular objects only , then get dominant color of each circles, then compair with respective refernce color , then get scalling factor independently , then using that scalling factor apply only on aside "Squared color objects (with respective position of circular objects only, not on entire image.)".
5. In above all tasks if you need image croping / cutting any section of source image , for better accuracy - otpimization or efficiency , you can do it.

The Objective is that, users can upload clicked images wrongly from their smart phones or mobile or tab any camera. So we need to ensure that, image must be clear, shadow free, any objstacle free (by which circular plus suqare objects) must be visible with their respective colors. Because tha image will go through further lab processing, report genration and further evaluation on the basis on colors detected in "Squared Objects". So we need to enhance or reduce that source image (as per image quality & position & object detection, whatever the case is).

In this Python project, proper project structure, modular , scalable coding style etc along with best suited design patterns & following SOLID principles and best optimization techniques must be followed.


medium size circular object detection then extracting that objects color , then matching with reference dominant color value , then if that color not matching each other then using image color enhancement or reduction technique applying on that detected circular part only.

### Approach

1. **Circular Object Detection:**

* The first step is to detect the circular object within the image.
* Methods like circle Hough Transform (CHT) or OpenCV's `cv2.findCircles` can be used for this.
* This involves identifying the object's boundaries and extracting the region of interest (ROI).

2. **Color Extraction:**

* Once the circular object is detected, its color needs to be extracted.
* This can be done by averaging the RGB (or other color space) values within the ROI.
* Alternatively, a dominant color detection algorithm can be used to find the most representative color within the object.

3. **Color Comparison and Matching:**

* The extracted color is then compared with a reference dominant color value.
* This comparison can be done using various color difference metrics like Euclidean distance or CIEDE2000.

4. **Color Enhancement/Reduction (if needed):**

* If the extracted color doesn't match the reference color, color enhancement or reduction techniques are applied.
* This could involve adjusting the hue, saturation, or value (HSV) values.
* Or, applying filters like contrast enhancement, color correction, or color balance.

5. **Applying to the Detected Area:**

* The color adjustment should be applied specifically to the detected circular region.
* This is achieved by masking the image with the detected circular shape.
