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
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P237C_35.png

P237C - 55 - #F8B9FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P237C_55.png

P237C - 70 - #F3A0FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P237C_70.png

P237C - 85 - #EC88FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P237C_85.png

P237C - 100 - #E578FF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P237C_100.png

P218C - 70 - #F17AFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P218C_70.png

P218C - 80 - #EC6AFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P218C_80.png

P218C - 90 - #E75BFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P218C_90.png

P218C - 100 - #E14BFF
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P218C_100.png

P310C - 35 - #C3FFED
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P310C_35.png

P310C - 50 - #ACFFE8
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P310C_50.png

P310C - 65 - #98FFE1
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P310C_65.png

P298C - 65 - #88E9F1
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P298C_65.png

P298C - 85 - #6AE1EE
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P298C_85.png

P298C - 100 - #52DAEC
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P298C_100.png

P301C - 65 - #468FCD
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P301C_65.png

P301C - 80 - #216EC3
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P301C_80.png

P298C - 100 - #003E98
/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/color_refs/P301C_100.png
