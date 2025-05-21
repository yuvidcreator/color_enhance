from PIL import Image
from ultralytics import YOLO


model = YOLO("pt_models/best.pt")
# result = model.val(data="dataset/data.yaml")
# results = model("/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/refernce_image.jpg")

# for result in results:
#     print(result.boxes)

# model.predict(
#     "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/refernce_image.jpg", 
#     save=True,
#     imgsz=640,
#     conf=0.5
# )  # save predictions as .txt and .jpg files

# Run inference on 'refernce_image.jpg'
results = model(
    [
        "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/AiGenomics_Projects/Welleys_Image_Analysis/Color_Enhancing/reference_data/refernce_image.jpg"
    ]
)  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"output_data/results{i}.jpg")