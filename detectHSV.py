import cv2
import numpy as np

def load_image(path, size=(800, 600)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("❌ Image could not be loaded!")
    return cv2.resize(img, size)

def get_hsv_range_from_roi(img):
    # OpenCV built-in ROI selector
    x, y, w, h = cv2.selectROI("Select Area", img)

    if w == 0 or h == 0:
        print("❌ No area selected!")
        return None, None

    roi = img[int(y):int(y+h), int(x):int(x+w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Min & Max HSV in ROI
    lower = np.array([
        hsv[:,:,0].min(),
        hsv[:,:,1].min(),
        hsv[:,:,2].min()
    ])

    upper = np.array([
        hsv[:,:,0].max(),
        hsv[:,:,1].max(),
        hsv[:,:,2].max()
    ])

    # Show selected region for confirmation
    cv2.imshow("Selected ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lower, upper


# ---- RUN ----
img = load_image("./photos/istanbul.png")

lower, upper = get_hsv_range_from_roi(img)

if lower is not None:
    print("\n🎯 HSV Range for selected area:")
    print("Lower HSV:", lower)
    print("Upper HSV:", upper)
