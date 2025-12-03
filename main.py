import cv2
import numpy as np
from math import pi

# ---------- CONFIG ----------
IMAGE_PATH = "./photos/istanbul.png"
TARGET_MIN_AREA = 500
KERNEL_SIZE = (5, 5)
# ---------------------------


def load_image(path, size=(800, 600)):
    """Load and resize image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return cv2.resize(img, size)


def create_red_mask(img):
    """Convert to HSV and extract red color mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 80])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return mask1 | mask2
def create_green_mask(img):
    """Convert to HSV and extract green color mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([40, 100, 80])
    upper1 = np.array([90, 255, 255])


    mask1 = cv2.inRange(hsv, lower1, upper1)
    return mask1

def clean_mask(mask, kernel_size=KERNEL_SIZE):
    """Apply morphological OPEN & CLOSE filters to remove noise and fill holes."""
    kernel = np.ones(kernel_size, np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def is_circular_sign(contour):
    """Check if contour is approximately circular using circularity formula."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area == 0 or perimeter == 0:
        return False

    circularity = 4 * pi * (area / (perimeter * perimeter))
    return circularity > 0.5
def is_rectangular_sign(contour):
    """Check if contour is approximately rectangular using aspect ratio."""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    return 0.8 < aspect_ratio < 1.2  # Adjust thresholds as needed

def detect_signs(img, mask):
    """Extract contours and draw bounding boxes for detected signs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TARGET_MIN_AREA:
            continue

        if is_circular_sign(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "SIGN", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif is_rectangular_sign(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "SIGN", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output


def show_step(window_name, img):
    """Display image or mask result."""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- MAIN WORKFLOW ----------
img = load_image(IMAGE_PATH)

mask_red = create_red_mask(img)
mask_green = create_green_mask(img)

mask_red = clean_mask(mask_red)
mask_green = clean_mask(mask_green)

result_red = detect_signs(img, mask_red)
result_green = detect_signs(img, mask_green)

# Show all intermediate steps if needed:
show_step("Original", img)
show_step("Red Mask", mask_red)
show_step("Detected Red Signs", result_red)
show_step("Green Mask", mask_green)
show_step("Detected Green Signs", result_green)
