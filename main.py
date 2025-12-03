import cv2
import numpy as np

image_file = "./all-signs.png"

img = cv2.imread(image_file)
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



img = cv2.imread(image_file)
img = cv2.resize(img, (800, 600))  # optional, just to make sizes stable

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Example: RED color range (red is split into two ranges in HSV)
lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 80])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2   # combine

cv2.imshow("red mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


kernel = np.ones((5, 5), np.uint8)

mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small blobs
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)  # fill small holes

cv2.imshow("clean mask", mask_clean)
cv2.waitKey(0)
cv2.destroyAllWindows()



contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:  # ignore tiny things
        continue

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Shape analysis — circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # circularity ~1 => circle. You can play with this threshold
    if circularity > 0.5:  
        # draw bounding box for "detected sign"
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, "SIGN", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("detected", output)
cv2.waitKey(0)
cv2.destroyAllWindows()