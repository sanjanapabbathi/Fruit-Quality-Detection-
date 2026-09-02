import cv2
import numpy as np

IMAGE_PATH= "rotten_apple.jpg"
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    print("Check the image name and location.")
    exit()

image = cv2.resize(image, (600, 600))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Detect dark/brown spots
lower = np.array([0, 40, 0])
upper = np.array([40, 255, 140])

mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

affected_area = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 100:
        affected_area += area

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

total_area = image.shape[0] * image.shape[1]

affected_percentage = (affected_area / total_area) * 100

if affected_percentage < 1:
    result = "FRESH APPLE"
elif affected_percentage < 3:
    result = "SLIGHTLY SPOILED"
else:
    result = "ROTTEN APPLE"

cv2.putText(
    image,
    result,
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imwrite("output.jpg", image)

print("FRUIT QUALITY DETECTION")
print("Affected Area:", round(affected_percentage, 2), "%")
print("Result:", result)

cv2.imshow("Fruit Quality Detection", image)
cv2.imshow("Detected Spots", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
