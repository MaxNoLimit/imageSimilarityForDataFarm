import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image
img = cv2.imread('Potret pointToSelf/pts1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 100  # Adjust as needed
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

convex_contours = [cnt for cnt in filtered_contours if cv2.isContourConvex(cnt)]

epsilon = 0.01  # Adjust as needed
approx_contours = [cv2.approxPolyDP(cnt, epsilon * cv2.arcLength(cnt, True), True) for cnt in convex_contours]

for cnt in approx_contours:
    # Draw contour
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

    # Calculate centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Draw vector from centroid to each vertex
    for pt in cnt:
        x, y = pt[0]
        cv2.line(img, (cx, cy), (x, y), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()