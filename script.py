import cv2

#load image
sample_image = cv2.imread('ptf1.jpg')
target_image = cv2.imread('ptf10.jpg')

hist_sample_image = cv2.calcHist([sample_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
# hist_sample_image[255, 255, 255] = 0 #ignore all white pixels
cv2.normalize(hist_sample_image, hist_sample_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_target_image = cv2.calcHist([target_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
# hist_target_image[255, 255, 255] = 0  #ignore all white pixels
cv2.normalize(hist_target_image, hist_target_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Find the metric value
metric_val = cv2.compareHist(hist_sample_image, hist_target_image, cv2.HISTCMP_CORREL)

# Print the metric value
print(f"Similarity Score: ", round(metric_val, 2))