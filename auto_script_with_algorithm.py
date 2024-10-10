import cv2

imageList = ['ptf2.jpg', 'ptf3.jpg', 'ptf4.jpg', 'ptf5.jpg', 'ptf6.jpg', 'ptf7.jpg', 'ptf8.jpg', 'ptf9.jpg', 'ptf10.jpg']
metric_value_list = []

# load the sample image
sample_image = cv2.imread('ptf1.jpg')

# calc hist of sample image and normalize
hist_sample_image = cv2.calcHist([sample_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
cv2.normalize(hist_sample_image, hist_sample_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

def averagingList(lst):
    return sum(lst) / len(lst)

for x in imageList:
    # load the target image
    target_image = cv2.imread(x)

    # calc hist of target image and normalize
    hist_target_image = cv2.calcHist([target_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_target_image, hist_target_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Find the metric value
    metric_val = cv2.compareHist(hist_sample_image, hist_target_image, cv2.HISTCMP_CORREL)
    metric_value_list.append(metric_val)

    # print the metric value
    print(f"\nSimilarity Score with base image of ptf1.jpg compared to {x}: ", round(metric_val, 2), "\n")

print(f'Average Similarity Score with base image of ptf1.jpg compared to ptf2.jpg until ptf10.jpg: ', round(averagingList(metric_value_list)*100, 2), '%')


