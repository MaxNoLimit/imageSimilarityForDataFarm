import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

imageList_ptf = ['ptf2.jpg', 'ptf3.jpg', 'ptf4.jpg', 'ptf5.jpg', 'ptf6.jpg', 'ptf7.jpg', 'ptf8.jpg', 'ptf9.jpg', 'ptf10.jpg']
imageList_pts = ['pts2.jpg', 'pts3.jpg', 'pts4.jpg', 'pts5.jpg', 'pts6.jpg', 'pts7.jpg', 'pts8.jpg', 'pts9.jpg', 'pts10.jpg']
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1
def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score


print("\nSimilarity between same ptf images")
for x in imageList_ptf:
    print(f"Similarity Score with base image of ptf1.jpg compared to {x}: ", round(generateScore('ptf1.jpg', x), 2))

print("\nSimilarity between ptf and pts images (cross)")
for x in imageList_pts:
    print(f"Similarity Score with base image of ptf1.jpg compared to {x}: ", round(generateScore('ptf1.jpg', x), 2))

print("\nSimilarity between same pts images")
for x in imageList_pts:
    print(f"Similarity Score with base image of pts1.jpg compared to {x}: ", round(generateScore('pts1.jpg', x), 2))

print("\nSimilarity between pts and ptf images (cross)")
for x in imageList_ptf:
    print(f"Similarity Score with base image of pts1.jpg compared to {x}: ", round(generateScore('pts1.jpg', x), 2))
