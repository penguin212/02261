import cv2
import os

for img in sorted(os.listdir("images/cam_3")):
    print(img)
    image = cv2.imread("images/cam_3/" + img)
    cv2.imshow("e", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()