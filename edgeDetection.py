#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2

class edgeDetection:

    def auto_canny(self,image):
        med = np.median(image)
        lower = int(max(0,(1.0 - 0.33) * med))
        upper = int(max(0,(1.0 + 0.33) * med))
        edged = cv2.Canny(image,lower, upper)
        return edged


    def show_image(self,image_path):
        orig_image = cv2.imread(image_path)
        graying = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(graying,(3,3), 0)
        wide_image  = cv2.Canny(blurred,10,200)
        tight_image = cv2.Canny(blurred,225,250)
        auto_image = self.auto_canny(blurred)
        cv2.imshow("Original",orig_image)
        cv2.imshow("Stack",np.hstack([wide_image, tight_image, auto_image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = input("Enter the image path")
    edgeObj = edgeDetection()
    edgeObj.show_image(image_path)
