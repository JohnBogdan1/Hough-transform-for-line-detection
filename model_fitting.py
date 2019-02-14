from math import hypot, cos, sin, radians, ceil
from itertools import product
import numpy as np
import cv2


def hough_transform(image, ignored_color=255):
    image_x, image_y = image.shape

    # get diagonal length of the image
    ro_max = ceil(hypot(image_x, image_y))

    # theta is between [-90, 90] and ro is between [-ro_max, ro_max]
    # i shifted theta interval with +90 and ro interval with +ro_max so i can define a matrix and iterate
    acc_image_x, acc_image_y = (2 * ro_max, 180)

    H = np.zeros((acc_image_y, acc_image_x), dtype="int")

    # print(image.shape)
    # print(H.shape)
    edges = list(product(range(image_x), range(image_y)))

    for edge in edges:
        x, y = edge
        color = image[x, y]
        # skip white points in the image
        if color == ignored_color:
            continue
        for theta in range(180):
            # use radians for cos and sin
            theta_rad = radians(theta)
            ro = x * cos(theta_rad) + y * sin(theta_rad)

            # ro is between [-ro_max, ro_max], so i need to add ro_max
            shifted_ro = int(acc_image_y / 2 + ro)
            H[theta, shifted_ro] += 1

    return H


def find_max_point(H):
    height, width = H.shape
    max_votes = -1
    theta_ro = None
    for i in range(height):
        line = H[i]
        max_v = max(line)

        if max_v > max_votes:
            max_votes = max_v
            theta_ro = (i - 90, list(line).index(max_v) - width / 2)
        # print((i, list(line).index(max_v)), max_v)

    return theta_ro


def main():
    img = cv2.imread("grid.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply edge filter to get edges before transform
    # change ignored_color to 0 when using "edges", and to 255 when using "gray"
    edges = cv2.Canny(gray, 75, 150)
    accumulator = hough_transform(edges, ignored_color=0)
    point = find_max_point(accumulator)

    print("(theta, ro): ", point)
    cv2.imwrite("H.png", accumulator)


if __name__ == "__main__":
    main()
