import cv2
import numpy as np
import math

# cap = cv2.VideoCapture(0)

# image = cv2.imread('14.jpg')

cv2.namedWindow('Pich', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Persp', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Cont_Persp', cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('Finish', cv2.WINDOW_KEEPRATIO)

positions = []
trans = np.array([[0, 0], [1080, 1920]], dtype='float32')

cap = cv2.VideoCapture(1)


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global positions
        if len(positions) < 4:
            positions += [[x, y]]


cv2.setMouseCallback("Pich", on_mouse_click)

while cap.isOpened():
    image = cap.read()[1]
    if len(positions) == 4:
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = positions[0]
        rect[1] = positions[1]
        rect[2] = positions[2]
        rect[3] = positions[3]

        widthA = np.sqrt(((rect[1][1] - rect[0][1]) ** 2) + ((rect[1][0] - rect[0][0]) ** 2))
        widthB = np.sqrt(((rect[2][1] - rect[3][1]) ** 2) + ((rect[2][0] - rect[3][0]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
        heightB = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))

        max_width = int(max(widthA, widthB))
        max_height = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        img_persp = cv2.warpPerspective(image, M, (max_width, max_height))

        gray_persp = cv2.cvtColor(img_persp, cv2.COLOR_BGR2GRAY)
        img_persp = cv2.GaussianBlur(img_persp, (5, 5), 0)
        # gray_persp = cv2.Laplacian(gray_persp, 3)
        canny_persp = cv2.Canny(img_persp, 1, 100)

        contours = cv2.findContours(canny_persp.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]

        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        square_length = int(math.sqrt(max_area))

        space_length = int(square_length * 0.203125 * 1.35)
        width_count = int(max_width / (square_length + space_length))
        height_count = int(max_height / (square_length + space_length))

        for i in range(1, width_count):
            cv2.line(img_persp, (i*(square_length+space_length), 0), (i*(square_length+space_length), max_height), (0, 255, 0), 3)

        for contour in contours:
            if (cv2.contourArea(contour) + max_area * 0.5) >= max_area:
                cv2.drawContours(img_persp, [contour], 0, (255, 0, 255), 3)

        cv2.imshow('Persp', canny_persp)
        cv2.imshow('Cont_Persp', img_persp)
        cv2.putText(image, "Positions: {}".format(positions),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow('Pich', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
# cv2.imshow('Contours',)
