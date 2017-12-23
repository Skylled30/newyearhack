import cv2
import numpy as np
import math

colors_arr = [
    {
        "type": "red",
        "colors": {
            "min": [164, 119, 67],
            "max": [184, 139, 87]
        }
    },
    {
        "type": "yellow",
        "colors": {
            "min": [8, 183, 114],
            "max": [28, 203, 134]
        }
    },
    {
        "type": "green",
        "colors": {
            "min": [70, 93, 53],
            "max": [100, 113, 73]
        }
    }
]
positions = []
sqrt_mass = []
trans = np.array([[0, 0], [1080, 1920]], dtype='float32')


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global positions
        if len(positions) < 4:
            positions += [[x, y]]


def transform_image(image: np.ndarray, positions: list) -> (bool, np.ndarray):
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
        return True, img_persp
    return False, None


def get_colors(rect: np.ndarray, colors) -> list:
    result = list()
    height = rect.shape[0]
    width = rect.shape[1]
    for i in range(2):
        top = height * i + height // 4
        bottom = height * (i + 1) - height // 4
        for j in range(2):
            left = width * j + width // 4
            right = width * (j + 1) - width // 4
            color_mat = rect[top:bottom, left:right]
            print("_______")
            print(color_mat)
            print("-------")
            color_mat = [
                color_mat[0].mean(),
                color_mat[1].mean(),
                color_mat[2].mean()
            ]
            print(color_mat)
            print("_______")
            for color in colors:
                min_colors = color["colors"]["min"]
                max_colors = color["colors"]["max"]
                print(min_colors)
                print(max_colors)
                print(color_mat)

                if (
                        min_colors[0] < color_mat[0] < max_colors[0] and
                        min_colors[1] < color_mat[1] < max_colors[1] and
                        min_colors[2] < color_mat[2] < max_colors[2]
                ):
                    result += color["type"]
    if len(result) != 4:
        return []
    return result


def handle_image(image: np.ndarray):
    global colors_arr
    succeed, img_persp = transform_image(image, positions)

    if succeed:
        img_persp = cv2.GaussianBlur(img_persp, (5, 5), 0)
        canny_persp = cv2.Canny(img_persp, 1, 100)

        contours = cv2.findContours(canny_persp.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]

        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        square_length = int(math.sqrt(max_area))
        if len(sqrt_mass) < 20:
            sqrt_mass.append(square_length)
        else:
            max_width = img_persp.shape[1]
            max_height = img_persp.shape[0]
            square_length = int(np.array(sqrt_mass).mean())

            space_length = int(square_length * 0.203125 * 1.2)
            width_count = int(max_width / (square_length + space_length))
            height_count = int(max_height / (square_length + space_length))

            for i in range(1, width_count + 1):
                left = (i - 1) * (square_length + space_length)
                right = i * (square_length + space_length)
                for j in range(1, height_count + 1):
                    top = (j - 1) * (square_length + space_length)
                    bottom = j * (square_length + space_length)

                    rect = cv2.cvtColor(img_persp[top:bottom, left:right], cv2.COLOR_BGR2HSV)

                    colors = get_colors(rect, colors_arr)
                    print(i, j, colors)

                cv2.line(img_persp, (i * (square_length + space_length), 0),
                         (i * (square_length + space_length), max_height), (0, 255, 0), 3)

        for contour in contours:
            if (cv2.contourArea(contour) + max_area * 0.5) >= max_area:
                cv2.drawContours(img_persp, [contour], 0, (255, 0, 255), 3)

        cv2.imshow('Persp', canny_persp)
        cv2.imshow('Cont_Persp', img_persp)
        cv2.putText(image, "Positions: {}".format(positions),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow('Pich', image)


def main():
    cv2.namedWindow('Pich', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Persp', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Cont_Persp', cv2.WINDOW_KEEPRATIO)

    cap = cv2.VideoCapture(1)
    cv2.setMouseCallback("Pich", on_mouse_click)

    while cap.isOpened():
        image = cap.read()[1]

        handle_image(image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
