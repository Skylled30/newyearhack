import cv2
import numpy as np

#cap = cv2.VideoCapture(0)

image = cv2.imread('14.jpg')

cv2.namedWindow('Pich', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Persp', cv2.WINDOW_KEEPRATIO)
#cv2.namedWindow('Finish', cv2.WINDOW_KEEPRATIO)

positions = []
trans = np.array([[0, 0], [1080, 1920]], dtype='float32')

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global positions
        if len(positions) < 4:
            positions += [[x, y]]

cv2.setMouseCallback("Pich", on_mouse_click)

while True:
    if len(positions) == 4:                
        rect = np.zeros((4, 2), dtype = "float32")
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
                	[0, max_height - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        img_persp = cv2.warpPerspective(image, M, (max_width, max_height))
        cv2.imshow('Persp', img_persp)
        cv2.putText(image, "Positions: {}".format(positions), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    
    cv2.imshow('Pich', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



cv2.destroyAllWindows()
#cv2.imshow('Contours',)

