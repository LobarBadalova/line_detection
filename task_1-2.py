# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cap1.png')


def canny_edge_detector(image):
    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, height), (640, 500), (1280, height)]
    ])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (4 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if len(right_fit) == len(left_fit) == 0:
        return np.array([0])
    if len(left_fit) == 0:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_avg)
        return np.array([right_line])
    elif len(right_fit) == 0:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_avg)
        return np.array([left_line])

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    s = left_fit_average[0] + right_fit_average[0]

    if s > 0.15:
        print("Right")
        cv2.putText(image, 'Right', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif s < -0.8:
        print("Left")
        cv2.putText(image, 'Left', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        print("Forward")
        cv2.putText(image, 'Forward', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None and len(lines[0]) == 4:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    else:
        x1 = 1
        y1 = 1
        x2 = 2
        y2 = 2
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 30)
    '''lx1 = lines[0][0]
    ly1 = lines[0][1]
    lx2 = lines[0][2]
    ly2 = lines[0][3]

    rx1 = lines[1][0]
    ry1 = lines[1][1]
    rx2 = lines[1][2]
    ry2 = lines[1][3]
    lslope = (ly1 - ly2) / (lx1 - lx2)
    rslope = (ry1 - ry2) / (rx1 - rx2)
    sslope = lslope + rslope

    if sslope > 0.15:
        print("Right")
        cv2.putText(line_image, 'Right', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif sslope < -0.8:
        print("Left")
        cv2.putText(line_image, 'Left', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("Forward")
        cv2.putText(line_image, 'Forward', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)'''
    return line_image


'''canny_image = canny_edge_detector(img)
cropped_image = region_of_interest(canny_image)

lines = cv2.HoughLinesP(cropped_image, 6, np.pi / 180, 50,
                            np.array([]), minLineLength=10,
                            maxLineGap=100)

averaged_lines = average_slope_intercept(img, lines)
line_image = display_lines(img, averaged_lines)
combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
#cv2.imshow("results", combo_image)
#cv2.waitKey(0)
plt.imshow(combo_image)
plt.show()'''

# Path of dataset directory
cap = cv2.VideoCapture("video.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny_edge_detector(frame)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 6, np.pi / 180, 100,
                            np.array([]), minLineLength=10,
                            maxLineGap=100)

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("results", combo_image)

    # When the below two will be true and will press the 'q' on
    # our keyboard, we will break out from the loop

    # # wait 0 will wait for infinitely between each frames.
    # 1ms will wait for the specified time only between each frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the video file
cap.release()

# destroy all the windows that is currently on
cv2.destroyAllWindows()
