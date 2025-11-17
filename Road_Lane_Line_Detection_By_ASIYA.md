```python
import cv2
import numpy as np

video_path = "test1.mp4"  # Your input video
cap = cv2.VideoCapture(video_path)

def region_of_interest(image):
    """Focus only on the lower part of the road."""
    height = image.shape[0]
    polygons = np.array([
        [(100, height), (image.shape[1]-100, height), (image.shape[1]//2, int(height*0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    """Draw detected lines."""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0:  # Avoid division by zero
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []
    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        averaged_lines.append(make_coordinates(image, left_fit_avg))
    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        averaged_lines.append(make_coordinates(image, right_fit_avg))
    return averaged_lines

# --- Optional smoothing memory (keeps last few frames)
memory = {"left": None, "right": None}
smooth_factor = 5  # smaller = more responsive, larger = smoother

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video finished — closing window.")
        break  # <-- stops the loop when the video ends

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)

    final_lines = []
    if averaged_lines:
        for line in averaged_lines:
            final_lines.append(line)

    line_image = display_lines(frame, final_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Smooth Road Line Detection", combo_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("❌ Quit key pressed.")
        break

cap.release()
cv2.destroyAllWindows()

```


```python

```
