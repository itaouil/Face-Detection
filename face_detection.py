# Face Detection

# Import OpenCV
import cv2

# Load pre-trained classifiers (eyes and faces)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detection function
def detect(img_gray, img_color):

    # Apply Viola-Jones algorithm for face detection
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    # Iterate over detected faces
    for (fx, fy, fw, fh) in faces:

        # Draw rectangle on colored image for the face
        cv2.rectangle(img_color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        # Get face reference for eyes detection
        eref_gray = img_gray[fy:fy+fh, fx:fx+fw]
        eref_color = img_color[fy:fy+fh, fx:fx+fw]

        # Apply Viola-Jones algorithm for eye detection (face reference)
        eyes = eye_cascade.detectMultiScale(eref_gray, 1.1, 3)

        # Iterate over detected eyes
        for (ex, ey, ew, eh) in eyes:

            # Draw rectangle on colored image for the eyes
            cv2.rectangle(eref_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Return depicted image
    return img_color

""" Apply detection to real time stream """

# Get video stream
video_capture = cv2.VideoCapture(0) # Using internal camera, otherwise 1

# Detect
while True:

    # Read stream
    _, img_color = video_capture.read()

    # Convert to image to gray scaleqqq
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Call detect function
    canvas = detect(img_gray, img_color)

    # Paint canvas
    cv2.imshow("Video", canvas)

    # Break video stream (when q pressed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shut down stream
video_capture.release()
cv2.destroyAllWindows()
