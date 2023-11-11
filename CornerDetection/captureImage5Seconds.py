import cv2
import time

# Create a VideoCapture object (0 represents the default camera, change if necessary)
cap = cv2.VideoCapture(0)

# Set a delay of 5 seconds (5000 milliseconds)
delay = 5000

# Infinite loop to continuously capture images
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Captured Image', frame)

    # Save the captured image with a timestamp as the filename
    timestamp = int(time.time())
    image_filename = f'captured_image_{timestamp}.jpg'
    cv2.imwrite(image_filename, frame)
    print(f'Image saved as {image_filename}')

    # Wait for 5 seconds (5000 milliseconds)
    cv2.waitKey(delay)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
