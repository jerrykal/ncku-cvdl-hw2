import cv2
import numpy as np


def find_tracking_point(image, draw=True):
    """Draw a tracking point on an image."""
    if image is None:
        return

    # Make a copy of the image
    image = image.copy()

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect tracking point
    point = cv2.goodFeaturesToTrack(
        gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7
    )

    if draw:
        x, y = point.ravel().astype(int)

        # Draw a red cross at tracking point
        cv2.line(image, (x - 10, y), (x + 10, y), (0, 0, 255), 4)
        cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 255), 4)

        # Display image
        cv2.imshow("Preprocessing", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return point


def video_tracking(frames):
    """Track a point in a video and draw its trajectory."""
    if len(frames) == 0:
        return

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_point = find_tracking_point(frames[0], draw=False)

    # Create empty array to store points
    points = []

    for frame in frames:
        # Make a copy of the frame
        frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        point, _, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_point, None  # type: ignore
        )

        # Store points
        points.append(point)

        # Draw tracking point
        x, y = point.ravel().astype(int)
        cv2.line(frame, (x - 10, y), (x + 10, y), (0, 0, 255), 4)
        cv2.line(frame, (x, y - 10), (x, y + 10), (0, 0, 255), 4)

        # Draw trajectory
        overlay = frame.copy()
        for i in range(1, len(points)):
            cv2.line(
                overlay,
                points[i - 1].ravel().astype(int),
                points[i].ravel().astype(int),
                (0, 255, 255),
                4,
            )

        # Add overlay to frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display frame
        cv2.imshow("Video Tracking", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        # Update previous frame and point
        prev_gray = gray
        prev_point = point

    cv2.destroyAllWindows()
