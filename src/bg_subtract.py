import cv2


def background_subtraction(frames):
    """Perform background subtraction on a video."""
    if len(frames) == 0:
        return

    # Create background subtractor
    back_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    for frame_count, frame in enumerate(frames):
        # Blur frame
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Perform background subtraction
        fg_mask = back_subtractor.apply(blur)

        # Display frame
        if frame_count > 4:
            # Show original frame, foreground mask, and background subtracted frame side by side
            out = cv2.hconcat(
                [
                    frame,
                    cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR),
                    cv2.bitwise_and(frame, frame, mask=fg_mask),
                ]
            )
            cv2.imshow("Background Subtraction", out)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
