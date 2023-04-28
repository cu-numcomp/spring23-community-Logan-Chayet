import cv2
import os

def extract_frames(video_path, output_dir):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read the video file
    cap = cv2.VideoCapture(video_path)

    # initialize frame counter
    count = 0

    # loop through each frame in the video
    while True:
        # read the next frame
        ret, frame = cap.read()

        # if there are no more frames, break out of the loop
        if not ret:
            break

        # save the current frame as a PNG image
        output_path = os.path.join(output_dir, 'frame_' + str(count) + '.png')
        cv2.imwrite(output_path, frame)

        # increment the frame counter
        count += 1

    # release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
