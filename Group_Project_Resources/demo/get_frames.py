import cv2

def get_frames(video_file, save_dir):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    # Get every frame
    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, true_frame = cap.read()
        cv2.imwrite(f"{save_dir}/true_wheel_{i}.png", true_frame)

    # # Get first frame
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # ret, true_frame_first = cap.read()
    # cv2.imwrite(f"{save_dir}/one.png", true_frame_first)

    # # Get middle frame
    # cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    # ret, true_frame_middle = cap.read()
    # cv2.imwrite(f"{save_dir}/true_frame_middle.png", true_frame_middle)

    # # Get last frame
    # cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    # ret, true_frame_last = cap.read()
    # cv2.imwrite(f"{save_dir}/two.png", true_frame_last)
    cap.release()



mp4_file = '../Results/SingleGearSmallMovement/TrueVideoSingleGear.mp4'
download_path = '../Results/SingleGearSmallMovement'
get_frames(mp4_file, download_path)