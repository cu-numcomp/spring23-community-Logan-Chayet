import os

def create_frame_triplets(directory):
    # get a list of all files in the directory
    files = os.listdir(directory)

    # filter the list to include only PNG files with the format "frame_xxx.png"
    frames = [f for f in files if f.endswith('.png') and f.startswith('frame_')]

    # sort the frames by their number (assuming the format is "frame_xxx.png")
    frames.sort(key=lambda x: int(x[6:-4]))

    # create a list of frame triplets
    triplets = []
    for i in range(1, len(frames), 2):
        # calculate the index of the last frame in the triplet
        j = i * 2

        # if j is out of range, break out of the loop
        if j >= len(frames):
            break

        # add the triplet to the list
        triplets.append(('frame_0.png', frames[i], frames[j]))

    return triplets
