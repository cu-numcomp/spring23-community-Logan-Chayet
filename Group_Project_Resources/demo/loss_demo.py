import extract_frames
import create_frame_triplets
import shutil
import os
import subprocess
import sys
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt


mp4 = '/Users/nicolasperrault/Desktop/spring23-community-Logan-Chayet/Group_Project_Resources/Results/Hand/hand.mp4'
output_dir = '/Users/nicolasperrault/Desktop/frame-interpolation/demo/true_frames'
true_middle_frame_output_dir = '/Users/nicolasperrault/Desktop/frame-interpolation/demo/true_middle_frames'
photos_dir = '/Users/nicolasperrault/Desktop/frame-interpolation/photos'
interpolated_middle_frame_dir = '/Users/nicolasperrault/Desktop/frame-interpolation/demo/interpolated_middle_frames'
vgg_weights = '/Users/nicolasperrault/Desktop/frame-interpolation/pretrained_models/vgg/imagenet-vgg-verydeep-19.mat'

# For videos:
video_interpolated_images_dir = '/Users/nicolasperrault/Desktop/frame-interpolation/photos/interpolated_frames'
frame_videos = '/Users/nicolasperrault/Desktop/frame-interpolation/demo/frame_videos'

extract_frames.extract_frames(mp4, output_dir)

sys.path.append('losses')
import losses

triplets = create_frame_triplets.create_frame_triplets(output_dir)
vgg_losses = []
end_frame_nums = []

for frame_first, frame_middle, frame_end in triplets:

    shutil.move(os.path.join(output_dir, frame_middle), true_middle_frame_output_dir)

    # move frame_first, frame_end to photos dir.
    shutil.copy(os.path.join(output_dir, frame_first), os.path.join(photos_dir, 'one.png'))
    shutil.copy(os.path.join(output_dir, frame_end), os.path.join(photos_dir, 'two.png'))
    
    # run interp
    subprocess.run([
    'python3', '-m', 'eval.interpolator_test',
    '--frame1', 'photos/one.png',
    '--frame2', 'photos/two.png',
    '--model_path', 'pretrained_models/film_net/Style/saved_model',
    '--output_frame', 'photos/output_middle.png'
    ])

    # delete the old one.png and two.png files
    os.remove(os.path.join(photos_dir, 'one.png'))
    os.remove(os.path.join(photos_dir, 'two.png'))

    # move output_middle to interpolated_middle_frames
    frame_num = re.search(r'\d+', frame_middle).group(0)
    middle_frame_num = int(frame_num)
    middle_frame_name = f'interpolated_middle_frame_{middle_frame_num}.png'
    shutil.move(os.path.join(photos_dir, 'output_middle.png'), os.path.join(interpolated_middle_frame_dir, middle_frame_name))

    # Calculate loss:
    true_image = Image.open(true_middle_frame_output_dir + '/' + frame_middle)
    interoplated_image = Image.open(interpolated_middle_frame_dir + '/' + middle_frame_name)
    tensor_image = img_to_array(true_image)
    tensor_image = tf.convert_to_tensor(tensor_image)
    example = {'y': tensor_image}
    interoplated_image = img_to_array(interoplated_image)
    interpolated_tensor_image = tf.convert_to_tensor(interoplated_image)
    prediction = {'image': interpolated_tensor_image}
    loss = losses.vgg_loss(example, prediction, vgg_weights)
    vgg_losses.append(loss.numpy())
    end_frame_nums.append(re.search(r'\d+', frame_end).group(0))

plt.scatter(end_frame_nums, vgg_losses)
plt.title('vgg loss vs frame distance')
plt.xlabel('Frame Distance')
plt.ylabel('VGG Loss')
plt.savefig('vgg_loss_scatter.png')
plt.show()

# create videos from different frame lengths:
for frame_first, frame_middle, frame_end in triplets:

    shutil.copy(os.path.join(output_dir, frame_first), os.path.join(photos_dir, 'one.png'))
    shutil.copy(os.path.join(output_dir, frame_end), os.path.join(photos_dir, 'two.png'))
    

    # run interp video
    subprocess.run([
    "python3", "-m", "eval.interpolator_cli", 
    "--pattern", "photos", 
    "--model_path", "pretrained_models/film_net/Style/saved_model", 
    "--times_to_interpolate", "6", 
    "--output_video"
    ])

    # delete the old one.png and two.png files
    os.remove(os.path.join(photos_dir, 'one.png'))
    os.remove(os.path.join(photos_dir, 'two.png'))

    # move interpolated video.
    video_num = re.search(r'\d+', frame_end).group(0)
    interp_video_num = int(video_num)
    interp_video = f'interpolated_middle_frame_{interp_video_num}.mp4'
    shutil.move(os.path.join(photos_dir, 'interpolated.mp4'), os.path.join(frame_videos, interp_video))

    # delete all interpolated frames
    for filename in os.listdir(video_interpolated_images_dir):
        file_path = os.path.join(video_interpolated_images_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")



