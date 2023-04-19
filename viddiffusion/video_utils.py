import cv2
import os


def get_video_fps(input_video_path):
	"""
	Get fps of a video
	:param input_video_path: path to input video
	:return: fps of input video
	"""
	video_capture = cv2.VideoCapture(input_video_path)
	video_fps = video_capture.get(cv2.CAP_PROP_FPS)
	video_capture.release()
	return video_fps


def get_video_duration(input_video_path):
	"""
	Get duration of a video
	:param input_video_path: path to input video
	:return: duration of input video
	"""
	video_capture = cv2.VideoCapture(input_video_path)
	video_fps = video_capture.get(cv2.CAP_PROP_FPS)
	video_frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
	video_duration = video_frame_count / video_fps
	video_capture.release()
	return video_duration


def get_video_fps_and_duration(input_video_path):
	"""
	Get fps and duration of a video
	:param input_video_path: path to input video
	:return: fps and duration of input video
	"""
	video_capture = cv2.VideoCapture(input_video_path)
	video_fps = video_capture.get(cv2.CAP_PROP_FPS)
	video_frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
	video_duration = video_frame_count / video_fps
	video_capture.release()
	return video_fps, video_duration


def split_video_to_images(input_video_path, output_dir_path, fps, start_time, end_time):
	"""
	Split a video into images
	:param input_video_path: path to input video
	:param video_fps: fps of input video
	:param start_time: start time of video
	:param end_time: end time of video
	:param output_dir_path: output directory path
	:return: None
	"""
	video_capture = cv2.VideoCapture(input_video_path)
	video_fps = video_capture.get(cv2.CAP_PROP_FPS)
	video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_time * video_fps)
	frame_count = 0
	while video_capture.isOpened():
		success, frame = video_capture.read()
		if not success:
			break
		if frame_count >= (end_time - start_time) * video_fps:
			break

		# save only fps number of images per second
		if int(frame_count % (video_fps / fps)) == 0:
			cv2.imwrite(os.path.join(output_dir_path, f'{frame_count:05d}.png'.format()), frame)
		frame_count += 1
			
	video_capture.release()


def combine_images_to_video(input_dir_path, output_video_path, fps):
	"""
	Combine images to a video
	:param input_dir_path: path to input directory
	:param output_video_path: path to output video
	:param fps: fps of output video
	:return: None
	"""
	image_names = sorted(os.listdir(input_dir_path))
	image_path = os.path.join(input_dir_path, image_names[0])
	image = cv2.imread(image_path)
	height, width, _ = image.shape
	video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
	for image_name in image_names:
		image_path = os.path.join(input_dir_path, image_name)
		image = cv2.imread(image_path)
		video_writer.write(image)
	video_writer.release()

