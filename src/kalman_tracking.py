import cv2
import numpy as np
import tpd
from video import Video
from io_video import IOVideo
from cpm import CPM
from logger import Logger
import sys
import csv
import os
import util

"""
This script is only for testing the perfomance of Kalman filter
"""

# Instantiate OCV kalman filter
class KalmanFilter():

	kf = cv2.KalmanFilter(4, 2)
	kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
	kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

	def Estimate(self, coordX, coordY):
		''' This function estimates the position of the object'''
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		self.kf.correct(measured)
		predicted = self.kf.predict()
		return predicted

class Tracking():

	def pe_and_kalman(self):

		JOINTS_NUM = 14
		level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'
		
		video_path = "../data/video/T000/Rust/kinect.avi"
		result_save_path = "../data/video/T000/Rust/"

		pe_logger = Logger('pe_logger',level_name)

		video_pe = Video(video_path)
		# video_pe.set_next_frame_index(716)  # for making joint video
		io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
							fps=30,height=368,
							width=368*video_pe.WIDTH/video_pe.HEIGHT) #368
		cpm = CPM()
		box_size = None

		# Create Kalman Filter Object
		kfObj = KalmanFilter()
		predictedCoords = np.zeros((2, 1), np.float32)

		# init video writer
		tracking_video_path = result_save_path + 'tracking_result.avi'
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out_video = cv2.VideoWriter(tracking_video_path,fourcc,1,\
								(int(368),int(368*video_pe.WIDTH/video_pe.HEIGHT)),isColor= True)

		while(video_pe.next_frame_index<5):
		# while(video_pe.next_frame_index<video_pe.FRAME_COUNT):
				
			pe_logger.info( 'Frame {}/{} is being processed'.format(
												video_pe.next_frame_index,
												int(video_pe.FRAME_COUNT)) )
			frame_pe = io_video.get_video_frames(video_pe,1,
														grayscale_on=False)
			[prediction,conf_maps] = cpm.locate_joints(frame_pe,
													return_conf_map=True)
			if prediction.shape[2]==0:
				print("Cannot find people in the frame. Skip the frame!")
				continue

			# np.savetxt(prediction_arr_path+'pred_{}.txt'.format(\
			#                         video_pe.next_frame_index),prediction)

			# if box_size is None:
			#     box_size = math.sqrt( math.pow(\
			#                             prediction[0,0]-prediction[1,0],2)+\
			#                          math.pow(\
			#                             prediction[0,1]-prediction[1,1],2) )
			#     box_size = int( box_size/2)
			
			# box_size = 22

			for joint_i in [4]: #changed in 28th Feb
			# for joint_i in range(JOINTS_NUM):
				# TODO: may have bug - box is out of image
				pred_y = int(prediction[joint_i,0])
				pred_x = int(prediction[joint_i,1])

				predictedCoords = kfObj.Estimate(pred_x, pred_y) # Prediction


				# joint_frame = frame_pe[pred_y-box_size:pred_y+box_size,
				#                         pred_x-box_size:pred_x+box_size]
				# cv2.imwrite(segment_img_path+'seg_{}_{}.png'.format(joint_i,
				#                                 video_pe.next_frame_index),
				#                                                 joint_frame)
				# Draw kalman filter predicted coords
				cv2.rectangle(frame_pe,(predictedCoords[0]-22,predictedCoords[1]-22),\
										(predictedCoords[0]+22,predictedCoords[1]+22),(0,255,0),1)
				cv2.putText(frame_pe, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0])

				# Draw CPM coords 
				cv2.rectangle(frame_pe,(pred_x-22,pred_y-22),\
										(pred_x+22,pred_y+22),(0,0,255),1)
				cv2.putText(frame_pe, "CPM", (pred_y + 50, pred_x + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,255])

				cv2.imwrite(result_save_path+'seg_{}_{}.png'.format(joint_i,
												video_pe.next_frame_index),
																frame_pe)
				out_video.write(frame_pe)
				# joint_conf_map = conf_maps[pred_y-box_size:pred_y+box_size,
				#                             pred_x-box_size:pred_x+box_size,
				#                             joint_i]
				# joint_conf_map = joint_conf_map / np.sum(joint_conf_map) # Renormalize
				# np.savetxt(conf_arr_path+'conf_{}_{}.txt'.format(joint_i,
				#                 video_pe.next_frame_index),joint_conf_map)
			tfd_logger = Logger('tfd_logger',level_name)

		del	out_video
		del cpm


	def save_resized_frames(self):
		video_path = "../data/video/T000/Rust/kinect.avi"
		result_save_path = "../data/video/T000/Rust/"
		frame_path = result_save_path + 'resized_frames/'

		if not os.path.isdir(frame_path):
			os.mkdir(frame_path)

		video_pe = Video(video_path)
		
		io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
							fps=30,height=368,
							width=368*video_pe.WIDTH/video_pe.HEIGHT) #368
		print video_pe.FRAME_COUNT
		while(video_pe.next_frame_index<int(video_pe.FRAME_COUNT)):
				
			frame_pe = io_video.get_video_frames(video_pe,1,
														grayscale_on=False)
			cv2.putText(frame_pe, "{}".format(video_pe.next_frame_index), (500,40), cv2.FONT_HERSHEY_SIMPLEX,1.0, [0,0,255])
			path = frame_path + "img{}.jpg".format(video_pe.next_frame_index)
			cv2.imwrite(path,frame_pe)



	def postions_and_kalman(self,cpm_joint_path,annotion_path):
		"""
		This function assume all jonit postions are saved and they are used as measurement data

		Args:
			cpm_joint_path: the folder contains the joint positions estimated by CPM
			annotion_path: the folder contains joint positions of manully annotion

		Return:
			error_sum: the sum of suqare difference between tracking position and annotion position
		"""

		# Create Kalman Filter Object
		kfObj = KalmanFilter()
		# predictedCoords = np.zeros((2, 1), np.float32)
		start = [[179],[274]]
		predictedCoords = np.array(start,np.float32)

		# read in joint posotions
		cpm_joint_pos = util.get_jonit_pos_sequence(cpm_joint_path,4,type="cpm")
		# print cpm_joint_pos
		annotion_pos = util.get_jonit_pos_sequence(annotion_path,0,type="annotion") # xmin ymin xmax ymax class
		# print annotion_pos

		predicted = []
		# for i in range(0,5):
		for i in range(0,len(annotion_pos)):
			pred_y = cpm_joint_pos[i][0]
			pred_x = cpm_joint_pos[i][1]
			predictedCoords = kfObj.Estimate(pred_x, pred_y)
			predicted.append(predictedCoords)

		print len(predicted)




# main function
def main():
	tracking = Tracking()
	# tracking.pe_and_kalman()
	# tracking.save_resized_frames()
	cpm_joint_path = "../results/clipped/prediction_arr/"
	annotion_path = "../results/clipped/bbox_txt/"
	tracking.postions_and_kalman(cpm_joint_path,annotion_path)

if __name__ == "__main__":
	main()
print('Program Completed!')