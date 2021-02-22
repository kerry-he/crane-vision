##################################################################
## VIDEO RECORDER
## Helper class to record and save sequence of images as a video.
##################################################################

# Libraries
import cv2

class VideoRecorder():
    def __init__(self, name, framerate, shape):
        name = name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter(name, fourcc, framerate, shape)

    def write(self, img):
        self.out.write(img)

    def release(self):
        self.out.release()
