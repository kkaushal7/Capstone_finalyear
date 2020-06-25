from threading import Thread
import sys
import cv2

from queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=512):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
    def start(self):
		# start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
		# keep looping infinitely
        while True:
            if self.stopped:
                return
            if not self.Q.full():
				# read the next frame from the file
                (grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
				# add the frame to the queue
                self.Q.put(frame)
    
    def read(self):
		# return next frame in the queue
        return self.Q.get()
    
    def more(self):
		# return True if there are still frames in the queue
        return self.Q.qsize() > 0
    
    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True