# Python script to read video frames and timestamps using ffmpeg
import subprocess as sp
import threading

import matplotlib.pyplot as plt
import numpy
import cv2

# connect to http://172.16.10.1:8888/ before running

ffmpeg_command = [ 'ffmpeg',
                   # '-nostats', # do not print extra statistics
                    #'-debug_ts', # -debug_ts could provide timestamps avoiding showinfo filter (-vcodec copy). Need to check by providing expected fps TODO
                    '-r', '30', # output 30 frames per second
                    '-i', 'tcp://127.0.0.1:8889?listen',
                    # '-vcodec', 'rawvideo',
                    #'-vcodec', 'copy', # very fast!, direct copy - Note: No Filters, No Decode/Encode, no quality loss
                    #'-vframes', '20', # process n video frames only. For Debugging
                    '-f', 'image2pipe', 'pipe:1' ] # outputs to stdout pipe. can also use '-' which is redirected to pipe


# seperate method to read images on stdout asynchronously
def AppendProcStdout(proc, nbytes, AppendList):
    while proc.poll() is None: # continue while the process is alive
        AppendList.append(proc.stdout.read(nbytes)) # read image bytes at a time


if __name__ == '__main__':
    # run ffmpeg command
    pipe = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE)

    # 2 threads to talk with ffmpeg stdout and stderr pipes
    framesList = [];
    appendFramesThread = threading.Thread(group=None, target=AppendProcStdout, name='FramesThread', args=(pipe, 640*480*3, framesList), kwargs=None) # assuming rgb video frame with size 1280*720

    # start threads to capture ffmpeg frames and info.
    appendFramesThread.start()

    # wait for few seconds and close - simulating cancel
    import time; time.sleep(15)
    pipe.terminate()

    # check if threads finished and close
    appendFramesThread.join()

    # save an image per 30 frames to disk
    for cnt,raw_image in enumerate(framesList):
        if (raw_image != b''):
            image1 =  numpy.fromstring(raw_image, dtype='uint8')
            image2 = image1.reshape((480,640,3))  # assuming rgb image with size 640 X 480
            # show video frame just to verify
            cv2.imshow(videoFrameName)
        print("Waiting")

    print("DONE")
