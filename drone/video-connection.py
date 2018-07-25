# # import socket
# # import sys
# # import numpy as np
# import ffmpeg
#
# # try:
# #     video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #     print("Socket successfully created")
# # except socket.error as err:
# #     print("socket creation failed with error %s" %(err))
#
# IP = "127.0.0.1"
# PORT = "8889"
# #
# # # Connect to Video Feed
# # video_socket.bind((IP, PORT))
#
# stream = ffmpeg.input('tcp://' + IP + ":" + PORT + "?listen")
# stream = ffmpeg.output(stream, "pipe:1", format='image2pipe')

import socketserver
import io
import cv2

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        print("howdy")
        self.data = self.request.recv(24000).strip()
        image = cv2.imread(io.BytesIO(self.data))
        print(image.shape)
        cv2.imshow(image)

if __name__ == "__main__":
    HOST, PORT = "localhost", 9090

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
