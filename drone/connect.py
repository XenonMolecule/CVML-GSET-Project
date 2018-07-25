import socket
import cv2
import numpy as np
import os
from struct import unpack

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('localhost', 9090))
s.listen(1)

client_socket, addr = s.accept()
print("Got Connection")
while True:
    print("Running")
    buf = b''
    while len(buf)<4:
        buf += client_socket.recv(4-len(buf))
    size = unpack('!i', buf)
    print("receiving %s bytes" % size)
    img = b''
    if(size[0] > 0):
        img = client_socket.recv(size[0]+4)
    if(not size[0] > 30000 and not size[0] < 0):
        image = open('test.jpg', 'wb')
        image.write(img)
        print("Check test.jpg")
        image_np = cv2.imread('test.jpg')
        cv2.imshow("Drone Feed :)", image_np)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

cv2.waitKey()
