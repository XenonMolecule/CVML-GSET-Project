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
    img_str = ''
    while True:
        data = client_socket.recv(1024)
        if len(img_str) >= size[0]:
            break
        img_str += str(data)
    image = np.fromstring(img_str, np.uint8)
    image_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imshow(image_np)
