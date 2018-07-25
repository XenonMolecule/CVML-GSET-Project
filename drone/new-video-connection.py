import socket
import io
import cv2
from PIL import Image
import numpy as np

TCP_IP = 'localhost'
TCP_PORT = 9090
BUFFER_SIZE = 23552  # Normally 1024, but we want *slow* response (no we don't rip)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print('Connection address:', addr)
while 1:
    data = conn.recv(BUFFER_SIZE).strip()
    if not data: break
    #image = Image.open(io.BytesIO(data))
    #image.show()
    #image_np = np.array(image)
    # cv2.imshow(image_np)
    print("received data:", data)
    conn.send(bytes(0xFF))
conn.close()
