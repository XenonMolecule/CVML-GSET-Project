import socket

TCP_IP = "127.0.0.1"
TCP_PORT = 8889

BUFFER_SIZE = 640*480*3

MSG = "Hello World"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect()
s.send(MSG)
data = s.recv(BUFFER_SIZE)
s.close()

print("Received Data:" + str(data))

# ADD ?listen
