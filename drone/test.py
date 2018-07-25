import socket
import sys

HOST, PORT = "localhost", 9090
data = " ".join(sys.argv[1:]) + "\n"
byte_data = data.encode('utf-8')

# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    while 1:
        sock.sendall(byte_data)

        # Receive data from the server and shut down
        received = sock.recv(1024)
finally:
    sock.close()

print("Sent:     {}".format(data))
print("Received: {}".format(received))
