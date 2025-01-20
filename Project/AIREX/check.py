import time 
start_time = time.time()
def Fib(n):
    if n <= 1:
        return n 
    return Fib(n-1)+Fib(n-2)
import socket

# Get and print the hostname
hostname = socket.gethostname()
print(f"The current node is: {hostname}")
import os

node_name = os.getenv("PBS_NODENAME", "Unknown Node")
print(f"The current PBS node is: {node_name}")

print(Fib(40))
print("--- %s seconds ---" % (time.time() - start_time))