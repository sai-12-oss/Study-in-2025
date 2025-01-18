import time
start_time = time.time() 
def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)
print(fib(40))
print("--- %s seconds ---" % (time.time() - start_time))