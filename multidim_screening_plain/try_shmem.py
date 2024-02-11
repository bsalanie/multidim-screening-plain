# SuperFastPython.com
# example of using shared memory with floats
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory


# task executed in a child process
def task(shared_mem, i):
    # read some float data from the shared memory
    float(shared_mem.buf[i])
    # close as no longer needed
    shared_mem.close()


# protect the entry point
if __name__ == "__main__":
    # create a shared memory
    shared_mem = SharedMemory(create=True, size=100)
    # report the shared memory
    data = [int(shared_mem.buf[i]) for i in range(5)]
    print(f"Then: {data=}")
    # create a child process
    process = Process(target=task, args=(shared_mem,))
    # start the child process
    process.start()
    # wait for the child process to finish
    process.join()
    # report the shared memory
    data = [int(shared_mem.buf[i]) for i in range(5)]
    print(f"Now: {data=}")
    # close the shared memory
    shared_mem.close()
    # release the shared memory
    shared_mem.unlink()
