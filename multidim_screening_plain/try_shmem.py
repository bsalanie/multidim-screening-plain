# SuperFastPython.com
# example of using shared memory with floats
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory


# task executed in a child process
def task(shared_mem, i):
    # read some float data from the shared memory
    sh_i = int(shared_mem.buf[i])
    # close as no longer needed
    shared_mem.close()
    return sh_i * sh_i


# protect the entry point
if __name__ == "__main__":
    # create a shared memory
    shared_mem = SharedMemory(create=True, size=100)
    # fill the shared memory
    for i in range(5):
        shared_mem.buf[i] = bytearray(i)
    # create tasks
    inputs = [[shared_mem, i] for i in range(5)]
    with Pool() as pool:
        res = pool.starmap(task, inputs)
    # close the shared memory
    shared_mem.close()
    # release the shared memory
    shared_mem.unlink()
    print(res)
