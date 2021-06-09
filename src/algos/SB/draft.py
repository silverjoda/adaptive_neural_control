import multiprocessing
import time


def sq_num(num):
    #proc_name = multiprocessing.current_process().name
    #print(f'Proc name: {proc_name}, num: {self.num}')
    time.sleep(1)
    return num# ** 2

def worker(input_queue, output_queue):
    while True:
        obj = input_queue.get()
        numsq = sq_num(obj)
        output_queue.put(numsq)

if __name__ == '__main__':
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    #print(dir(queue)); exit()
    p = multiprocessing.Process(target=worker, args=(input_queue, output_queue))
    p.start()

    input_queue.put(0)
    for i in range(1, 100):
        if not output_queue.empty():
            print(f"Output: {output_queue.get()}")
            input_queue.put(i)
        time.sleep(0.1)

    # Wait for the worker to finish
    input_queue.close()
    input_queue.join_thread()
    p.join()
