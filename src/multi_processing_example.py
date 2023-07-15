import time
from multiprocessing import Process
import concurrent.futures

start = time.perf_counter()

def do_something(seconds):

    print(f'Sleeping for {seconds} second')
    time.sleep(seconds)
    return f'Done sleeping {seconds}'

if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [5, 4, 3, 2, 1]
        results = executor.map(do_something, secs)

        for result in results:
            print(result)

    # processes = []
    # for _ in range(10):
    #     p = Process(target=do_something, args=[2])
    #     p.start()
    #     processes.append(p)

    # for process in processes:
    #     process.join()

    finish = time.perf_counter()
    print(f'Finished running in {round(finish-start, 2)} sec')