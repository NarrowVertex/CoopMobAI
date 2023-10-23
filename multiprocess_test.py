import multiprocessing
import time

COMPUTE_NUMBER = 100000000
process_number = 4  # multiprocessing.cpu_count()


def serial_main():
    compute_results = []
    for i in range(process_number):
        result = count_func(COMPUTE_NUMBER)
        compute_results.append(result)
    # print(f"Compute results : {compute_results}")

    end_results = []
    for i in range(process_number):
        end_results.append(True)
    # print(f"End results : {end_results}")

def multiprocessing_main():
    multiprocessing.set_start_method("spawn")

    locals = []
    processes = []
    for i in range(process_number):
        local, remote = multiprocessing.Pipe()
        locals.append(local)
        p = multiprocessing.Process(target=worker, args=(remote, count_func))
        p.daemon = True
        p.start()
        processes.append(p)
        remote.close()

    compute_results = []
    for local in locals:
        local.send(("work", COMPUTE_NUMBER))
    # compute_results = [local.recv() for local in locals]
    # print(f"Compute results : {compute_results}")

    end_results = []
    for local in locals:
        local.send(("end", None))
    # end_results = [local.recv() for local in locals]
    # print(f"End results : {end_results}")

    for p in processes:
        p.join()
        p.close()


def worker(conn, func):
    while True:
        cmd, data = conn.recv()
        if cmd == "work":
            result = func(data)
            conn.send(result)
        elif cmd == "end":
            conn.send(True)
            return
        else:
            raise NotImplementedError

def count_func(number):
    count = 0
    for i in range(number):
        count += 1
    return count


def check_process_time(func, message):
    start_time = time.time()

    func()

    end_time = time.time()
    elapsed_time = end_time - start_time
    message = message.replace("%t", f"{elapsed_time}")
    print(message)


if __name__ == "__main__":
    check_process_time(serial_main,
                       "Serial computing time : \t\t\t%ts")
    check_process_time(multiprocessing_main,
                       "Multiprocessing computing time : \t%ts")
