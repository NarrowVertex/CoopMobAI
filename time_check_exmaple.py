import time
"""
start_time = time.time()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Spent time : {elapsed_time}s")
"""

"""
%t to time
"""
def check_process_time(func, message):
    start_time = time.time()

    func()

    end_time = time.time()
    elapsed_time = end_time - start_time
    message = message.replace("%t", f"{elapsed_time}")
    print(message)
