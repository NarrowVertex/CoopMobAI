# start("a")
# end("a")
# start("b")
# end("b")
# summary()
from datetime import datetime, timedelta


class TimeChecker:

    def __init__(self):
        self.time_map = {}

    def start(self, time_id):
        if time_id not in self.time_map:
            self.time_map[time_id] = [get_time()]
        else:
            self.time_map[time_id].append(get_time())

    def end(self, time_id):
        if time_id not in self.time_map:
            print(f"No key has registered id[{time_id}]")
            return

        self.time_map[time_id][-1] = get_time() - self.time_map[time_id][-1]

    def summary(self):
        for time_data in self.time_map.items():
            time_id = time_data[0]
            time_data = time_data[1]

            sum_time_data = 0
            for data in time_data:
                sum_time_data += int(data.total_seconds() * 1000.0)
            average_time = sum_time_data / len(time_data)
            average_time_second, average_time_millisecond = divmod(average_time, 1000)
            average_time = timedelta(seconds=average_time_second, milliseconds=average_time_millisecond)
            total_time = sum_time_data
            total_time_second, total_time_millisecond = divmod(total_time, 1000)
            total_time = timedelta(seconds=total_time_second, milliseconds=total_time_millisecond)

            print(f"Average time for [{time_id}] is {average_time} (s) with {len(time_data)} (count) and total is {total_time} (s)")

    def reset(self):
        self.time_map.clear()

def get_time():
    return datetime.now()

