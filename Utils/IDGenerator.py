import datetime

def generate_id():
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y%m%d%H%M%S%f")  # Format: YYYYMMDDHHMMSSmicroseconds
    return timestamp_str
