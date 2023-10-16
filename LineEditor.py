import os


def format_log(log: str) -> str:
    keywords = ["start episode", "reset env", "trace_map", "map", "target_pos", "episode",
                "last_agent_pos", "action", "curr_agent_pos", "reward"]
    replace_keywords = [["trace_map", "trace"]]
    for replace_keyword in replace_keywords:
        for i in range(len(keywords)):
            keyword = keywords[i]
            if keyword == replace_keyword[0]:
                keywords[i] = replace_keyword[1]
        log = log.replace(replace_keyword[0], replace_keyword[1])

    for keyword in keywords:
        log = log.replace(keyword, "\n" + keyword)

    for replace_keyword in replace_keywords:
        log = log.replace(replace_keyword[1], replace_keyword[0])

    return log.strip()


def read_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 파일인 경우에만 내용 읽기
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                formatted_content = format_log(content)
                print(f"--- {filename} ---")
                # print(content)
                print("\n")

        # 클론 파일 생성
        clone_file_path = os.path.join(folder_path, filename + ".csv")
        with open(clone_file_path, 'w', encoding='utf-8') as clone_f:
            clone_f.write(formatted_content)


# Input log
log = """start episode : 0 - 0reset envmap : [[some array1]]trace_map : [[some array2]]target_pos : [8, 17]episode : 0, time_step : 0last_agent_pos : [16, 16], last_agent_angle : 0action: [0, 0]curr_agent_pos : [16.0, 16.0], curr_agent_angle : -10reward: 0.0, done: Falseepisode : 0, time_step : 1last_agent_pos : [16.0, 16.0], last_agent_angle : -10action: [0, 1]curr_agent_pos : [16.0, 16.0], curr_agent_angle : 0reward: 0.0, done: Falseepisode : 0, time_step : 2last_agent_pos : [16.0, 16.0], last_agent_angle : 0action: [1, 1]curr_agent_pos : [16.09848077530122, 16.017364817766694], curr_agent_angle : 10reward: -0.09561959628601535, done: False"""

formatted_log = format_log(log)
print(formatted_log)

folder_path = "C:/Users/jebum/Desktop/Projects/Python/Reinforcement Learning/Simulation3/Saves/Pathfinding_MIMO/20231004132613667936/debugs"
read_files_in_folder(folder_path)
