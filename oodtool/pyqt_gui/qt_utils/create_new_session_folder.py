import os


def create_new_session_folder(dataset_root_path):
    existing_sessions = []
    for file in os.listdir(dataset_root_path):
        d = os.path.join(dataset_root_path, file)
        if os.path.isdir(d) and file.startswith("oodsession_"):
            _, session_id = file.split("_", 1)
            try:
                existing_sessions.append(int(session_id))
            except ValueError:
                continue
    next_session = 0
    if len(existing_sessions) > 0:
        existing_sessions.sort()
        next_session = existing_sessions[-1] + 1
    metadata_folder = os.path.join(dataset_root_path,
                                   "oodsession_" + str(next_session))
    os.makedirs(metadata_folder)
    return metadata_folder
