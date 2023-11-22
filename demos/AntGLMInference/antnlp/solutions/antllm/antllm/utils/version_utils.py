import os


# 将原 antllm.utils.version_utils.is_old_version 移到此处
def is_oldest_version(path):
    new_vocab_files = ['merge.model']
    new_vocab_file_exists = []
    for filename in new_vocab_files:
        if not os.path.exists(os.path.join(path, filename)):
            new_vocab_file_exists.append(False)
        else:
            new_vocab_file_exists.append(True)
    if all(new_vocab_file_exists):
        return False
    if any(new_vocab_file_exists):
        return 'new_version_file_absent'
    else:
        return True
