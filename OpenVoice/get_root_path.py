import os


def get_root_path():
    """获得项目根目录"""
    path = os.path.abspath(__file__)
    return os.path.split(path)[0]


if __name__ == "__main__":
    print(get_root_path())
