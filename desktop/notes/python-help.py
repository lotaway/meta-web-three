from cmd import Cmd
import os
import requests


class Mm:
    def __init__(self, radio=1.0):
        self.radio = radio

    def add(self, a, b) -> int:
        return self.radio + a + b


class Status:
    def __init__(self,default_value: bool):
        self.is_start = default_value


def main(args: [str, ...] = None):
    with open("./python.md", encoding="utf-8") as file_run_dev:
        file_run_dev_content = file_run_dev.read()
        print("get file")
        if file_run_dev_content:
            lines = file_run_dev_content.split("\n")
            count = 0
            status = Status(False)
            for line in lines:
                is_start = line.find("```python") > -1
                if status.is_start is True or is_start:
                    print("This is a python line", count, ':', line)
                    count += 1
                if status.is_start is True and line.find("```") > -1:
                    print("\n")
                    status.is_start = False
                if is_start:
                    status.is_start = True
        file_run_dev.close()
    cmd = Cmd(file_run_dev_content)
    requests.get("https://demo.8248.net")
    # os.system("dir")


if __name__ == '__main__':
    main()
