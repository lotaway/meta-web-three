from cmd import Cmd
import os


class Mm:
    def __init__(self, redio=1.0):
        self.redio = redio

    def add(self, a, b):
        return self.redio + a + b


def main():
    a = 1
    b = "2"
    c = True
    d = a + int(b) + c
    print("Running Python", a, d)
    mm = Mm(1.5)
    print(mm.add(a, int(b)))
    with open("./run-dev.js") as file_run_dev:
        file_run_dev_content = file_run_dev.read()
        print("get file")
        file_run_dev.close()
    cmd = Cmd(file_run_dev_content)
    os.system("dir")


if __name__ == '__main__':
    main()
