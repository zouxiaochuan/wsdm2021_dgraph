import os
import sys
import re
import sys


def get_ckpt_num(name):
    search = re.search('epoch=(.*)-', name)

    return int(search.group(1))


if __name__ == '__main__':
    config_file = sys.argv[1]
    ckpt_path = sys.argv[2]

    ckpts = os.listdir(ckpt_path)

    ckpts = sorted(ckpts, key=get_ckpt_num)

    for ckpt in ckpts:
        fullpath = os.path.join(ckpt_path, ckpt)

        cmd = '{0} validate.py {1} {2}'.format(
            sys.executable, config_file, ckpt)

        print(cmd)
        os.system(cmd)
        pass
    pass
