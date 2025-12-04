import platform
import numpy as np

from wzk.subprocess2 import call2, ssh_call2

cmd_linux_n_cpu = "grep -c ^processor /proc/cpuinfo"
cmd_darwin_n_cpu = "sysctl -a | grep machdep.cpu.core_count | awk '{print $2}'"

cmd_linux_cpu_usage = "grep 'cpu ' /proc/stat | awk '{print ($2+$4)*100/($2+$4+$5)}'"

cmd_linux_load = "uptime | awk '{print $10,$11,$12}''"
cmd_darwin_load = "uptime | awk '{print $10,$11,$12}'"


def get_n_cpu(host=None):
    if host is None:
        if platform.system() == "Darwin":
            n_cpu = call2(cmd_darwin_n_cpu)

        elif platform.system() == "Linux":
            n_cpu = call2(cmd_linux_n_cpu)

        else:
            raise ValueError
    else:
        n_cpu = ssh_call2(host=host, cmd=cmd_linux_n_cpu)

    return int(n_cpu)


def get_cpu_usage(host):
    # https://stackoverflow.com/a/9229580/7570817

    if host is None:
        if platform.system() == "Darwin":
            raise NotImplementedError

        elif platform.system() == "Linux":
            cpu_usage = call2(cmd_linux_cpu_usage)

        else:
            raise ValueError
    else:
        cpu_usage = ssh_call2(host=host, cmd=cmd_linux_cpu_usage)

    return float(cpu_usage)


def get_load(host):
    # https://stackoverflow.com/a/24841357/7570817

    if host is None:
        if platform.system() == "Darwin":
            load = call2(cmd_darwin_load)

        elif platform.system() == "Linux":
            load = call2(cmd_linux_load)

        else:
            raise ValueError

    else:
        load = ssh_call2(host=host, cmd=cmd_linux_load)

    return np.array([float(ll) for ll in load.split(",")]).sum() / 3
