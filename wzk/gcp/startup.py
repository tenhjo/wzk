def make_startup_file(user, bash_file):
    s = (
        f"#!/bin/bash\n"
        f"source ~/.bashrc\n"
        f"sudo -H -u {user} sudo chmod 777 -R /home/{user}/src\n"
        f"sudo -H -u {user} tmux new-session -d -s main\n"
        f"sudo -H -u {user} tmux send 'python /home/{user}/src/wzk/wzk/git2.py' C-m\n"
        f"sudo -H -u {user} tmux send 'source {bash_file}' C-m\n"
        f"sudo -H -u {user} tmux send 'cd /home/{user}' C-m\n"
        f"sudo -H -u {user} tmux -2  attach-session -t main\n"
    )
    return s


# f"python /home/{user}/src/wzk/wzk/git2.py\n" \         # f"sudo chmod 777 -R /home/{user}/src/\n" \
