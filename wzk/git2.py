import os
from subprocess import call

# GIT_USER = os.environ['GIT_USER']
repository_list = ["wzk", "rokin", "rokin-meshes", "rocal", "jupla", "mopla"]


def git_pull_all():
    path = os.path.normpath(f"{__file__}/../../..")
    print("git pull...")
    for rep in repository_list:
        print(rep)
        if os.path.exists(f"{path}/{rep}"):
            call(f"cd {path}/{rep}; git add .; git stash; git pull", shell=True)


if __name__ == "__main__":
    git_pull_all()
