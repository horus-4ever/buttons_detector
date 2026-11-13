import git

REPO = git.Repo(".")

def branch_infos() -> dict[str, str]:
    """
    Returns a tuple containing the current branch name and hash.
    """
    global REPO
    return {
        "name": REPO.active_branch.name,
        "hash": REPO.head.commit.hexsha
    }