#!/usr/bin/env python3

import os
from subprocess import CalledProcessError, run
from typing import Dict, List

import click


def github_repo_name():
    return os.environ["GITHUB_REPOSITORY"].split("/")[1]


def git_list_changes() -> List[str]:
    return run(
        ["git", "log", "-1", "--name-only", "--pretty="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def git_branch_name() -> str:
    return os.environ["GITHUB_REF"][len("refs/heads/") :]


def target_branch() -> str:
    if git_branch_name() == "staging":
        return "release"
    else:
        return "staging"


def git_commit_title() -> str:
    return run(
        ["git", "log", "-1", r"--pretty=format:%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def git_commit_author() -> str:
    return run(
        ["git", "log", "-1", r"--pretty=format:%an"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def git_commit_author_email() -> str:
    return run(
        ["git", "log", "-1", r"--pretty=format:%ae"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def git_short_sha() -> str:
    return os.environ["GITHUB_SHA"][:7]


def is_dev_branch() -> bool:
    return git_branch_name() not in ["release", "staging"]


def ci_yaml_changed() -> bool:
    return ".github/workflows/ci.yml" in git_list_changes()


def docker_tag() -> str:
    return f"{git_branch_name()}-{git_short_sha()}"


def docker_stack_name() -> str:
    return f"{github_repo_name()}-{git_branch_name()}-{git_short_sha()}"


def should_upload_package() -> bool:
    return (ci_yaml_changed() and is_dev_branch()) or git_branch_name() == "release"


def flutter_path() -> str:
    return "/opt/flutter"


def overwrite_path() -> str:
    return ":".join(
        [
            os.environ["PATH"],
            "/usr/local/bin/",
            "/home/runner/.local/bin/",
            f"{flutter_path()}/bin",
        ]
    )


def pr_body() -> str:
    if target_branch() == "staging":
        return 'To merge into the staging branch, please use "Rebase and merge", or "Squash and merge".'
    elif target_branch == "release":
        return 'To merge into the release branch, please use "Create a merge commit".'
    return ""


def get_env() -> Dict[str, str]:
    return {
        "PROJECT_NAME": github_repo_name(),
        "DOCKER_TAG": docker_tag(),
        "CI_YAML_CHANGED": str(ci_yaml_changed()),
        "IS_DEV_BRANCH": str(is_dev_branch()),
        "BRANCH_NAME": git_branch_name(),
        "TARGET_BRANCH": target_branch(),
        "COMMIT_TITLE": git_commit_title(),
        "COMMIT_AUTHOR": git_commit_author(),
        "COMMIT_AUTHOR_EMAIL": git_commit_author_email(),
        "SHOULD_UPLOAD_PACKAGE": str(should_upload_package()),
        "FLUTTER_PATH": flutter_path(),
        "PATH": overwrite_path(),
        "PR_BODY": pr_body(),
    }


@click.command()
@click.option("-w", "--write", is_flag=True)
def main(write):
    content = ""
    for key, val in get_env().items():
        content += f"{key}={val}\n"
    if write:
        with open(os.environ["GITHUB_ENV"], "a") as env_file:
            env_file.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    try:
        main()
    except CalledProcessError as err:
        exit(err.stdout + err.stderr)
