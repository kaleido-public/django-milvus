#!/usr/bin/env python3

from subprocess import CalledProcessError, run

import click


def pr_exists(*, base: str, head: str) -> bool:
    if run(["gh", "pr", "view", "--head", head, "--base", base]).returncode == 0:
        return True
    else:
        return False


def update_pr(*, head: str, title: str, body: str) -> None:
    run(["gh", "pr", "edit", head, "--title", title, "--body", body], check=True)


def create_pr(*, base: str, title: str, body: str) -> None:
    run(
        ["gh", "pr", "create", "--base", base, "--title", title, "--body", body],
        check=True,
    )


@click.command()
@click.option("--base")
@click.option("--head")
@click.option("--body")
@click.option("--title")
def main(base: str, head: str, body: str, title: str) -> None:
    try:
        create_pr(base=base, title=title, body=body)
    except CalledProcessError:
        update_pr(head=head, title=title, body=body)


if __name__ == "__main__":
    try:
        main()
    except CalledProcessError as err:
        exit(err.stdout + err.stderr)
