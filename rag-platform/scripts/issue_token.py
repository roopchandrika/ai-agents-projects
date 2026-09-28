"""Print a demo JWT for one of the seed users (there is no login flow in this project).

Usage: python -m scripts.issue_token nw-alice [--ttl 60]
       python -m scripts.issue_token --list
"""
import argparse
import json

from rag import config
from rag.acl import User
from rag.auth import issue_token


def main() -> None:
    users = json.loads((config.TENANT_DIR / "manifest.json").read_text(encoding="utf-8"))["users"]
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", nargs="?")
    parser.add_argument("--ttl", type=int, default=config.TOKEN_TTL_MINUTES, help="minutes")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.user_id:
        for u in users:
            print(f"{u['id']:<10} {u['tenant_id']:<10} {', '.join(u['roles'])}")
        return
    match = next((u for u in users if u["id"] == args.user_id), None)
    if match is None:
        raise SystemExit(f"Unknown user {args.user_id!r}. Use --list.")
    print(issue_token(User(match["id"], match["tenant_id"], tuple(match["roles"])), args.ttl))


if __name__ == "__main__":
    main()
