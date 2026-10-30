"""Command-line interface for ego-anno."""
import argparse
import getpass
import http.client
import json
import os
import sys

from .client import EgoAnnoClient
from .transport import ClientError, check_response, connect, headers


def _ping():
    conn = connect()
    try:
        headers(conn, "GET", "/healthz", None)
        response = conn.getresponse()
        check_response(response, 200)
        response.read(1024 * 1024)
        print("HTTP 200 /healthz; certificate pin verified")
    finally:
        conn.close()


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if arguments == ["_ping"]:
            _ping()
            return 0
        parser = argparse.ArgumentParser(
            prog="ego-anno", description="Submit and retrieve ego-anno annotations.")
        commands = parser.add_subparsers(dest="command", required=True)
        sub = commands.add_parser("submit", help="submit video for annotation")
        sub.add_argument("--video")
        sub.add_argument("--left")
        sub.add_argument("--right")
        sub.add_argument("--calibration")
        sub.add_argument("--method", choices=("baseline", "stereo_sj", "stereo_sjcf"))
        sub.add_argument("--idempotency-key")
        sub.add_argument("--wait", action="store_true")
        status = commands.add_parser("status", help="show job status")
        status.add_argument("job_id")
        waiter = commands.add_parser("wait", help="wait for a result")
        waiter.add_argument("job_id")
        waiter.add_argument("--interval", type=float, default=10)
        dl = commands.add_parser("download", help="download results")
        dl.add_argument("job_id")
        dl.add_argument("-o", default="annotation.json")
        args = parser.parse_args(arguments)
        if args.command == "submit":
            if args.video:
                if args.left or args.right or args.calibration or args.method:
                    parser.error("--video cannot be combined with stereo options")
            elif not all((args.left, args.right, args.calibration)):
                parser.error("provide --video or all of --left, --right and --calibration")
        if args.command == "wait" and args.interval <= 0:
            parser.error("--interval must be positive")
        key = os.environ.get("EGO_ANNO_API_KEY") or getpass.getpass("API key: ")
        client = EgoAnnoClient(key)
        if args.command == "submit":
            if args.video:
                client.submit_mono(args.video,
                                   idempotency_key=args.idempotency_key, wait=args.wait)
            else:
                client.submit_stereo(args.left, args.right, args.calibration, method=args.method,
                                     idempotency_key=args.idempotency_key,
                                     wait=args.wait)
        elif args.command == "status":
            print(json.dumps(client.status(args.job_id), indent=2))
        elif args.command == "wait":
            client.wait(args.job_id, args.interval)
        else:
            client.download(args.job_id, args.o)
        return 0
    except (ClientError, ValueError, KeyError, OSError, http.client.HTTPException) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
