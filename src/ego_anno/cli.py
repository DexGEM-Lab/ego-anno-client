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
        sub.add_argument("--pipeline", choices=("dag", "e2e"), default="dag",
                         help="dag (default) or e2e (Cosmos3 hand+head; needs ffmpeg)")
        sub.add_argument("--prompt", help="e2e only: task description, at most 1000 characters")
        sub.add_argument("--camera", help="e2e only: camera.json with source_size_wh and source_K")
        sub.add_argument("--save-inputs", metavar="DIR",
                         help="e2e only: preserve the exact upload and camera for offline rendering")
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
        render = commands.add_parser("render", help="render a downloaded result offline")
        render.add_argument("--result", required=True, help="downloaded result JSON (DAG or Cosmos e2e)")
        render.add_argument("--inputs", help="directory created by submit --save-inputs (e2e bundle)")
        render.add_argument("--video", help="the exact video the job was submitted with")
        render.add_argument("--camera", help="Cosmos only: calibration JSON for the processed video")
        render.add_argument("--mano-dir", help="licensed MANO asset directory for surface rendering")
        render.add_argument("--fps", type=float, help="DAG only: source frame rate when it cannot be probed")
        render.add_argument("--world-up", metavar="X,Y,Z",
                            help="gravity-up direction in the result's world frame; estimated when omitted")
        render.add_argument("--layout", choices=("three-pane", "two-pane"), default="three-pane",
                            help="three-pane (default, 1920x1080): overlay | follow view | upper-body "
                                 "skeleton; two-pane: overlay | follow view at the source size")
        render.add_argument("-o", "--output", required=True, help="output H.264 MP4")
        render.add_argument("--width", type=int)
        render.add_argument("--height", type=int)
        args = parser.parse_args(arguments)
        if args.command == "submit":
            if args.pipeline == "e2e":
                if not args.video:
                    parser.error("--pipeline e2e requires --video")
                if args.left or args.right or args.calibration or args.method:
                    parser.error("--pipeline e2e cannot be combined with stereo options or --method")
            elif args.prompt is not None or args.camera is not None or args.save_inputs is not None:
                parser.error("--prompt, --camera and --save-inputs require --pipeline e2e")
            if args.video:
                if args.left or args.right or args.calibration or args.method:
                    parser.error("--video cannot be combined with stereo options")
            elif not all((args.left, args.right, args.calibration)):
                parser.error("provide --video or all of --left, --right and --calibration")
        if args.command == "wait" and args.interval <= 0:
            parser.error("--interval must be positive")
        if args.command == "render":
            from .rendering import render_result, render_video_result
            if (args.inputs is None) == (args.video is None):
                parser.error("provide exactly one of --inputs DIR (saved e2e bundle) or --video FILE")
            if args.inputs is not None:
                for name in ("camera", "mano_dir", "fps", "world_up"):
                    if getattr(args, name) is not None:
                        parser.error("--{} applies only to the --video form".format(name.replace("_", "-")))
                output = render_result(args.result, args.inputs, args.output,
                                       args.width, args.height, layout=args.layout)
            else:
                output = render_video_result(args.result, args.video, args.output,
                                             width=args.width, height=args.height,
                                             camera=args.camera, fps=args.fps, mano_dir=args.mano_dir,
                                             world_up=args.world_up, layout=args.layout)
            print("Rendered {}".format(output))
            return 0
        key = os.environ.get("EGO_ANNO_API_KEY") or getpass.getpass("API key: ")
        client = EgoAnnoClient(key)
        if args.command == "submit":
            if args.pipeline == "e2e":
                client.submit_e2e(args.video, prompt=args.prompt, camera=args.camera,
                                  idempotency_key=args.idempotency_key, wait=args.wait,
                                  save_inputs=args.save_inputs)
            elif args.video:
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
