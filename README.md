# ego-anno client

A Python 3.8+ client and command-line tool for submitting egocentric videos to
ego-anno, retrieving annotation results, and optionally rendering Cosmos e2e
results offline. Submission/download use only the Python standard library;
the optional `[render]` extra installs NumPy and OpenCV. The service endpoint
and certificate SHA-256 pin are built in: no IP configuration, SSH access,
private repository, CA file, or CA flag is needed. Certificate identity is
checked before the API key or video is sent.

## Start here: Cosmos annotation → JSON → MP4

You need **your own API key and local video**, plus Python and system FFmpeg
4.4+ with `ffmpeg`, `ffprobe`, and the `libx264` encoder. Create an account
and API key at **https://anno.dex-gem.ai** (API keys page); the account needs
annotation credits. Never put a key in chat, logs, source code, or a Git commit.
If you have no key/video, obtain them from the user rather than inventing them.

Use a mono clip with 15–30 fps, duration at most 20 s, and short side
640–1440 px. A clip of roughly 2–16 seconds is convenient for a first test.
From a fresh directory, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "ego-anno-client[render] @ https://github.com/DexGEM-Lab/ego-anno-client/releases/download/v0.3.0/ego_anno_client-0.3.0-py3-none-any.whl"

# Set EGO_ANNO_API_KEY privately in the environment before running the client.
.venv/bin/ego-anno submit --pipeline e2e --video ./clip.mp4 \
  --save-inputs ./run/inputs --wait

JOB_ID=$(.venv/bin/python -c 'import json; print(json.load(open("run/inputs/submission.json"))["annotation_job_id"])')
.venv/bin/ego-anno download "$JOB_ID" -o ./run/annotation.json
.venv/bin/ego-anno render --result ./run/annotation.json \
  --inputs ./run/inputs -o ./run/preview.mp4 --width 1600 --height 900
```

On Windows, use `.venv\\Scripts\\python.exe` and `.venv\\Scripts\\ego-anno.exe`.
If known, add `--camera ./camera.json` to **submit**; the client saves its
rescaled calibration for render automatically. Without calibration the MP4
shows the source video and predicted 3D skeleton/trajectory, but does **not**
invent a source-video projection. Without beta, no MANO mesh is rendered.

Success means: the job is `succeeded`, downloaded JSON has schema
`cosmos-e2e-result/v1` with `num_frames - 1` rows of 147 finite values, and
`preview.mp4` decodes as H.264 at 15 fps with `num_frames` frames.

The default `dag` pipeline renders from the video you submitted instead of a
saved bundle:

```bash
ego-anno submit --video ./clip.mp4 --wait
JOB_ID=...  # printed by submit
ego-anno download "$JOB_ID" -o ./run/annotation.json
ego-anno render --result ./run/annotation.json --video ./clip.mp4 -o ./run/preview.mp4
```
Submission alone, or `--wait` alone, does not download a result.
Each `--save-inputs` directory must be new; choose a new run directory for a
new submission. Keep existing results and keys private.


Two pipelines share the same account, API key, and credit balance:

| | `dag` (default) | `e2e` |
| --- | --- | --- |
| Model | ego-anno annotation DAG | Cosmos3 hand147 end-to-end model |
| Input | Mono video, or stereo videos + calibration | Mono video only |
| Video limits | Up to 1800 frames per view at native fps and resolution | 15–30 fps, at most 20 s, short side 640–1440 px |
| Local processing | None; the file is uploaded as-is | Resampled to 15 fps, first 17–241 frames kept, short side above 1080 px scaled to 1080 |
| Extra inputs | `--method` (stereo only) | Optional `--prompt` and `--camera` intrinsics |
| Billing | Whole seconds of video duration | `max(1, ceil(duration × 0.1))` seconds of the uploaded 15 fps clip |
| Result | `ego.annotation.product_json.v3` annotation JSON | `cosmos-e2e-result/v1` JSON with 15 Hz camera motion and 147-D hand actions |
| Local render | `ego-anno render --video <the submitted video>` | `ego-anno render --inputs <saved inputs>` |
| Needs | Python only | Python plus FFmpeg (`ffmpeg` and `ffprobe` on PATH) |

The e2e billing ratio (default 0.1) is a deployment setting.

## 1. Download and install the client

Install Python 3.8+ with pip, preferably in a virtual environment. Get the
released client from the public [v0.3.0 Release](https://github.com/DexGEM-Lab/ego-anno-client/releases/tag/v0.3.0).
Install its wheel directly (no Git checkout required):

```bash
python3 -m pip install "https://github.com/DexGEM-Lab/ego-anno-client/releases/download/v0.3.0/ego_anno_client-0.3.0-py3-none-any.whl"
ego-anno --help
```

Alternatively, [download the wheel](https://github.com/DexGEM-Lab/ego-anno-client/releases/download/v0.3.0/ego_anno_client-0.3.0-py3-none-any.whl)
from that Release and install the local file:

```bash
python3 -m pip install ./ego_anno_client-0.3.0-py3-none-any.whl
```

Rendering `dag` results locally (`--video`, section 6) is in this source tree
and ships in the next release. Until that release is published, install the
renderer from the repository, which needs no build backend beyond `pip`:

```bash
python3 -m pip install "ego-anno-client[render] @ git+https://github.com/DexGEM-Lab/ego-anno-client@main"
```

If `ego-anno` is not on your PATH, use `python3 -m ego_anno` in its place.

The e2e pipeline additionally needs FFmpeg 4.4 or newer with libx264 (for
example `sudo apt install ffmpeg` or `brew install ffmpeg`); the client runs
the `ffprobe` and `ffmpeg` found on PATH and stops with an error if either is
missing. The DAG pipeline does not use FFmpeg.

## 2. Configure your API key

Create an API key at [anno.dex-gem.ai](https://anno.dex-gem.ai); it is shown
only once. Export it in the terminal where you run the client:

```bash
export EGO_ANNO_API_KEY='<your API key>'
```

If the variable is not set, the CLI prompts for the key without echoing it.
Keep your key private; do not commit it to a repository.

## 3. Submit a video

Choose one input mode. Submit a monocular video to the default DAG pipeline:

```bash
ego-anno submit --video ./video.mp4
```

Or submit synchronized stereo videos with calibration:

```bash
ego-anno submit --left ./left.mp4 --right ./right.mp4 --calibration ./calibration.json
```

For stereo only, `--method` can
select `baseline`, `stereo_sj`, or `stereo_sjcf`; when omitted, the server
chooses the default. Do not combine `--video` with stereo options.

### e2e pipeline

Submit a monocular video to the Cosmos3 hand147 model with `--pipeline e2e`:

```bash
ego-anno submit --video ./video.mp4 --pipeline e2e
ego-anno submit --video ./video.mp4 --pipeline e2e \
  --prompt "pick up the cup" --camera ./camera.json --wait
```

`--prompt` is an optional task description of 1–1000 characters (not blank).
`--camera` is an optional JSON file with the pinhole intrinsics of the source
video, which improves wrist depth:

```json
{"source_size_wh": [2560, 1440], "source_K": [[1800, 0, 1279.5], [0, 1800, 719.5], [0, 0, 1]]}
```

`source_size_wh` must equal the video's displayed size (width and height after
any rotation metadata is applied), and `source_K` must be finite with zero skew
and positive focal lengths. e2e cannot be combined with stereo options or
`--method`; `--prompt` and `--camera` are only accepted with `--pipeline e2e`.

Before uploading, the client checks and converts the video in a temporary
directory with FFmpeg:

- Frame rate above 30 fps or below 15 fps, duration above 20 s, or a short
  side below 640 px or above 1440 px is rejected with an error.
- The video is resampled by timestamp to exactly 15 fps (each output frame is
  the source frame nearest to its 1/15 s slot), the first
  min(241, available) frames are kept, and the count is trimmed to the
  largest N with (N - 1) divisible by 4. Fewer than 17 frames (about 1.07 s)
  is rejected. A clip with 240 sampled frames is trimmed to 237; a clip with
  at least 241 sampled frames keeps 241 (spanning 16.0 s); a 10 s clip uses 149.
- A short side above 1080 px is scaled down to 1080 px, keeping the aspect
  ratio with even dimensions (2560x1440 becomes 1920x1080). An odd width or
  height at or below 1080 px loses its last pixel column or row. Rotation
  metadata is applied to the pixels.
- Camera intrinsics are scaled with the image using the pixel-center
  convention: fx' = fx·sx, cx' = (cx + 0.5)·sx − 0.5, and the same for y.
- The upload is H.264 MP4, yuv420p, constant 15 fps, CRF 12, no audio,
  `+faststart`. It is re-probed and must contain exactly N frames at 15 fps
  at the expected size before upload.

The server enforces the converted form (15 fps, 17–241 frames with (N - 1)
divisible by 4, short side 640–1080, long side at most 2560) and answers
`422 invalid_e2e_input`, `invalid_camera`, or `invalid_pipeline` otherwise;
`503 annotation_e2e_disabled` means the deployment has e2e switched off.

The client prints upload progress to stderr and these identifiers to stdout:

```text
Idempotency-Key: <submission UUID>
annotation_job_id: <job UUID>
```

Save the `annotation_job_id` UUID, not the idempotency key, for the commands
below. Submission acceptance means the job is queued, not finished. Add
`--wait` to either submit command to poll immediately after acceptance.

To retry the same logical submission after an interrupted connection, pass
`--idempotency-key <submission UUID>` with the same input files and options.
Never reuse a key for different input or modify files while uploading. For
e2e, the key covers the converted video together with the pipeline, prompt,
and camera fields; the conversion is deterministic for the same input file
and FFmpeg build, so rerunning the same command replays the original job.

## 4. Check status and wait

Replace `<job UUID>` with the `annotation_job_id` printed by submit:

```bash
JOB_ID='<job UUID>'
ego-anno status "$JOB_ID"
ego-anno wait "$JOB_ID" --interval 10
```

`status` prints the current JSON status. `wait` prints status updates while the
job is `queued` or `running`; it succeeds only when `status` is `succeeded` and
`result_available` is `true`. A failed job or another terminal state produces
an error including `failure_code`. The default polling interval is 10 seconds;
`--interval` must be positive. Waiting does not download files.

## 5. Download the results

After a successful wait, download the annotation:

```bash
ego-anno download "$JOB_ID" -o annotation.json
```

`-o` defaults to `annotation.json`. Output directories must already exist.
The file is written to a temporary file and atomically moved into place after
the complete response arrives, replacing an existing destination.

e2e jobs download a `cosmos-e2e-result/v1` JSON document with `fps` 15,
`num_frames` N, and `action`: N - 1 rows of 147 physical values (row k is the
camera motion from frame k to k + 1 and both hands at frame k + 1), plus
`channel_layout`, `frame_alignment`, `model`, and `receipt` describing the
inference. The server returns JSON only; use the offline render command below
to produce an MP4.

## 6. Render results offline

A result JSON is the complete product of a job. The server-side MP4 is an
opt-in per-job option for API callers (`render=true` on submission); this
client never requests it, so submissions stay JSON-only and an acknowledged
job costs no rendering time. Render locally when you want a video:

| You have | Command |
| --- | --- |
| A `dag` result and the exact video you submitted | `ego-anno render --result annotation.json --video clip.mp4 -o preview.mp4` |
| An `e2e` result and its saved inputs | `ego-anno render --result annotation.json --inputs ./run/inputs -o preview.mp4` |

Install this release with the `[render]` extra (see the quickstart). The wheel
contains the renderer and physical-action decoder; no private repository,
GPU, API key, or network connection is needed to render. System FFmpeg with
libx264 is required. The output is H.264/yuv420p/faststart and is published
atomically: rendering never overwrites the result, the video, or an existing
destination.

### dag results

```bash
ego-anno render --result ./annotation.json --video ./clip.mp4 -o ./preview.mp4
```

The `dag` result is `ego.annotation.product_json.v3`: per-frame 21-joint hand
coordinates, their pixel projections, MANO shape parameters, the camera pose,
and the subtitle rows. `--video` must be **the exact file you submitted** — the
DAG decodes at native resolution and never resamples or trims, so those pixel
projections only describe that file. The command refuses a video whose frame
count differs from `Metadata.帧数`, and `--fps` must agree with the video's own
frame rate when you supply it (it is only needed when the container reports no
frame rate).

The preview is the same two-pane layout the server produces: the source video
with the measured joint skeletons overlaid, and a 3D pane with the camera
trajectory and hands, above a subtitle band. Optional `--width`/`--height`
change only the layout, never the calibration.

MANO hand surfaces require the licensed MANO model, which is not distributed
with this client. Without `--mano-dir` the render draws the measured 21-joint
skeletons that the result already contains, in both panes, and prints that
limitation; nothing is fabricated and no surface is invented. Point
`--mano-dir` at a directory containing `MANO_LEFT.pkl`, `MANO_RIGHT.pkl`, and
`mano_runtime.npz` (the `render_pack/mano` directory of the private
ego-compute checkout, if you have licensed access) to also draw the fitted
surfaces.

### e2e results

During submission, `--save-inputs DIR` (e2e only) retains:

- `video.mp4`: exactly the bytes uploaded after frame sampling and resizing.
- `camera.json`: processed-pixel intrinsics, **only** when supplied.
- `submission.json`: public job ID, idempotency key, frame count, dimensions,
  fps and optional task prompt. It never stores the API key.

The directory is saved before upload. If upload fails, its job ID remains
null and the inputs/idempotency key remain available. Existing directories
are never overwritten. To retry a submission with the same idempotency key,
omit `--save-inputs` or choose a different save directory; do not replace the
original bundle with unrelated footage.

```bash
ego-anno render --result ./run/annotation.json --inputs ./run/inputs \
  -o ./run/preview.mp4
```

Optional `--width`/`--height` change only the output layout, not calibration.
Use the saved inputs belonging to this result, **not** the original
30 fps/resized video or unscaled camera. Frame count, fps and processed
dimensions must match. The original temporary video is deleted without
`--save-inputs`, so retain the bundle when you intend to visualize.

The preview keeps all N video frames. I0 has no hand prediction; action row k
is shown on I(k+1). Translations/joints are already physical meters and are
not inverse-normalized again. Camera motion is integrated with I0 as world
origin. Hand visibility/object classes and MANO beta are not inferred.
The command writes H.264/yuv420p/faststart MP4 atomically and preserves an
existing output if rendering fails. It does not overwrite bundle inputs.

The renderer snapshot's source commit/hashes are recorded in the installed
`ego_anno/rendering/provenance.json`; this is provenance, not a runtime download.


## Input requirements

For the DAG pipeline, the server validates these input requirements:

- At most 1800 frames per view; longer videos are rejected, not truncated.
- Use native fps and resolution: no frame dropping, resampling, or rescaling.
  Videos must already be undistorted.
- Stereo left/right videos must match in resolution, frame count, fps, and
  duration; fps must be at least 30.
- Total upload size is at most 2 GiB. `calibration.json` is at most 1 MiB and
  contains `K_left`, `K_right`, `R_left_to_right`, `T_left_to_right_m`, and
  `image_size_wh`.

For the e2e pipeline, see [e2e pipeline](#e2e-pipeline).

## Errors and exit codes

Errors are printed to stderr as `Error: ...`. Network errors and HTTP 429/5xx
responses are retried up to 10 attempts, with exponential backoff capped at
60 seconds. Automatic submission retries reuse the same idempotency key and
request bytes. Other HTTP errors and certificate pin mismatches are terminal.

| HTTP code | Meaning | Action |
| --- | --- | --- |
| 200 | Query or download succeeded | Proceed. |
| 202 | Submission accepted and queued | Store the job ID and poll. |
| 400 / 422 | Malformed request / invalid input | Fix the input and resend. |
| 401 / 403 | Key invalid, revoked, or scope refused | Check the API key and its scope. |
| 404 | Job not found or not owned | Check the job ID. |
| 409 | Idempotency conflict, no credit, or result not ready | Use a new key for a new job, contact the operator for credit, or poll first. |
| 413 | Upload too large | Meet the input limits. |
| 429 / 5xx | Backpressure / server fault | The client retries with the same key and bytes. |

CLI process exit codes are `0` for success, `1` for a client/server/network
error (including a failed job), `2` for invalid command-line usage, and `130`
for interruption. Job IDs and explicitly supplied idempotency keys must be
valid UUIDs; invalid UUIDs exit with code `1`.

## Python usage

```python
from ego_anno import EgoAnnoClient

client = EgoAnnoClient()  # reads EGO_ANNO_API_KEY
job_id = client.submit_mono("./video.mp4")
client.wait(job_id)
client.download(job_id, "annotation.json")
```

You can also pass `api_key=` explicitly. Stereo submission is available through
`submit_stereo(left, right, calibration, method=...)`, and the e2e pipeline
through `submit_e2e(video, prompt=None, camera=None)`, where `camera` is a path
to the camera JSON (or the equivalent dict). `status(job_id)` returns
the decoded status document. Client failures raise `ClientError`; a certificate
identity mismatch raises its `PinError` subclass before any request bytes are
sent.

## Development

```bash
python -m unittest discover -s tests -v
python -m build
```

The test suite is offline and uses only the standard library; its FFmpeg tests
run when `ffmpeg` and `ffprobe` are installed and are skipped otherwise.

Two opt-in checks render real saved artifacts instead of synthetic fixtures and
diff the result against the server-rendered product; both are skipped unless
their environment variable is set:

```bash
# A directory holding annotation-result.json, rgb.mp4 and (optionally)
# annotation-rendered.mp4 from one real job.
EGO_ANNO_REAL_RESULT_DIR=/path/to/artifacts python -m pytest tests/test_rendering.py -k RealSavedArtifactTests

# Adds the licensed-asset parity check (draws MANO surfaces too).
EGO_ANNO_MANO_DIR=/path/to/render_pack/mano EGO_ANNO_REAL_RESULT_DIR=/path/to/artifacts \
  python -m pytest tests/test_rendering.py -k licensed
```
