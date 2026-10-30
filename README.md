# ego-anno client

A Python 3.8+ client and command-line tool for submitting egocentric videos to
ego-anno and retrieving annotation results. It uses only the Python standard
library. The service endpoint and server certificate SHA-256 pin are built in;
no endpoint configuration, CA file, or CA flag is required. The client verifies
the pin before sending your API key or video data.

## 1. Download and install the client

Install Python 3.8+ with pip, preferably in a virtual environment. Get the
released client from the public [v0.1.0 Release](https://github.com/DexGEM-Lab/ego-anno-client/releases/tag/v0.1.0).
Install its wheel directly (no Git checkout required):

```bash
python3 -m pip install "https://github.com/DexGEM-Lab/ego-anno-client/releases/download/v0.1.0/ego_anno_client-0.1.0-py3-none-any.whl"
ego-anno --help
```

Alternatively, [download the wheel](https://github.com/DexGEM-Lab/ego-anno-client/releases/download/v0.1.0/ego_anno_client-0.1.0-py3-none-any.whl)
from that Release and install the local file:

```bash
python3 -m pip install ./ego_anno_client-0.1.0-py3-none-any.whl
```

If `ego-anno` is not on your PATH, use `python3 -m ego_anno` in its place.

## 2. Configure your API key

Create an API key in the workspace; it is shown only once. Export it in the
terminal where you run the client (the shell examples below use Bash):

```bash
export EGO_ANNO_API_KEY='<your API key>'
```

If the variable is not set, the CLI prompts for the key without echoing it.
Keep your key private; do not commit it to a repository.

## 3. Submit a video

Choose one input mode. Submit a monocular video:

```bash
ego-anno submit --video ./video.mp4
```

Or submit synchronized stereo videos with calibration:

```bash
ego-anno submit --left ./left.mp4 --right ./right.mp4 --calibration ./calibration.json --render
```

`--render` requests the optional rendered MP4 and works for either input mode;
omit it if you only need the annotation JSON. For stereo only, `--method` can
select `baseline`, `stereo_sj`, or `stereo_sjcf`; when omitted, the server
chooses the default. Do not combine `--video` with stereo options.

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
Never reuse a key for different input or modify files while uploading.

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

For a job submitted with `--render`, download both outputs:

```bash
ego-anno download "$JOB_ID" -o annotation.json --rendered rendered.mp4
```

`-o` defaults to `annotation.json`. Output directories must already exist.
Each file is written to a temporary file and atomically moved into place after
the complete response arrives, replacing an existing destination. The two
files download separately: if the rendered download fails, the completed JSON
remains on disk.

## Input requirements

The server validates these input requirements:

- At most 1800 frames per view; longer videos are rejected, not truncated.
- Use native fps and resolution: no frame dropping, resampling, or rescaling.
  Videos must already be undistorted.
- Stereo left/right videos must match in resolution, frame count, fps, and
  duration; fps must be at least 30.
- Total upload size is at most 2 GiB. `calibration.json` is at most 1 MiB and
  contains `K_left`, `K_right`, `R_left_to_right`, `T_left_to_right_m`, and
  `image_size_wh`.

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
job_id = client.submit_mono("./video.mp4", render=True)
client.wait(job_id)
client.download(job_id, "annotation.json", rendered="rendered.mp4")
```

You can also pass `api_key=` explicitly. Stereo submission is available through
`submit_stereo(left, right, calibration, method=...)`; `status(job_id)` returns
the decoded status document. Client failures raise `ClientError`; a certificate
identity mismatch raises its `PinError` subclass before any request bytes are
sent.

## Development

```bash
python -m unittest discover -s tests -v
python -m build
```

The test suite is offline and uses only the standard library.
