# ego-anno client

A Python 3.8+ client and command-line tool for submitting egocentric videos to
ego-anno and retrieving annotation results. It uses only the Python standard
library and verifies the server identity automatically.

## Install

```bash
pip install "git+https://github.com/DexGEM-Lab/ego-anno-client.git@v0.1.0"
```

Create an API key in the workspace; it is shown only once. Export it in the
terminal where you run the client:

```bash
export EGO_ANNO_API_KEY='<your API key>'
```

If the variable is not set, the CLI prompts for the key without echoing it.

## Commands

Submit a monocular video:

```bash
ego-anno submit --video ./video.mp4
```

Submit synchronized stereo videos with calibration:

```bash
ego-anno submit --left ./left.mp4 --right ./right.mp4 --calibration ./calibration.json
```

Add `--render` to request a rendered MP4. For stereo, `--method` optionally
selects `baseline`, `stereo_sj`, or `stereo_sjcf`; when omitted, the server
chooses the default. The command prints an idempotency key and the accepted job
ID. Add `--wait` to poll immediately, or save the job ID for the commands below.

```bash
ego-anno status <job-id>
ego-anno wait <job-id>
ego-anno download <job-id> -o annotation.json
```

For a render job, add `--rendered rendered.mp4` to `download`. `wait --interval`
sets the polling interval in seconds. Use `--idempotency-key` only to resume the
same logical submission; never reuse a key for different input.

Network errors and HTTP 429/5xx responses are retried with the same idempotency
key and request bytes. Downloads are written to a temporary file and atomically
moved into place only after the complete response arrives.

## Input requirements

- At most 1800 frames per view; longer videos are rejected, not truncated.
- Use native fps and resolution: no frame dropping, resampling, or rescaling.
  Videos must already be undistorted.
- Stereo left/right videos must match in resolution, frame count, fps, and
  duration; fps must be at least 30.
- Total upload size is at most 2 GiB. `calibration.json` is at most 1 MiB and
  contains `K_left`, `K_right`, `R_left_to_right`, `T_left_to_right_m`, and
  `image_size_wh`.

## Return codes

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
error, `2` for invalid command-line usage, and `130` for interruption.

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
