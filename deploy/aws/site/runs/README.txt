Smoke Rescue results site

Serve deploy/aws/site/ as the document root. No build step or external assets
are required. Run IDs contain letters, digits, underscores and hyphens only.

Expected layout
  index.json
  <run_id>/walkthrough.mp4
  <run_id>/agent_trace.jsonl
  <run_id>/compare.json

The catalog is an object with a runs array, newest publication first.
Example: {"runs": [{"id": "haze_test20", "label": "20-second haze test"}]}
All URLs are relative to the site root. The picker controls video, trace and
comparison together. The site also works below a local URL prefix.

The trace is the unmodified JSONL from brain.agent. Records group by step.
step_start carries robot, time and event. tool carries name, arguments,
result_id, verdict, refused, reason and error. step_end carries orders,
refusals, summary and latency_s. Commander steps may omit step_start.
The explorer preserves approval records, model errors and incomplete steps.
Raw records include model replies and their individual latencies. Tool filters
use executed tool records. A robot filter also includes commander orders for
that robot. The explorer renders 30 matching steps at a time.

compare.json is the top-level array written by brain.compare_runs --out.
Pass both recordings, normally baseline first and agents second. Do not replace
missing measurements with zero. Agent labels use nonempty agent metadata.
The table shows each source recording and does not infer a performance gain.
Heat and collision data retain their per-robot keys and units from the evaluator.
A null first_found_s means no real survivor was found.

Publish from the repository root, with an existing comparison under data/.
  python3 tools/publish_site.py haze_test20 --recording data/recordings/unity_haze_agents_test20 --walkthrough data/evidence/walkthrough_agents_test20_chat.mp4 --compare data/runs/haze_test20/compare.json

If no comparison exists, --baseline runs the existing evaluator using the brain
virtual environment. It writes the comparison only into the published run.
  python3 tools/publish_site.py haze_test20 --recording data/recordings/unity_haze_agents_test20 --walkthrough data/evidence/walkthrough_agents_test20_chat.mp4 --baseline data/recordings/unity_haze_auto_test20

Defaults are data/recordings/<run_id>/agent_trace.jsonl,
data/evidence/walkthrough_<run_id>.mp4 and data/runs/<run_id>/compare.json.
Use --label for a readable picker label. Re-publishing the same ID replaces its
three files and moves it to the front of the catalog. Sources must be under
data/. The publisher never uploads, accesses credentials or edits recordings.
The --baseline option needs brain/.venv with the project's dependencies.

Local preview
  python3 -m http.server 8000 --directory deploy/aws/site
Open http://localhost:8000/. The recorded sections work without the API.
A static server returns an API unavailable badge and cannot analyze uploads.

Deployment
The existing deploy/aws/deploy.sh syncs this entire directory, including runs/,
to the site S3 bucket. This script does not invoke deploy.sh or AWS. CloudFront
routes /api/* to the agent server and strips /api. POST /api/vision/<tool> sends
a raw PNG with Content-Type image/png. Thermal input is single-channel uint16,
kelvin times ten, not a colourized image. The upload limit in the UI is 10 MiB.
GET /api/health is checked on load and with the Check API button. It reports
reachability and configuration, not a Bedrock inference readiness test.
The results explorer reads published traces, not the live GET /api/trace route.
The site displays approvals as evidence and does not approve robot actions.

Sample evidence
haze_test20 is the recorded 20-second test. Its model is local Qwen3 through
Ollama, not Bedrock. Its fire-spread call reports a chronological-frame error.
Do not treat the model's later summary as a successful fire-spread measurement.
This short test is not evidence of a long-run rescue improvement. The inline SVG
describes the configured AWS deployment, not a verified live deployment.
