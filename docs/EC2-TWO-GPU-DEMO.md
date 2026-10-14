# Private two-GPU demo

This optional experiment distributes individual crop transcriptions over a P5
H100 and a G7e Blackwell GPU. Detection, crop creation, catalogue mapping and
atomic publication stay on the P5. The exact model, context, temperature,
confidence handling and first-pass processing are preserved. Normal local
execution and the native local launcher remain unchanged.

The two instances must be distinct, running in the same VPC and Region; use the
same Availability Zone/subnet to avoid cross-AZ transfer charges. Four independent
persistent Ollama services run on each GPU, with one request slot per service.
CUDA MPS is explicitly selected to let these processes share the GPU. The
[NVIDIA MPS guidance](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
describes where this scheduling can help; measured results decide whether it
helps this workload. This experiment does not guarantee a ten-second command.

## Launch and security

Use Ubuntu 24.04 LTS **amd64**, an SSH key, and **at least 100 GiB encrypted gp3
root storage per instance**. Paste the corresponding standalone
[P5 User Data](../scripts/demo-ec2-p5-user-data.sh) or
[G7e User Data](../scripts/demo-ec2-g7e-user-data.sh) into Advanced details → User
Data. Region-specific AMI IDs must be resolved in the selected Region. No IAM
role or AWS credentials are required by these scripts.

Restrict both instances' SSH TCP 22 to the presenter's current public IP `/32`.
On the G7e security group, allow **TCP 9443–9446 only from the P5 private IP
`/32`**. Permit matching outbound traffic if custom outbound restrictions exist.
Keep 11434 closed. The gateway binds only its RFC1918 address and requires a
client certificate; it exposes only model inventory/readiness and chat, with
no model pull, administrative or arbitrary proxy routes.

Wait for EC2 status checks and driver reboot. On each machine:

```bash
sudo cloud-init status --wait
sudo systemctl status uma-bct-demo.service --no-pager
/usr/local/bin/uma-bct-demo-health status
```

An HTTP response alone is insufficient. Readiness requires NVIDIA tooling,
Ollama's exact model digest, context 16384 and complete GPU residency. A cold
model, CPU spill, changed application revision or missing staged photo prevents
a successful live-run report.

## Prepare once, before presentation timing

From the repository on the Mac, set the current connection details:

```bash
export UMA_BCT_EC2_HOST='ubuntu@P5_PUBLIC_IPV4'
export UMA_BCT_EC2_KEY='/absolute/path/to/tokyo-key.pem'
export PHOTO='/absolute/path/to/approved-demo.jpg'
chmod 400 "$UMA_BCT_EC2_KEY"

.venv/bin/python scripts/demo_ec2_pair.py prepare \
  --peer-host 'ubuntu@G7E_PUBLIC_IPV4' --image "$PHOTO"
./scripts/demo-ec2.sh status --transcription-workers 8
```

Use `--peer-key /absolute/path/to/peer.pem` if the peer uses a different key.
The helper verifies instance identity/VPC, prepares both source-only application
bundles and all eight services, transfers **only the explicitly selected
photograph to the P5**, and generates private 30-day mutual TLS credentials.
The CA signing key stays in a temporary Mac directory and is never uploaded.
Only crop pixels leave the P5 over the private authenticated HTTPS link during
fresh processing. The G7e gateway forwards requests without persisting payloads
or logging photograph/prompt text. Originals remain locally unchanged;
byte-identical approved staging preserves source and staged SHA256 association.

Preparation and model loading are outside the live timer. Model residency does
not eliminate new vision processing or text generation. Each model uses the
production 16,384-token context, temperature zero, thinking disabled, persistent
keep-alive, and `OLLAMA_NO_CLOUD=1`. The pinned image and weights are unchanged.
Photographs, generated results, TLS secrets and model state stay outside Git,
release packages, hosted CI and container build contexts.

## One live command

```bash
./scripts/demo-ec2.sh run --image "$PHOTO" --verification never \
  --transcription-workers 8 --stream-results
```

This selects exactly one staged photograph by its original fingerprint, checks
both GPUs, performs new inference with `--reprocess` into a unique run folder,
shows existing CLI progress, and retrieves the catalogue, review HTML, crops and
run details over the same SSH connection. The complete transfer manifest is
validated before atomic local publication; relative crop links are retained.
Earlier runs are never overwritten. The command prints complete/partial/failed
status, CLI exit status, elapsed time and local results location. Add `--json`
for machine-readable stdout; progress stays on stderr. Add `--timings` only for
private per-request diagnostics without content payloads.

The external wall timer includes Python startup, SSH, readiness, CLI preflight,
detection, all crop transcription, validation, exports and complete retrieval.
Browser opening is not timed: open the downloaded review HTML afterward and
measure that separately if needed. A progress message, first token, partial run
or server-only duration does not satisfy the completion target.

`--verification never` explicitly omits the existing optional second-reading
pass. If requested, use `--verification always` and record its extra duration
separately; its behaviour remains unchanged.

## Rehearsal and evidence

```bash
uv run python scripts/demo_benchmark.py ec2 --image "$PHOTO" \
  --results-directory "$HOME/UMA-BCT-Demo/Benchmarks" --runs 5 \
  --verification never --transcription-workers 8 --stream-results --timings
```

Every trial performs fresh processing. The private feasibility report contains
every server/whole-command duration, median and slowest run, model/application
identity, settings, physical-box count, catalogue rows and fallbacks. Its
original five-second target flag is conservative and remains distinct from the
later requested approximate ten-second goal. Human accuracy is unreviewed until
every visible box, identifier, number, printed text and handwriting is checked
against the source/crops. Historical eleven-box model output is comparison
evidence, not ground truth; inferred missing-box rows are not physical boxes.

## Stop/start and troubleshooting

Stop both instances in the Console between rehearsals, after results are safely
retrieved. User Data runs at first launch; persistent systemd services restore
the scheduler/model services and peer gateway on later starts. Allow cold
reheating to finish before live timing. Update public IPv4/DNS and host-key
records carefully after stop/start. Verify the P5 private IP still matches the
G7e TCP rule; if instances/VPC/private IP changed, update the rule and repeat
pair preparation. Repeat preparation before the 30-day certificates expire.

```bash
sudo journalctl -u uma-bct-demo.service -u uma-bct-demo-mps.service \
  -u uma-bct-demo-parallel.service -u uma-bct-demo-peer.service -n 80 --no-pager
```

Readiness failures are explicit; status never secretly warms the model. A
connection timeout usually requires checking the current public/private IP and
security group. TLS failures require the prepared client certificate and matching
private server address. Interrupted upstream streams fail rather than being
accepted as completed transcriptions. Shared worker locks exclude reconfiguration
during active requests; credential versions are published under the exclusive
configuration lock. Preserve failed/partial folders for diagnosis and retry
into a new run; use `retrieve --run-id RECORDED_ID --archive-results` if the server
completed but Mac transfer failed.

For single-GPU operation, explicitly prepare/run with four or fewer workers on
the P5. Returning to normal scheduling uses `prepare --transcription-workers 1`
without `--gpu-mps`. An existing pair coordinator file affects only the explicit
eight-worker mode; keep eight reserved for this pair until it is removed during
manual cleanup. Preserve private results before terminating instances or deleting
EBS volumes. No script creates, stops or terminates AWS instances automatically.

## Budget

Official Tokyo rates checked **14 October 2026**, Linux On-Demand before tax:
P5.4xlarge **$8.60/hour**, G7e.2xlarge **$4.87752/hour**, combined **$13.47752/hour**.
Two public IPv4 addresses add **$0.01/hour**. Each 100 GiB gp3 volume costs
**$9.60/month**; two retained for 14 days cost about **$8.96** using a 30-day
planning month. Internet ingress is free. The first paid Tokyo internet-egress
tier is **$0.114/GB**, after the shared global 100 GB/month allowance; private
same-AZ traffic avoids cross-AZ charges. Extra gp3 IOPS/throughput, snapshots,
retained Elastic IPs, other instances and taxes must be added if applicable.

[Official Tokyo EC2/EBS price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/ap-northeast-1/index.json),
[Tokyo transfer price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/ap-northeast-1/index.json),
[public IPv4 pricing](https://aws.amazon.com/vpc/pricing/),
[EC2 transfer pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

With **$130 total credits**, reserve **$30** for presentation/contingency. A
planning envelope of six hours with both GPUs running across setup, benchmarks
and rehearsal costs **$80.93**, plus 14-day storage **$8.96** and 1 GB charged
egress **$0.114**: about **$90.00 before the reserve**. This leaves only about
**$10** for earlier instances' usage and other charges. One presentation hour
fits the $30 reserve; two consume about $26.98 before remaining contingencies.
Deduct charges already incurred before using this envelope. Stop idle older
instances in the Console; stopped instances still accrue retained EBS/snapshot
charges and any retained billable addresses. Credit eligibility and remaining
balance were not verified through account access: check Billing → Credits and
eligible EC2/EBS/network charges, expiration and exclusions. Credits are not a
hard spending cap. [Credit terms](https://aws.amazon.com/awscredits/).
