# Temporary private EC2 demonstration

This is an opt-in exception for **explicitly approved photographs**. Normal local
launchers and processing defaults remain unchanged. The server runs the existing
CLI beside a persistent Ollama service; the Mac controls it through SSH. The
Ollama API is never published. There is no automatic AWS launch or stop action.

**The five-second target is unverified on EC2.** Timing evidence is initially
accuracy-unverified. A complete CLI status or 11 records does not prove every
physical box and handwritten annotation was transcribed correctly. Do not
represent historical model results as ground truth.

## Console launch checklist

1. Select **Asia Pacific (Sydney), ap-southeast-2**. In **Service Quotas → Amazon
   EC2 → Running On-Demand G and VT instances**, check the applied regional
   allowance minus current usage. `g6.2xlarge` requires eight available vCPUs;
   `g6.xlarge` requires four. Melbourne's approval does not transfer to Sydney.
2. In EC2's instance-type browser, confirm the candidate is offered in the chosen
   Availability Zone. AWS lists G6 in Sydney, but offerings and quotas do not
   guarantee live capacity. Try another offered Sydney subnet/AZ if launch fails
   for capacity; do not reserve capacity or switch to another billable type
   without making that cost decision explicitly.
3. Launch one **On-Demand g6.2xlarge** initially, named `uma-bct-demo`. Choose
   **Canonical Ubuntu Server 24.04 LTS, HVM, 64-bit x86**, without Ubuntu Pro.
   Canonical's AMI owner is `099720109477`; select the current official image in
   Sydney rather than copying an AMI ID from another Region. No IAM role is needed.
4. Select a dedicated SSH key pair. Keep its private key on your Mac (`chmod 400`).
   Use a public subnet with an internet-gateway route and automatic public IPv4.
   The security group permits **TCP 22 from your current public IP/32 only**.
   Do not add 11434, HTTP, or HTTPS inbound rules. No NAT gateway is needed.
5. Provision **100 GiB encrypted gp3 root EBS**, default AWS-managed encryption,
   **3,000 IOPS and 125 MiB/s**, with deletion on termination. Enable IMDSv2.
   Filesystem expansion cannot enlarge the EBS volume. User Data checks the root
   block device and requires at least 30 GiB free; fix inadequate storage in the
   Console before proceeding.
6. In **Advanced details → User data**, paste the **whole**
   [`demo-ec2-user-data.sh`](../scripts/demo-ec2-user-data.sh). It is self-contained
   and below EC2's 16 KiB raw User Data limit. Do not put credentials, tokens, SSH
   private keys, or photograph download URLs there. Repository/artifact deployment
   is performed later by the authenticated Mac preparation command.
7. Allow package installation, the scheduled driver reboot, model download, and
   loading to finish. A running instance, two EC2 status checks, or an HTTP health
   response does not establish model/GPU readiness.

The script uses Docker's signed apt repository, Ubuntu's NVIDIA 580 server driver,
NVIDIA Container Toolkit 1.20.1-1, and the repository's digest-pinned Ollama 0.32.9
image. The model must be `qwen3-vl:8b-instruct-q4_K_M`, matching the established
local digest `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`.
A different digest is a readiness failure; do not silently accept changed weights.

Official guidance: [regional offerings](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html),
[regional quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html),
[Docker installation](https://docs.docker.com/engine/install/ubuntu/),
[Canonical driver installation](https://ubuntu.com/aws/docs/aws-how-to/instances/install-nvidia-drivers/),
[NVIDIA Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[Compose GPU access](https://docs.docker.com/compose/how-tos/gpu-support/).

## Preparation and one-command operation

In your existing locked checkout, configure connection details. Replace the
placeholders; these are not repository credentials or a private download URL.

```bash
export UMA_BCT_EC2_HOST="ubuntu@<current-public-DNS-or-IPv4>"
export UMA_BCT_EC2_KEY="$HOME/.ssh/<dedicated-key>.pem"
export UMA_BCT_DEMO_RESULTS="$HOME/UMA-BCT-Demo/Results"
export PHOTO="/absolute/path/to/one-approved-photo.jpg"
chmod 400 "$UMA_BCT_EC2_KEY"
```

First establish the SSH host key using an ordinary SSH connection and verify its
fingerprint against the EC2 system console's SSH-key fingerprint output. The
wrapper uses strict host-key checking and unattended public-key authentication;
it never disables host verification or accepts changed keys automatically.

```bash
ssh -i "$UMA_BCT_EC2_KEY" "$UMA_BCT_EC2_HOST"
# On EC2, inspect preparation if necessary:
sudo tail -n 100 /var/log/uma-bct-demo-setup.log
sudo journalctl -u uma-bct-demo.service --no-pager
/usr/local/bin/uma-bct-demo-health status
exit

./scripts/demo-ec2.sh prepare --image "$PHOTO"
./scripts/demo-ec2.sh status
./scripts/demo-ec2.sh run --image "$PHOTO"
```

`prepare` explicitly reheats and deploys the application. It transfers a source
allowlist (Dockerfile, lock/metadata, README, application Python and the controller),
builds the existing application Dockerfile on EC2, and transfers **only the selected
JPEG** over SSH/SCP. It never archives the whole checkout. The deployment records
HEAD, dirty state, source fingerprints and the built image identity. No repository
token or model weights are transferred from your Mac. Python 3.12 is used inside
the existing application image. The Mac wrapper needs Python 3.12+ and OpenSSH;
it prefers the checkout's `.venv/bin/python` and accepts `UMA_BCT_DEMO_PYTHON`.

The original JPEG stays intact on the Mac. The staged copy is byte-identical,
including its existing EXIF, under `/home/ubuntu/demo_images/<SHA-256>/`.
`Source.json` records association, source and staged hashes. This is the selected
unchanged-copy policy; there is no resampling, re-encoding or inference shortcut.

`run` selects one image by its local SHA-256 and verifies the remote association
and bytes. It checks the pinned model, production context, NVIDIA availability,
actual service device access and full GPU residency. A cold/missing model or image
fails with preparation instructions. It does not pull, build, warm or start EC2.
It invokes the existing CLI with a new result destination and `--reprocess`,
so each demonstration performs fresh detection and every crop transcription.

**First-pass demonstration uses explicit `--verification never`.** Normal
missing-box rows, fallbacks, confidence, review flags and atomic exports remain
enabled. Qwen detection → crops → sequential Qwen crop transcription → mapping
and export remain the processing sequence. Existing recovery fallbacks remain
available and are recorded. No saved inference is replayed.

The terminal shows existing CLI progress. Complete/partial/failed outcomes retain
the CLI exit code (usually 0/1/2); transport, readiness or retrieval failure with
no nonzero CLI outcome returns 2. A post-run GPU failure invalidates the demo
outcome while retaining the separate CLI outcome. JSON mode emits one object on
stdout and retains progress/warnings on stderr:

```bash
./scripts/demo-ec2.sh run --image "$PHOTO" --json
./scripts/demo-ec2.sh run --image "$PHOTO" --verification always --json
./scripts/demo-ec2.sh run --image "$PHOTO" --timings --json
```

`--verification always` retains the existing second reading and conditional third
reading. Measure it as a separate group, with additional cost relative to the
first-pass median; do not mix verification modes in one reported latency median.

Results remain on encrypted root EBS in `/home/ubuntu/demo_results/<run-id>/`.
The wrapper retrieves the whole folder into a new private local directory,
verifies its file list, sizes and SHA-256 hashes, and then publishes `Results/`.
Catalogue, review HTML, crops, CLI stdout, run details, provenance and any request
timings stay together. Relative crop links continue to work. Earlier and partial
copies are retained. Local result destinations inside the checkout are rejected.
Open the returned batch's `Review.html` manually; browser opening is outside
inference/retrieval timing and is not included in a five-second claim.

If retrieval failed after inference, use the printed run ID:

```bash
./scripts/demo-ec2.sh retrieve --run-id <32-character-run-id> --json
```

This downloads a finished run again without repeating inference. If SSH was lost
while the CLI was active, inspect `docker ps` and the recorded run directory first.
If no `EC2 Run.json` exists, the controller did not finish: use authenticated SCP
to preserve the interrupted folder as evidence, or stop only its specifically
named `uma-bct-run-<run-id>` container before making a new fresh run. Do not label
an interrupted checkpoint or a failed transfer a completed demonstration.

## Stop/start, readiness, and troubleshooting

User Data normally runs once. The enabled `uma-bct-demo.service` starts Ollama and
warms it after the initial driver reboot and subsequent EC2 starts. Model data in
`/var/lib/uma-bct-demo/models` persists on EBS, never on ephemeral instance-store
NVMe. The service does not run an image-inference loop. `OLLAMA_KEEP_ALIVE=-1`,
one loaded model and one parallel slot retain residency without eliminating fresh
image/text generation. The warm probe uses context 16,384, temperature zero,
thinking disabled and one output token, matching production preflight settings.

After starting EC2 in the Console, update `UMA_BCT_EC2_HOST` if the automatic public
address changed. Also update the SSH source IP/32 if your presenting network
changed. Start sufficiently early to finish cold/reheat preparation, then:

```bash
./scripts/demo-ec2.sh prepare       # no image transfer; reheat and cached app deployment
./scripts/demo-ec2.sh status
```

For failures, inspect cloud-init and service logs. On EC2:

```bash
sudo cloud-init status --long
sudo tail -n 100 /var/log/cloud-init-output.log
sudo journalctl -u uma-bct-demo.service --no-pager
nvidia-smi
docker compose -p uma-bct-demo -f /opt/uma-bct-demo/compose.yaml ps
docker compose -p uma-bct-demo -f /opt/uma-bct-demo/compose.yaml exec -T ollama ollama ps
sudo cat /var/lib/uma-bct-demo/latest-warm.json
```

Driver/module mismatches require fixing packages and rebooting during preparation.
Missing models can be prepared with `sudo systemctl restart uma-bct-demo.service`;
changed model digests require investigating the established model identity, not
changing the pin to get past readiness. `100% CPU` or mixed CPU/GPU residency is
not an accepted GPU demo. Check NVIDIA activity during rehearsal (`nvidia-smi`
or `nvidia-smi dmon`) and retain evidence privately. An idle GPU can show zero
utilization; require its loaded runner/residency, not constant artificial activity.

## Measurements and correctness

Establish the local baseline before comparing GPU instances. Keep reports in an
ignored private-photo directory or outside the checkout:

```bash
uv run python scripts/demo_benchmark.py local --image "$PHOTO" \
  --results-directory "add_repository_photos/demo_benchmarks" --runs 5 --verification never

# Prepared EC2: the benchmark wraps the WHOLE Mac shell command externally.
uv run python scripts/demo_benchmark.py ec2 --image "$PHOTO" \
  --results-directory "$HOME/UMA-BCT-Demo/Benchmarks" --runs 5 --verification never
```

Reports contain every trial, median, slowest and fastest durations, statuses,
physical-box records, catalogue rows, source fingerprint, application identity,
model and settings. Failed trials remain in the report. At least five complete
trials are needed for the provisional timing gate. The final accuracy gate is
always unverified until human review; fastest observed is not fastest correct.

Timing boundaries are explicit:

- **Cold preparation:** bootstrap/install/download timestamps in setup logs and
  `latest-warm.json`'s probe wall time and Ollama load time. Download and model
  loading are preparation, not part of a claimed warm run. Local preparation
  records installed-model load/probe timing, not an already completed download.
- **Warm EC2 CLI:** before `docker run` through process exit after publication,
  including container/Python startup, CLI preflight and all pipeline work.
- **Mac operation:** launcher external wall timer starts before a fresh wrapper
  shell/Python process and ends after complete verified retrieval. It includes
  local input hashing, readiness, SSH, remote Python startup, inference and all
  transfer validation. Its timer/last formatter overhead is separately excluded
  and its 0.01-second resolution recorded. The **benchmark's full Mac command**
  timer includes even those outer launcher/formatter costs; use that value for
  the hard five-second acceptance gate and report the slowest of all five runs.

When diagnosing a missed target, repeat fresh runs with `--timings`. This enables
`UMA_BCT_DEMO_TIMINGS_FILE` inside the run folder. JSONL contains only request kind,
success, wall duration and Ollama's final load/prompt/generation timings and counts,
never text, pixels or token log probabilities. Request validation and confidence
remain unchanged. Sum detection/transcription costs and compare them with whole
execution; crop creation, validation and exports remain included in total time.
Missing timing fields stay missing. This diagnostic is opt-in and Mac/Linux only.
[Ollama timing fields](https://docs.ollama.com/api/chat),
[Ollama residency and keep-alive](https://docs.ollama.com/faq).

First benchmark `g6.2xlarge`; then stop and change the same instance to `g6.xlarge`
if offered in its AZ, start/reheat and repeat five fresh trials. Both have one
24 GB L4; the larger size adds CPU and host RAM, not GPU capacity. Select xlarge
only if it fits without host-memory pressure, completes correctly and meets the
same timing gate at lower cost. A CPU `c7i.2xlarge` is an optional functional
rehearsal using the ordinary CPU Docker workflow; this GPU User Data intentionally
rejects it. CPU rehearsal is not evidence of five-second feasibility.

Review every visible box, including missed/duplicate boxes, identifiers, four-digit
unit numbers, printed labels and handwriting against the original and crops.
Keep expected labels and review notes privately beside the source photographs.
`scripts/verification_benchmark.py` can draft private expectations and compare
human-reviewed references; its drafts are deliberately ineligible as ground truth.
Distinguish physical-box records from inferred `Missing Box` catalogue rows.
If the target is missed, state it, identify the measured bottleneck and report the
fastest **reviewed-correct** configuration if one exists. Do not change context,
resolution, concurrency, model, quantization or processing behaviour on assumption.

**Measurement status:** EC2 and actual presentation runtime remain unmeasured
until the presenter launches and prepares a server. Actual local, EC2, rehearsal
and presentation evidence belongs in private `Feasibility Report.json` and
`Demo Run.json` artifacts, not committed documentation. Record presentation timing
with the full-command benchmark, include all trial durations, and retain the
actual presentation run's report; never replace it with a historical catalogue.

## Budget and rehearsal

Sydney public Linux On-Demand rates checked **9 October 2026**, USD before tax:

| Resource | Rate |
| --- | ---: |
| g6.xlarge, 4 vCPU / 16 GiB RAM / one L4 | $1.04640/hour |
| g6.2xlarge, 8 vCPU / 32 GiB RAM / one L4 | $1.27107/hour |
| c7i.2xlarge optional CPU rehearsal | $0.46620/hour |
| gp3 storage | $0.096/GiB-month |
| Public IPv4 | $0.005/hour |
| Internet ingress | $0 |
| Sydney internet egress, first paid tier | $0.114/GB |

Sources: [official compute feed](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2.json),
[EBS feed](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json),
[transfer feed](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/datatransfer/USD/current/datatransfer.json),
[IPv4 pricing](https://aws.amazon.com/vpc/pricing/).
Recheck before launch. AWS's 100 GB monthly internet-egress allowance is shared
across services/Regions, not exclusive to this demo. Check remaining allowance.

Plan four g6.2xlarge preparation hours, six benchmark hours and four rehearsal
hours, plus four g6.xlarge comparison hours. Including IPv4, 14 days of 100 GiB
gp3 (about $4.48 using a 30-day month), and $1.14 for 10 GB charged egress gives
approximately **$27.69 before presentation**. Reserve **$30** for presentation
and contingencies. Two g6.2xlarge presentation hours bring expected total to
approximately **$30.24**, well below $120. Storage remains charged for its entire
retention period, even while EC2 is stopped.

Credit coverage was user-reported; this workspace has not independently inspected
the AWS account. Before launching, verify Billing → Credits still
shows sufficient unexpired credit through presentation, with EC2/EBS/network
charges eligible; tax treatment and exclusions depend on the credit/account.
Check net charges as well as gross usage; credits and budget alerts are not a hard
spending cap. [Credit rules](https://aws.amazon.com/awscredits/).

Stop EC2 in the Console between preparation, benchmarks and rehearsals. Allow
cold startup and warming before the presentation. Stopping ends compute charges,
but EBS, snapshots and retained Elastic IPs can remain billable. This setup uses
automatic public IPv4, which is normally released on stop. Do not create a NAT
gateway, paid capacity reservation, public result bucket, or retained snapshots
as part of this temporary workflow. [Stop/start behaviour](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html).

After downloading and verifying all needed evidence, terminate the instance,
check deletion of its root EBS volume, and remove any separately retained volumes,
snapshots or Elastic IPs. Preserve the local private reports and reviewed results.

## Model-free validation

```bash
uv sync --locked --dev
uv run ruff check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv build
```

Demo tests use synthetic files and mocked model/SSH interactions. Unix-only demo
tests skip on Windows; native CLI behaviour and model-free CI remain supported.
Real photographs, model inference, benchmarks and expectations stay off hosted CI.
