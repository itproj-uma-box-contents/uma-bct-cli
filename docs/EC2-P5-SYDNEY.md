# Sydney P5.4xlarge H100 trial

This is a bounded hardware experiment with the unchanged UMA BCT model and
pipeline. **H100 runtime has not been measured; five seconds is not guaranteed.**
Launch infrastructure yourself in the AWS Console. Stop unused G6/M7i instances
there to preserve the USD 120 credit budget, and reserve USD 30 for presentation
and contingencies. Keep all photographs, results and model state private.

## Quota and availability first

**Purchase-option correction (13 October 2026):** An On-Demand pricing-catalogue
entry does not prove that the instance can be launched through that option in
Sydney. AWS's published p5.4xlarge launch announcement lists Sydney for Capacity
Blocks and lists Tokyo, Mumbai, Jakarta, London and Sao Paulo for On-Demand/Spot.
Confirm current regional purchase support before following the On-Demand
checklist below; live capacity is a separate check. The earlier USD 8.944/hour
quote is a catalogue rate, not a capacity guarantee.
[AWS purchase-option announcement](https://aws.amazon.com/about-aws/whats-new/2025/08/amazon-p5-single-gpu-instances-now-available/).

Read-only CloudShell check:

```bash
aws ec2 describe-instance-types --region ap-southeast-2 \
  --instance-types p5.4xlarge \
  --query 'InstanceTypes[0].SupportedUsageClasses' --output json
```

Spot requires the separate regional **All P Spot Instance Requests** quota and
spare capacity; it can be interrupted and is unsuitable as the sole live-demo
plan. Regional Reserved Instances do not reserve hardware; zonal reservations
have capacity benefits but one-/three-year commitments are unsuitable for this
temporary budget. An immediate Capacity Reservation can also fail for insufficient
capacity and is billed while held, even unused.

Capacity Blocks are scheduled, non-cancellable purchases with a documented
minimum one-day duration. AWS's current indicative p5.4xlarge block rate is
USD 5.970/hour: a 24-hour block is about **USD 143.28 before other charges**, above
the USD 120 total budget. Actual offering prices depend on supply/demand. Do not
purchase a block within this budget; upfront credit eligibility is unverified.
[Block duration](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-how.html),
[Block pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/),
[Spot quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html).

Select **Asia Pacific (Sydney), ap-southeast-2**. Open **Service Quotas → Amazon
EC2 → Running On-Demand P instances**. You need **16 available vCPUs**, after
subtracting other P usage/reservations. Request an applied quota of at least 16
if needed. G/VT quota approvals and approvals in other Regions do not cover this.
The default P quota can be zero. [AWS quota documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html).

AWS's regional listing and current Sydney price list include P5/p5.4xlarge.
Live AZ capacity and your account quota are unverified: confirm that p5.4xlarge
is offered in the selected subnet/AZ in the Console. Try another offered Sydney
public subnet if capacity is unavailable. Keep the exact instance selection;
do not substitute p5.48xlarge or create a paid capacity reservation/block.
[Regional offerings](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html).

## Exact launch settings

| Console field | Selection |
| --- | --- |
| Region | Asia Pacific (Sydney), ap-southeast-2 |
| Name | uma-bct-demo-h100 |
| Number of instances | 1 |
| Purchase option | On-Demand; shared/default tenancy |
| Instance type | **p5.4xlarge** |
| GPU | 1 NVIDIA H100, 80 GB HBM3 |
| CPU | 16 vCPU; AMD EPYC 7R13, listed clock 2.95 GHz; keep CPU options default |
| RAM | 256 GiB |
| Architecture | x86-64 / amd64 |
| Operating system | Ubuntu Server 24.04 LTS, standard server, without Ubuntu Pro |
| Virtualization | HVM |
| AMI publisher | Canonical, owner **099720109477** |
| AMI name pattern | ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-* |
| Key pair | Existing Sydney key only if its matching private key is available; otherwise create uma-bct-demo-h100, ED25519, .pem |
| VPC | Default Sydney VPC or another VPC with a public subnet and Internet Gateway route |
| Subnet/AZ | An offered Sydney public subnet; leave AZ preference open where the wizard permits |
| Auto-assign public IPv4 | Enable |
| Security group | Dedicated uma-bct-demo-h100-ssh |
| Inbound rule | SSH, TCP 22, **My IP /32** only; refresh when your network changes |
| Ollama/HTTP/HTTPS inbound | No rules for 11434, 80 or 443 |
| Outbound | Default allow-all; needed for Ubuntu, Docker, NVIDIA and model downloads |
| Root EBS | **100 GiB gp3**, encrypted, AWS-managed aws/ebs key |
| gp3 performance | 3,000 IOPS; 125 MiB/s; no extra paid performance |
| Root delete on termination | Enable |
| Instance-store SSD | 1 × 3,840 GB local SSD; leave unused for this demo |
| Network hardware | Listed 100 Gbps EFA; ordinary ENA networking is sufficient; no EFA setup needed |
| EBS bandwidth | Listed 10 Gbps; gp3's selected volume performance still applies |
| IAM instance profile | None required |
| Detailed monitoring | Disabled; default basic monitoring |
| Placement group | None |
| Capacity reservation preference | None; do not create reservations |
| Spot / Capacity Block | Off; this checklist uses On-Demand |
| Shutdown behaviour | Stop |
| Hibernate | Disabled/default |
| Metadata endpoint | Enabled; IMDSv2 required; hop limit 1 is sufficient for host checks |
| Other advanced settings | Leave AMI/default settings unchanged |
| User Data | Paste the whole scripts/demo-ec2-p5-user-data.sh, with base64-encoded input unchecked |

[AWS P5 specifications](https://aws.amazon.com/ec2/instance-types/p5/).
AMI IDs change and differ by Region. Canonical's public locator did not return
a Noble Sydney row during this check; resolve the exact current AMI through
Canonical's official public SSM parameter in **AWS CloudShell**, in Sydney:

```bash
aws ssm get-parameter --region ap-southeast-2 \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text
```

This read-only command prints the AMI ID; it does not launch anything. Select
that ID in the launch wizard and verify owner 099720109477, Ubuntu 24.04,
amd64/x86-64, HVM and EBS. Alternatively, use Quick Start's official Canonical
Ubuntu Server 24.04 LTS x86 image and verify the same fields.
[Canonical AMI discovery](https://ubuntu.com/aws/docs/aws-how-to/instances/find-ubuntu-images/).

## User Data

The complete self-contained script is
[demo-ec2-p5-user-data.sh](../scripts/demo-ec2-p5-user-data.sh).
From the Mac checkout:

```bash
cd /Users/morayyajain/Developer/uma-bct-cli
pbcopy < scripts/demo-ec2-p5-user-data.sh
```

Paste into **Advanced details → User data**. Do not paste the G6 script instead:
the P5 variant explicitly expects H100. No repository token, AWS credentials,
private key or photo download URL belongs in User Data.

It installs official Docker/Compose, Ubuntu's NVIDIA 580 server driver/utilities
and NVIDIA Container Toolkit 1.20.1-1. Public apt keyrings are readable by _apt,
while private data directories use restrictive permissions. It checks a >=100 GiB
root block device and >=30 GiB free, schedules one driver reboot, and enables a
persistent model-preparation service for later boots/stop-starts. Filesystem
commands do not enlarge EBS; provision the 100 GiB volume in the Console.

Ollama remains the repository's digest-pinned 0.32.9 image. It downloads exactly
qwen3-vl:8b-instruct-q4_K_M and checks the established digest. Model storage is
on root EBS under /var/lib/uma-bct-demo/models; ephemeral instance storage is not
used. One loaded model and one request slot are the starting configuration, with
keep_alive -1, OLLAMA_NO_CLOUD=1, context 16,384, temperature zero and thinking
disabled. The text preparation probe loads the production context; it is not
photo inference or a promise to cache every photograph.

The script establishes /home/ubuntu/demo_images, demo_results, demo_app and
demo_incoming. Application/source deployment and the one approved photo transfer
happen later over SSH via prepare. Nothing private enters Git, hosted CI,
release archives or container build contexts. The Ollama API has no published
ports; GPU readiness requires NVIDIA tooling plus the exact model at 100% GPU.
CPU spill, wrong GPU, multiple GPUs, wrong context or changed model are failures.

Setup references: [Docker](https://docs.docker.com/engine/install/ubuntu/),
[Ubuntu GPU drivers](https://ubuntu.com/aws/docs/aws-how-to/instances/install-nvidia-drivers/),
[NVIDIA toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[Ollama Docker](https://docs.ollama.com/docker).

## After launch

Wait for EC2 status checks and the driver reboot; allow installation/download and
model loading to finish. Running/status checks alone do not prove GPU readiness.
Use the instance's **current** public IPv4/DNS and matching Sydney key. The earlier
Malaysia key pair is a different regional resource. If reusing the existing
Sydney uma-bct-demo pair, its previously working local key was
/Users/morayyajain/.ssh/uma-bct-demo.pem; confirm that pair is still selected.

```bash
export UMA_BCT_EC2_HOST="ubuntu@<H100-current-public-IPv4-or-DNS>"
export UMA_BCT_EC2_KEY="/absolute/path/to/matching-sydney-key.pem"
chmod 400 "$UMA_BCT_EC2_KEY"
ssh -i "$UMA_BCT_EC2_KEY" "$UMA_BCT_EC2_HOST"
```

Verify the first SSH host-key fingerprint using the instance's EC2 system-console
output. The wrapper keeps strict host checking; do not disable it to bypass a
changed key. In the Ubuntu SSH terminal:

```bash
sudo cloud-init status --wait
sudo journalctl -u uma-bct-demo.service -n 50 --no-pager
/usr/local/bin/uma-bct-demo-health status
```

Expected: cloud-init done; model service Result=success, active/exited after
preparation; readiness status ready, one H100, exact digest, context 16384 and
100% GPU. The driver reboot disconnects SSH; reconnect after it finishes.

For failures, keep the logs and diagnose before timing any photograph:

```bash
sudo tail -n 100 /var/log/cloud-init-output.log
sudo tail -n 100 /var/log/uma-bct-demo-setup.log
sudo journalctl -u uma-bct-demo.service -n 100 --no-pager
nvidia-smi
```

## Prepare and run from the Mac

```bash
cd /Users/morayyajain/Developer/uma-bct-cli
export PHOTO="/Users/morayyajain/Downloads/add_repository_photos/PXL_20260814_014117788.jpg"
./scripts/demo-ec2.sh prepare --image "$PHOTO" --transcription-workers 1
./scripts/demo-ec2.sh status

# One fresh timed photograph, with complete verified local retrieval:
./scripts/demo-ec2.sh run --image "$PHOTO" \
  --verification never --archive-results --timings
```

First-pass processing is explicitly **--verification never**. Detection, all
crops, all per-crop transcription, confidence, catalogue mapping and atomic
exports are preserved. Each run is fresh (--reprocess and a new results folder),
with old runs retained. The command shows progress, exit/status, measured elapsed
time and private local location. Use --json for clean machine-readable stdout.

Record five fresh warm runs only after setup/readiness passes:

```bash
uv run python scripts/demo_benchmark.py ec2 --image "$PHOTO" \
  --results-directory "$HOME/UMA-BCT-Demo/Benchmarks" --runs 5 \
  --verification never --archive-results --timings
```

This compares the unchanged sequential pipeline first. The optional two-runner
experiment remains separately selectable with --transcription-workers 2 on
prepare/status/run/benchmark; increasing OLLAMA_NUM_PARALLEL alone does not make
Qwen3-VL parallel in the pinned server. H100 speed, parallel benefit and output
quality require measurement. Do not claim a first token or partial/failed run as
five-second completion. Human review must cover every visible box, identifier,
unit number, printed label and handwritten annotation; historical counts are not
ground truth. Verification always retains its existing extra pass and needs a
separate timing/cost comparison.

## Cost limit and cleanup

Sydney Linux shared On-Demand p5.4xlarge is **USD 8.944/hour**, checked
10 October 2026 against the current AWS regional feed (publication
2026-10-09T20:31:34Z). Two running hours, including all cold setup, are
**USD 17.888 compute**. Public IPv4 adds USD 0.005/hour. 100 GiB gp3 costs
USD 9.60 for a full month at USD 0.096/GiB-month; retained for 14 days it is
about USD 4.48 using a 30-day month. Internet ingress is free; first paid Sydney
internet-egress tier is USD 0.114/GB after the shared monthly 100 GB allowance.
Two running hours, 14 days storage and 10 GB charged egress would total about
**USD 23.52 before tax**, within a USD 25 trial envelope. Include existing
G6/M7i/volume charges, verify eligible remaining credits in Billing, and retain
the USD 30 presentation reserve. Account credit coverage was user-reported,
not independently verified. A budget alert is not an automatic spend cap.

[Sydney price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/ap-southeast-2/index.json),
[IPv4 pricing](https://aws.amazon.com/vpc/pricing/),
[transfer pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

Stop the instance in the Console by the two-running-hour trial limit. Stopping
ends compute charges but EBS, snapshots and retained Elastic IPs remain billable.
This setup uses an automatic public IPv4, released on stop; update connection
details after start, allow the persistent service to reheat and run prepare/status
before another timed command. Reboot preserves EBS; instance-store contents are
not a persistence mechanism. After retrieving and verifying required evidence,
terminate the instance and check root EBS deletion and any retained resources.

Implementation has model-free tests, shell syntax and User Data size checks.
No H100 hardware launch or real-photo timing has been performed yet.
