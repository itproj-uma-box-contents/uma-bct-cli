# GPU User Data variants

Paste exactly one **whole** script into EC2 **Advanced details → User data**,
with base64-encoded input unchecked. These files are self-contained, commented,
and below EC2's 16 KiB raw limit. None launches AWS infrastructure.

| Instance candidate | Required GPU | User Data | Regional On-Demand quota |
| --- | --- | --- | --- |
| p5.4xlarge | One H100, 80 GB | [Existing P5 script](../scripts/demo-ec2-p5-user-data.sh) | P: 16 available vCPUs |
| g7e.2xlarge | One RTX PRO 6000 Blackwell Server Edition, 96 GB | [G7e script](../scripts/demo-ec2-g7e-user-data.sh) | G/VT: 8 available vCPUs |
| g6e.2xlarge | One L40S, 48 GB | [G6e script](../scripts/demo-ec2-g6e-user-data.sh) | G/VT: 8 available vCPUs |

For the alternatives discussed, select Tokyo (ap-northeast-1), then verify the
instance/purchase option in its offered AZs and your account. Quotas are regional;
the Sydney P approval does not cover Tokyo P or either G type. Spot also requires
the separate All P Spot Instance Requests or All G and VT Spot Instance Requests
quota, and spare capacity. Spot can interrupt a run; keep original inputs and
retrieved results on the Mac. A listed offering/price is not live capacity.

The P5 file is Region-independent and works for Tokyo or another Region offering
the exact single-H100 type; select that Region's AMI, key and security group.
The [Sydney guide](EC2-P5-SYDNEY.md) includes the purchase-option correction:
check supported usage classes before attempting another Sydney On-Demand launch.
Avoid one-/three-year reservations and unaffordable Capacity Blocks for this
temporary USD 120 project; keep the USD 30 presentation reserve.

All three use **Canonical Ubuntu Server 24.04 LTS amd64/x86-64**, standard server,
HVM, and **100 GiB encrypted gp3 root EBS**, 3,000 IOPS, 125 MiB/s. Select a public
subnet with automatic public IPv4, SSH/TCP 22 restricted to your current IP/32,
no Ollama/HTTP/HTTPS inbound rules, and default outbound download access. No IAM
role is needed. Instance-store disks are left unused; the model remains on EBS.
Use your chosen Region's matching key pair and private .pem, not a different
Region's pair.

Resolve the current Tokyo Ubuntu AMI in AWS CloudShell (read-only):

```bash
aws ssm get-parameter --region ap-northeast-1 \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text
```

Verify Canonical owner 099720109477 and the name pattern
ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*.
[Canonical AMI discovery](https://ubuntu.com/aws/docs/aws-how-to/instances/find-ubuntu-images/).

Copy the chosen script on your Mac:

```bash
cd /Users/morayyajain/Developer/uma-bct-cli
pbcopy < scripts/demo-ec2-p5-user-data.sh
# OR:
pbcopy < scripts/demo-ec2-g7e-user-data.sh
# OR:
pbcopy < scripts/demo-ec2-g6e-user-data.sh
```

G7e explicitly installs **nvidia-driver-580-server-open** and matching server
utilities. NVIDIA Blackwell requires open kernel modules; the proprietary driver
is not substituted. G6e/P5 retain the existing Ubuntu server-driver selection.
All use NVIDIA Container Toolkit 1.20.1-1 and the pinned Ollama 0.32.9 container.
[NVIDIA requirement](https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/),
[Ubuntu package](https://packages.ubuntu.com/noble-updates/nvidia-driver-580-server-open),
[Ollama GPU support](https://docs.ollama.com/gpu).

No credentials, private keys, repository tokens or photo URLs go into User Data.
The scripts check root capacity/free space, install packages, schedule the driver
reboot, persist an enabled model service, download the exact
qwen3-vl:8b-instruct-q4_K_M model and warm the production context. Initial mode is
one model and one request slot, keep_alive -1, OLLAMA_NO_CLOUD=1, context 16,384,
temperature zero and thinking disabled. Readiness checks the selected GPU, exact
model/digest/context and 100% GPU placement; CPU fallback is refused. Setup is not
photograph inference. Dependencies/model loading happen before presentation timing.

After reboot and setup, SSH as ubuntu and check:

```bash
sudo cloud-init status --wait
sudo journalctl -u uma-bct-demo.service -n 50 --no-pager
/usr/local/bin/uma-bct-demo-health status
```

Cloud-init should report done; the model service should finish successfully;
readiness must report ready, the chosen GPU and 100% GPU. If not, inspect
/var/log/uma-bct-demo-setup.log and service logs before running inference.

After verifying the SSH host key, set the new connection and matching key on
the Mac. Transfer only the explicitly approved photo with prepare:

```bash
export UMA_BCT_EC2_HOST="ubuntu@<current-public-IP-or-DNS>"
export UMA_BCT_EC2_KEY="/absolute/path/to/matching-key.pem"
chmod 400 "$UMA_BCT_EC2_KEY"
export PHOTO="/Users/morayyajain/Downloads/add_repository_photos/PXL_20260814_014117788.jpg"
./scripts/demo-ec2.sh prepare --image "$PHOTO"
./scripts/demo-ec2.sh status
./scripts/demo-ec2.sh run --image "$PHOTO" --verification never --archive-results --timings
```

The model, quantization and existing detection → crops → per-crop transcription
→ catalogue/exports sequence are unchanged. Verification never explicitly means
first pass. Each invocation is fresh, retrieves full results, retains earlier
runs, preserves CLI outcomes and prints actual elapsed time. The exact model
weights, private photographs and outputs remain outside Git, release packages,
hosted CI and container build contexts. Optional verification/parallel experiments
must be separately selected and measured, with human-reviewed output checks.

Before stop/start, retain required local results. Stop EC2 in the Console between
sessions; EBS and retained networking/snapshots can still cost money. On start,
update the public address, allow model reheating, then prepare/status before the
live run. After the trial, terminate and check root EBS deletion/retained resources.

These new variants have model-free generation, syntax, driver and readiness tests.
**H100, Blackwell and L40S photo-processing speeds are unmeasured; five seconds is
not guaranteed.** No hardware trial is automatically launched by these scripts.
