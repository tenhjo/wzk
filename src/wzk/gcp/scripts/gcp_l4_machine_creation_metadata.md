# GCP L4 Machine Creation Metadata

## Machine spec

```yaml
project: $GCP_PROJECT
name: johten-gpu-bench
zone: us-central1-c
machine_type: g2-standard-4
gpu:
  type: nvidia-l4
  count: 1
provisioning_model: SPOT
instance_termination_action: DELETE
maintenance_policy: TERMINATE
boot_disk:
  image_family: ubuntu-2204-lts
  image_project: ubuntu-os-cloud
  device_name: johten-gpu-bench
  type: pd-ssd
  size_gb: 200
labels:
  user: $GCP_USER_LABEL
boot_disk_labels:
  user: $GCP_USER_LABEL
metadata:
  install-nvidia-driver: "True"
scopes:
  - https://www.googleapis.com/auth/cloud-platform
```

## Creation command

```bash
gcloud compute instances create johten-gpu-bench \
  --project=${GCP_PROJECT} \
  --zone=us-central1-c \
  --machine-type=g2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --labels=user=${GCP_USER_LABEL} \
  --metadata=install-nvidia-driver=True \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --create-disk=auto-delete=yes,boot=yes,device-name=johten-gpu-bench,image-family=ubuntu-2204-lts,image-project=ubuntu-os-cloud,mode=rw,size=200,type=pd-ssd,labels=user=${GCP_USER_LABEL}
```

## Quick verification

```bash
gcloud compute ssh johten-gpu-bench \
  --project=${GCP_PROJECT} \
  --zone=us-central1-c \
  --command='nvidia-smi'
```
