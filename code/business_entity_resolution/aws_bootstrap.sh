#!/bin/bash
# EC2 user-data for an Ubuntu 24.04 instance: runs once as root at first boot.
# Installs system packages, clones the repo into /home/ubuntu/er and builds the
# Python env. Progress: /var/log/cloud-init-output.log; done when
# /home/ubuntu/BOOTSTRAP_DONE exists. The dataset is uploaded separately.
set -euxo pipefail
apt-get update -y
apt-get install -y python3-venv python3-pip libgomp1 git tmux
sudo -u ubuntu -H bash -c '
  set -e
  cd ~ && git clone https://github.com/priyanshu3232/Amazon-ML-challenge.git er
  cd er && python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r code/business_entity_resolution/requirements.txt
  .venv/bin/python -c "import sparse_dot_topn, lightgbm, pandas; print(\"ENV OK\", pandas.__version__, lightgbm.__version__)"
  touch ~/BOOTSTRAP_DONE
'
