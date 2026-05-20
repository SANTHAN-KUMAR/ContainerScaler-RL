#!/bin/bash
# k3s-setup.sh
# Installs k3s and deploys the baseline infrastructure for ContainerScale-RL.

set -e

echo "1. Installing k3s..."
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

echo "Waiting for k3s to be ready..."
sleep 15
kubectl wait --for=condition=Ready nodes --all --timeout=60s

echo "2. Installing Prometheus Stack (simplified for metrics server)..."
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
# We will apply a custom prometheus instance in deploy/prometheus.yaml

echo "3. Deploying target application (podinfo)..."
kubectl apply -f deploy/podinfo.yaml

echo "4. Deploying Prometheus instance..."
kubectl apply -f deploy/prometheus.yaml

echo "Setup complete. To run the locust load test:"
echo "locust -f deploy/locustfile.py --headless -u 100 -r 10 --run-time 1h --host http://<podinfo-svc-ip>:9898"
