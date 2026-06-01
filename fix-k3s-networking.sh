#!/bin/bash
# Fix K3s networking — run with: sudo bash fix-k3s-networking.sh
set -e

echo "=== Stopping K3s ==="
systemctl stop k3s

echo "=== Writing K3s config ==="
mkdir -p /etc/rancher/k3s
cat > /etc/rancher/k3s/config.yaml << 'EOF'
# Bind API server to all interfaces so pods can reach it via CNI bridge
bind-address: "0.0.0.0"
# Advertise the CNI bridge IP — pods can always reach this
node-ip: "10.42.0.1"
# TLS SANs so kubectl works from localhost and from pods
tls-san:
  - "127.0.0.1"
  - "10.42.0.1"
  - "localhost"
EOF

echo "=== Starting K3s ==="
systemctl start k3s

echo "=== Waiting for K3s API to be ready ==="
for i in $(seq 1 30); do
    if kubectl get nodes &>/dev/null; then
        echo "K3s API is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "=== Waiting for CoreDNS to stabilise ==="
kubectl wait --for=condition=Ready pod -l k8s-app=kube-dns -n kube-system --timeout=120s 2>/dev/null || true

echo ""
echo "=== Re-applying Online Boutique ==="
kubectl apply -f deploy/online-boutique.yaml
kubectl apply -f deploy/prometheus.yaml

echo ""
echo "=== Waiting for all boutique pods ==="
kubectl wait --for=condition=Ready pods --all -n default --timeout=180s 2>/dev/null || true

echo ""
echo "=== Final status ==="
kubectl get pods -A
echo ""
echo "=== Testing connectivity ==="
echo -n "Prometheus targets: "
curl -s 'http://127.0.0.1:30090/api/v1/query?query=up' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"data\"][\"result\"])} active')" 2>/dev/null || echo "not ready yet"
echo -n "Boutique HTTP status: "
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30808/ 2>/dev/null || echo "not ready yet"
echo ""
echo "=== Done! Now run: source venv/bin/activate && python src/dashboard/app.py ==="
