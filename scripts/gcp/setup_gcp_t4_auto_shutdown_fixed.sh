#!/bin/bash
# GCP T4 GPU with AUTO-SHUTDOWN (45 minutes idle)
# Cost-optimized version with Preemptible VM

set -e

PROJECT="leapdesign-esg"
ZONE="us-central1-a"
VM_NAME="colab-t4-auto"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"

echo "=================================="
echo "GCP T4 GPU - Auto-Shutdown Edition"
echo "=================================="
echo ""
echo "💰 Cost Optimization:"
echo "   - Preemptible VM: ~80% cheaper"
echo "   - Auto-shutdown: 45 minutes idle"
echo "   - Estimated: ~\$0.08/session"
echo ""

# Create startup script file
cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
# Auto-shutdown after 45 minutes of idle
apt-get update
apt-get install -y bc

cat > /usr/local/bin/auto-shutdown.sh << 'INNERSCRIPT'
#!/bin/bash
IDLE_THRESHOLD=2700  # 45 minutes in seconds
CHECK_INTERVAL=300    # Check every 5 minutes

echo "[$(date)] Auto-shutdown monitor started (45 min idle threshold)"

while true; do
    IDLE=$(cat /proc/uptime | awk '{print int($2)}')
    if [ $IDLE -gt $IDLE_THRESHOLD ]; then
        echo "[$(date)] Idle for 45 minutes, shutting down..."
        sudo poweroff
    fi
    sleep $CHECK_INTERVAL
done
INNERSCRIPT

chmod +x /usr/local/bin/auto-shutdown.sh
nohup /usr/local/bin/auto-shutdown.sh > /var/log/auto-shutdown.log 2>&1 &

echo "[$(date)] Auto-shutdown monitor configured"
EOF

# Create Preemptible VM with auto-shutdown
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=1" \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata=install-nvidia-driver=True \
    --metadata-from-file=startup-script=/tmp/startup-script.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=jupyter-server

# Clean up temp file
rm /tmp/startup-script.sh

echo ""
echo "✅ Preemptible VM created with auto-shutdown!"
echo ""
echo "⚠️  Notes:"
echo "   - VM will auto-shutdown after 45 minutes idle"
echo "   - Preemptible: Google may terminate anytime (24h max)"
echo "   - Perfect for testing/development"
echo "   - Cost: ~\$0.08 per 45-minute session"
echo ""
echo "📝 Start VM (if stopped):"
echo "   gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "🛑 Manual stop:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "⏳ Waiting for VM to be ready..."
sleep 30

# Get VM external IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "📍 VM External IP: $EXTERNAL_IP"
echo ""
echo "🔥 Setting up firewall rules for Jupyter..."

# Create firewall rule for Jupyter (if not exists)
if ! gcloud compute firewall-rules describe allow-jupyter --project=$PROJECT &>/dev/null; then
    gcloud compute firewall-rules create allow-jupyter \
        --project=$PROJECT \
        --allow=tcp:8888 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=jupyter-server \
        --description="Allow Jupyter notebook access"
    echo "✅ Firewall rule created"
else
    echo "✅ Firewall rule already exists"
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. SSH into VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "2. Inside VM, run:"
echo "   cd ~/trans-gemma && bash scripts/gcp/setup_colab_runtime.sh"
echo ""
echo "3. Or use one-liner (from local machine):"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- 'curl -sSL https://raw.githubusercontent.com/jimmyliao/trans-gemma/main/scripts/gcp/setup_colab_runtime.sh | bash'"
echo ""
echo "4. Connect Colab to:"
echo "   http://$EXTERNAL_IP:8888/?token=YOUR_TOKEN"
echo ""
echo "💰 Estimated cost: ~\$0.08 for 45-minute session"
echo "⏰ Auto-shutdown: 45 minutes of idle time"
echo ""
