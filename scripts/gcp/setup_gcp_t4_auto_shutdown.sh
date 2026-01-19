#!/bin/bash
# GCP T4 GPU with AUTO-SHUTDOWN (1 hour idle)
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
echo "   - Auto-shutdown: 1 hour idle"
echo "   - Estimated: ~\$0.11/hour (vs \$0.54 regular)"
echo ""

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
    --metadata="install-nvidia-driver=True,startup-script=#!/bin/bash
# Auto-shutdown after 1 hour of idle
apt-get update
apt-get install -y bc
cat > /usr/local/bin/auto-shutdown.sh << 'SCRIPT'
#!/bin/bash
IDLE_THRESHOLD=3600  # 1 hour in seconds
while true; do
    IDLE=\$(cat /proc/uptime | awk '{print int(\$2)}')
    if [ \$IDLE -gt \$IDLE_THRESHOLD ]; then
        echo \"Idle for 1 hour, shutting down...\"
        sudo poweroff
    fi
    sleep 300  # Check every 5 minutes
done
SCRIPT
chmod +x /usr/local/bin/auto-shutdown.sh
nohup /usr/local/bin/auto-shutdown.sh &
" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=jupyter-server

echo ""
echo "✅ Preemptible VM created with auto-shutdown!"
echo ""
echo "⚠️  Notes:"
echo "   - VM will auto-shutdown after 1 hour idle"
echo "   - Preemptible: Google may terminate anytime (24h max)"
echo "   - Perfect for testing/development"
echo ""
echo "📝 Start VM (if stopped):"
echo "   gcloud compute instances start $VM_NAME --zone=$ZONE"
echo ""
echo "🛑 Manual stop:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
