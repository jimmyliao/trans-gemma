#!/bin/bash
# Simplified GCP T4 GPU setup for Colab
# Using Ubuntu 20.04 + CUDA drivers

set -e

PROJECT="gde-jimmyliao-redeem"  # GDE project with free credits
ZONE="us-central1-a"
VM_NAME="colab-t4-auto"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"

echo "=================================="
echo "GCP T4 GPU for Colab (Simplified)"
echo "=================================="
echo ""
echo "💰 Cost:"
echo "   - Preemptible VM: ~\$0.08/45min"
echo "   - Auto-shutdown: 45 minutes idle"
echo "   - Project: $PROJECT (GDE credits)"
echo ""

# Create startup script
cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
set -e

echo "[$(date)] Starting VM setup..."

# NVIDIA drivers already included in image
# Auto-shutdown script (45 minutes idle)
cat > /usr/local/bin/auto-shutdown.sh << 'INNERSCRIPT'
#!/bin/bash
IDLE_THRESHOLD=2700  # 45 minutes
while true; do
    IDLE=$(cat /proc/uptime | awk '{print int($2)}')
    if [ $IDLE -gt $IDLE_THRESHOLD ]; then
        echo "[$(date)] Idle 45min, shutting down..."
        sudo poweroff
    fi
    sleep 300
done
INNERSCRIPT

chmod +x /usr/local/bin/auto-shutdown.sh
nohup /usr/local/bin/auto-shutdown.sh > /var/log/auto-shutdown.log 2>&1 &

echo "[$(date)] VM setup complete"
EOF

# Create VM
echo "Creating T4 GPU VM..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=1" \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --metadata-from-file=startup-script=/tmp/startup-script.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=jupyter-server

rm /tmp/startup-script.sh

echo ""
echo "✅ VM created! Waiting for startup..."
sleep 30

# Get IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "📍 VM IP: $EXTERNAL_IP"

# Firewall
if ! gcloud compute firewall-rules describe allow-jupyter --project=$PROJECT &>/dev/null; then
    gcloud compute firewall-rules create allow-jupyter \
        --project=$PROJECT \
        --allow=tcp:8888 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=jupyter-server
    echo "✅ Firewall created"
else
    echo "✅ Firewall exists"
fi

echo ""
echo "=================================="
echo "✅ VM Ready!"
echo "=================================="
echo ""
echo "📝 Next: SSH and setup Jupyter"
echo ""
echo "Option 1 - Direct setup:"
echo "gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='curl -sSL https://raw.githubusercontent.com/jimmyliao/trans-gemma/main/scripts/gcp/setup_colab_runtime.sh | bash'"
echo ""
echo "Option 2 - Interactive SSH:"
echo "gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo "# Then run: curl -sSL https://raw.githubusercontent.com/jimmyliao/trans-gemma/main/scripts/gcp/setup_colab_runtime.sh | bash"
echo ""
echo "📍 Colab URL (after setup):"
echo "http://$EXTERNAL_IP:8888/?token=YOUR_TOKEN"
echo ""
echo "💰 Cost: ~\$0.08 for 45 minutes"
echo ""
