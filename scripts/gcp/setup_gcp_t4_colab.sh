#!/bin/bash
# Setup GCP T4 GPU VM for Colab Custom Runtime
# This creates a VM identical to Colab's free T4 environment

set -e

PROJECT="leapdesign-esg"
ZONE="us-central1-a"  # T4 available zone
VM_NAME="colab-t4-runtime"
MACHINE_TYPE="n1-standard-4"  # 4 vCPU, 15GB RAM (similar to Colab)
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=================================="
echo "GCP T4 GPU VM Setup for Colab"
echo "=================================="
echo ""
echo "Project: $PROJECT"
echo "Zone: $ZONE"
echo "VM Name: $VM_NAME"
echo "Machine: $MACHINE_TYPE + T4 GPU"
echo "Image: $IMAGE_FAMILY (includes CUDA, PyTorch, Jupyter)"
echo ""

# Check if VM already exists
if gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT &>/dev/null; then
    echo "⚠️  VM '$VM_NAME' already exists"
    read -p "Do you want to DELETE and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing VM..."
        gcloud compute instances delete $VM_NAME \
            --zone=$ZONE \
            --project=$PROJECT \
            --quiet
    else
        echo "Exiting. Use existing VM or change VM_NAME."
        exit 1
    fi
fi

# Create VM with T4 GPU
echo ""
echo "🚀 Creating T4 GPU VM..."
echo ""

gcloud compute instances create $VM_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --maintenance-policy=TERMINATE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=jupyter-server

echo ""
echo "✅ VM created successfully!"
echo ""

# Wait for VM to be ready
echo "⏳ Waiting for VM to boot..."
sleep 30

# Get VM external IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "📍 VM External IP: $EXTERNAL_IP"
echo ""

# Create firewall rule for Jupyter (if not exists)
echo "🔥 Setting up firewall rules for Jupyter..."
if ! gcloud compute firewall-rules describe allow-jupyter --project=$PROJECT &>/dev/null; then
    gcloud compute firewall-rules create allow-jupyter \
        --project=$PROJECT \
        --allow=tcp:8888 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=jupyter-server \
        --description="Allow Jupyter notebook access"
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
echo "2. Inside VM, run setup script:"
echo "   curl -sSL https://raw.githubusercontent.com/jimmyliao/trans-gemma/main/setup_colab_runtime.sh | bash"
echo ""
echo "3. Get Jupyter URL with token:"
echo "   # Will be displayed after running setup script"
echo ""
echo "4. In Colab:"
echo "   - Click: Connect → Connect to a local runtime"
echo "   - Enter: http://$EXTERNAL_IP:8888/?token=YOUR_TOKEN"
echo ""
echo "💰 Estimated Cost (T4 GPU in us-central1):"
echo "   - T4 GPU: \$0.35/hour"
echo "   - n1-standard-4: \$0.19/hour"
echo "   - Total: ~\$0.54/hour (~\$13/day if running 24/7)"
echo ""
echo "💡 Remember to STOP the VM when not in use:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
