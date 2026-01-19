#!/bin/bash
# Setup Jupyter runtime on GCP VM for Colab connection
# Run this INSIDE the GCP VM

set -e

echo "=================================="
echo "Setting up Colab Runtime on VM"
echo "=================================="

# Install required packages
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install jupyter notebook jupyter_http_over_ws
pip install uv

# Clone trans-gemma repo
echo "📥 Cloning trans-gemma repository..."
cd ~
if [ -d "trans-gemma" ]; then
    cd trans-gemma
    git pull
else
    git clone https://github.com/jimmyliao/trans-gemma.git
    cd trans-gemma
fi

# Install project dependencies
echo "📦 Installing trans-gemma dependencies..."
uv pip install --system -e ".[examples]"

# Enable Jupyter extension
echo "🔧 Configuring Jupyter..."
jupyter serverextension enable --py jupyter_http_over_ws

# Generate Jupyter config
jupyter notebook --generate-config

# Set Jupyter to allow all IPs
cat >> ~/.jupyter/jupyter_notebook_config.py << 'EOF'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
EOF

# Check GPU
echo ""
echo "🔍 Checking GPU..."
nvidia-smi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "🚀 Starting Jupyter server..."
echo ""
echo "📝 Copy the URL with token that appears below"
echo "   Replace 127.0.0.1 with your VM's EXTERNAL IP"
echo ""
echo "💡 In Colab:"
echo "   Connect → Connect to a local runtime"
echo "   Paste: http://YOUR_VM_IP:8888/?token=YOUR_TOKEN"
echo ""
echo "=================================="
echo ""

# Start Jupyter
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --no-browser
