#!/bin/bash
# Deploy UNIA Gaming OS Test Infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎮 UNIA Gaming OS - AWS Test Infrastructure Deployment${NC}"
echo -e "${BLUE}====================================================${NC}"

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed${NC}"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform is not installed${NC}"
    echo "Please install Terraform: https://terraform.io/downloads"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo "Please configure AWS credentials: aws configure"
    exit 1
fi

# Check SSH key
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo -e "${YELLOW}⚠️ SSH key not found, generating new key...${NC}"
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Get AWS account info
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")

echo -e "${BLUE}📋 Deployment Configuration:${NC}"
echo "AWS Account: $AWS_ACCOUNT"
echo "AWS Region: $AWS_REGION"
echo "SSH Key: ~/.ssh/id_rsa.pub"

# Confirm deployment
read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Initialize Terraform
echo -e "${YELLOW}🚀 Initializing Terraform...${NC}"
terraform init

# Plan deployment
echo -e "${YELLOW}📋 Planning deployment...${NC}"
terraform plan -var="aws_region=$AWS_REGION" -out=tfplan

# Apply deployment
echo -e "${YELLOW}🏗️ Deploying infrastructure...${NC}"
terraform apply tfplan

# Get outputs
echo -e "${GREEN}✅ Infrastructure deployed successfully!${NC}"
echo -e "${BLUE}📊 Test Instance Information:${NC}"

GPU_IP=$(terraform output -raw gpu_test_instance_ip)
CPU_IP=$(terraform output -raw cpu_test_instance_ip)
MEMORY_IP=$(terraform output -raw memory_test_instance_ip)
NETWORK_IP=$(terraform output -raw network_test_instance_ip)
S3_BUCKET=$(terraform output -raw test_results_bucket)

echo "🎯 GPU Test Instance (g4dn.xlarge): $GPU_IP"
echo "🧠 CPU Test Instance (c5n.4xlarge): $CPU_IP"
echo "💾 Memory Test Instance (r5.2xlarge): $MEMORY_IP"
echo "🌐 Network Test Instance (c5n.large): $NETWORK_IP"
echo "☁️ S3 Results Bucket: $S3_BUCKET"

# Create connection scripts
echo -e "${YELLOW}📝 Creating connection scripts...${NC}"

cat > connect-gpu-instance.sh << EOF
#!/bin/bash
echo "Connecting to GPU Test Instance..."
ssh -i ~/.ssh/id_rsa ec2-user@$GPU_IP
EOF

cat > connect-cpu-instance.sh << EOF
#!/bin/bash
echo "Connecting to CPU Test Instance..."
ssh -i ~/.ssh/id_rsa ec2-user@$CPU_IP
EOF

cat > connect-memory-instance.sh << EOF
#!/bin/bash
echo "Connecting to Memory Test Instance..."
ssh -i ~/.ssh/id_rsa ec2-user@$MEMORY_IP
EOF

cat > connect-network-instance.sh << EOF
#!/bin/bash
echo "Connecting to Network Test Instance..."
ssh -i ~/.ssh/id_rsa ec2-user@$NETWORK_IP
EOF

chmod +x connect-*.sh

# Wait for instances to be ready
echo -e "${YELLOW}⏳ Waiting for instances to initialize (this may take 5-10 minutes)...${NC}"
sleep 300

# Test connectivity
echo -e "${YELLOW}🔗 Testing connectivity...${NC}"
for ip in $GPU_IP $CPU_IP $MEMORY_IP $NETWORK_IP; do
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ec2-user@$ip "echo 'Connected to $ip'" 2>/dev/null; then
        echo -e "${GREEN}✅ $ip - Connected${NC}"
    else
        echo -e "${RED}❌ $ip - Connection failed${NC}"
    fi
done

# Create test execution script
cat > run-tests.sh << EOF
#!/bin/bash
echo "🎮 Running UNIA Gaming OS Tests..."
cd test-automation
./run-all-tests.sh
EOF

chmod +x run-tests.sh

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Wait for instances to fully initialize (5-10 minutes)"
echo "2. Run comprehensive tests: ./run-tests.sh"
echo "3. Connect to individual instances:"
echo "   - GPU Instance: ./connect-gpu-instance.sh"
echo "   - CPU Instance: ./connect-cpu-instance.sh"
echo "   - Memory Instance: ./connect-memory-instance.sh"
echo "   - Network Instance: ./connect-network-instance.sh"
echo ""
echo -e "${YELLOW}💡 Test Results will be saved to S3: s3://$S3_BUCKET${NC}"
echo ""
echo -e "${GREEN}🎮 Ready to test UNIA Gaming OS in the cloud!${NC}"
