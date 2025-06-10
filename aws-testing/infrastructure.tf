# UNIA Gaming OS - AWS Cloud Testing Infrastructure
# Comprehensive hardware and operations testing setup

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region for testing infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "gaming-test"
}

# VPC for isolated testing
resource "aws_vpc" "gaming_test_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "unia-gaming-test-vpc"
    Environment = var.environment
    Purpose     = "Gaming OS Testing"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "gaming_igw" {
  vpc_id = aws_vpc.gaming_test_vpc.id

  tags = {
    Name = "unia-gaming-test-igw"
  }
}

# Public Subnet for test instances
resource "aws_subnet" "gaming_public_subnet" {
  vpc_id                  = aws_vpc.gaming_test_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "unia-gaming-public-subnet"
  }
}

# Route Table
resource "aws_route_table" "gaming_public_rt" {
  vpc_id = aws_vpc.gaming_test_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gaming_igw.id
  }

  tags = {
    Name = "unia-gaming-public-rt"
  }
}

resource "aws_route_table_association" "gaming_public_rta" {
  subnet_id      = aws_subnet.gaming_public_subnet.id
  route_table_id = aws_route_table.gaming_public_rt.id
}

# Security Group for gaming test instances
resource "aws_security_group" "gaming_test_sg" {
  name_prefix = "unia-gaming-test-"
  vpc_id      = aws_vpc.gaming_test_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # VNC for GUI testing
  ingress {
    from_port   = 5900
    to_port     = 5910
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Gaming ports
  ingress {
    from_port   = 7000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # UDP for gaming
  ingress {
    from_port   = 7000
    to_port     = 8000
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "unia-gaming-test-sg"
  }
}

# High-Performance Gaming Test Instance (GPU-enabled)
resource "aws_instance" "gaming_gpu_test" {
  ami           = "ami-0c02fb55956c7d316" # Amazon Linux 2 AMI
  instance_type = "t3.medium"           # Smaller instance for initial testing
  
  subnet_id                   = aws_subnet.gaming_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.gaming_test_sg.id]
  associate_public_ip_address = true
  
  key_name = aws_key_pair.gaming_test_key.key_name

  user_data = base64encode(file("${path.module}/scripts/setup-gpu-instance.sh"))

  tags = {
    Name        = "unia-gaming-gpu-test"
    Environment = var.environment
    Purpose     = "GPU Gaming Testing"
  }
}

# CPU-Intensive Testing Instance
resource "aws_instance" "gaming_cpu_test" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.large" # High-performance CPU for AI testing
  
  subnet_id                   = aws_subnet.gaming_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.gaming_test_sg.id]
  associate_public_ip_address = true
  
  key_name = aws_key_pair.gaming_test_key.key_name

  user_data = base64encode(file("${path.module}/scripts/setup-cpu-instance.sh"))

  tags = {
    Name        = "unia-gaming-cpu-test"
    Environment = var.environment
    Purpose     = "CPU AI Testing"
  }
}

# Memory-Intensive Testing Instance
resource "aws_instance" "gaming_memory_test" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.large" # High-memory for large game worlds
  
  subnet_id                   = aws_subnet.gaming_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.gaming_test_sg.id]
  associate_public_ip_address = true
  
  key_name = aws_key_pair.gaming_test_key.key_name

  user_data = base64encode(file("${path.module}/scripts/setup-memory-instance.sh"))

  tags = {
    Name        = "unia-gaming-memory-test"
    Environment = var.environment
    Purpose     = "Memory Testing"
  }
}

# Network Performance Testing Instance
resource "aws_instance" "gaming_network_test" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.medium" # Enhanced networking
  
  subnet_id                   = aws_subnet.gaming_public_subnet.id
  vpc_security_group_ids      = [aws_security_group.gaming_test_sg.id]
  associate_public_ip_address = true
  
  key_name = aws_key_pair.gaming_test_key.key_name

  user_data = base64encode(file("${path.module}/scripts/setup-network-instance.sh"))

  tags = {
    Name        = "unia-gaming-network-test"
    Environment = var.environment
    Purpose     = "Network Performance Testing"
  }
}

# Key Pair for SSH access
resource "aws_key_pair" "gaming_test_key" {
  key_name   = "unia-gaming-test-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

# S3 Bucket for test results and artifacts
resource "aws_s3_bucket" "gaming_test_results" {
  bucket = "unia-gaming-test-results-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "UNIA Gaming Test Results"
    Environment = var.environment
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "gaming_test_results_versioning" {
  bucket = aws_s3_bucket.gaming_test_results.id
  versioning_configuration {
    status = "Enabled"
  }
}

# CloudWatch Log Group for test logs
resource "aws_cloudwatch_log_group" "gaming_test_logs" {
  name              = "/unia-gaming-os/test-logs"
  retention_in_days = 30

  tags = {
    Environment = var.environment
    Purpose     = "Gaming OS Test Logs"
  }
}

# Outputs
output "gpu_test_instance_ip" {
  description = "Public IP of GPU test instance"
  value       = aws_instance.gaming_gpu_test.public_ip
}

output "cpu_test_instance_ip" {
  description = "Public IP of CPU test instance"
  value       = aws_instance.gaming_cpu_test.public_ip
}

output "memory_test_instance_ip" {
  description = "Public IP of memory test instance"
  value       = aws_instance.gaming_memory_test.public_ip
}

output "network_test_instance_ip" {
  description = "Public IP of network test instance"
  value       = aws_instance.gaming_network_test.public_ip
}

output "test_results_bucket" {
  description = "S3 bucket for test results"
  value       = aws_s3_bucket.gaming_test_results.bucket
}
