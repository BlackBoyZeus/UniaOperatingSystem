# Terraform version constraint
terraform {
  required_version = "~> 1.0"
}

# Import environment variable from parent module
variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
}

# VPC CIDR block configuration
variable "vpc_cidr" {
  description = "CIDR block for the VPC supporting TALD UNIA's mesh network"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR block must be a valid IPv4 CIDR notation"
  }
}

# Availability zones configuration
variable "availability_zones" {
  description = "List of availability zones for network redundancy"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]

  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "At least 2 availability zones are required for high availability"
  }
}

# Private subnets configuration
variable "private_subnets" {
  description = "CIDR blocks for private subnets hosting game services"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]

  validation {
    condition     = length(var.private_subnets) >= 2
    error_message = "At least 2 private subnets are required for redundancy"
  }
}

# Public subnets configuration
variable "public_subnets" {
  description = "CIDR blocks for public subnets hosting load balancers"
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]

  validation {
    condition     = length(var.public_subnets) >= 2
    error_message = "At least 2 public subnets are required for load balancer redundancy"
  }
}

# NAT Gateway configuration
variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnet internet access"
  type        = bool
  default     = true
}

# VPN Gateway configuration
variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway for secure administrative access"
  type        = bool
  default     = false
}

# CloudFront configuration
variable "cloudfront_price_class" {
  description = "CloudFront distribution price class for global content delivery"
  type        = string
  default     = "PriceClass_200"

  validation {
    condition     = can(regex("^PriceClass_[1-9][0-9]{2}$", var.cloudfront_price_class))
    error_message = "Invalid CloudFront price class"
  }
}

# Route53 configuration
variable "route53_domain" {
  description = "Root domain for TALD UNIA's DNS configuration"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\\.[a-z]{2,}$", var.route53_domain))
    error_message = "Invalid domain name format"
  }
}

# Network tags configuration
variable "tags" {
  description = "Additional tags for network resources"
  type        = map(string)
  default     = {}
}

# WebRTC configuration for P2P mesh network
variable "webrtc_config" {
  description = "WebRTC configuration for P2P mesh network"
  type = object({
    max_fleet_size    = number
    signaling_port    = number
    stun_servers      = list(string)
    turn_servers      = list(string)
    max_latency_ms    = number
  })

  default = {
    max_fleet_size    = 32      # As per technical spec requirement
    signaling_port    = 8443
    stun_servers      = ["stun:stun.l.google.com:19302"]
    turn_servers      = []      # To be configured per deployment
    max_latency_ms    = 50      # As per technical spec requirement
  }

  validation {
    condition     = var.webrtc_config.max_fleet_size <= 32
    error_message = "Maximum fleet size cannot exceed 32 devices as per specification"
  }

  validation {
    condition     = var.webrtc_config.max_latency_ms <= 50
    error_message = "Maximum latency must not exceed 50ms as per specification"
  }
}