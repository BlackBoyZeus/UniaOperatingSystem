# AWS Provider configuration with version constraint
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Local variables for resource tagging
locals {
  tags = {
    Project       = "TALD-UNIA"
    Environment   = var.environment
    ManagedBy     = "Terraform"
    SecurityLevel = "High"
    NetworkType   = "Gaming"
  }
}

# VPC Module for gaming infrastructure
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "tald-unia-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  # WebRTC dedicated subnets for P2P gaming traffic
  intra_subnets = [for i, az in var.availability_zones : 
    cidrsubnet(var.vpc_cidr, 8, length(var.private_subnets) + length(var.public_subnets) + i)
  ]

  # Enable DNS support for service discovery
  enable_dns_hostnames = true
  enable_dns_support   = true

  # NAT Gateway configuration for private subnet internet access
  enable_nat_gateway     = var.enable_nat_gateway
  single_nat_gateway     = false
  one_nat_gateway_per_az = true

  # VPN Gateway for secure administrative access
  enable_vpn_gateway = var.enable_vpn_gateway

  # Enable VPC flow logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  tags = merge(local.tags, var.tags)
}

# Security group for WebRTC traffic
resource "aws_security_group" "webrtc" {
  name_prefix = "webrtc-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = var.webrtc_config.signaling_port
    to_port     = var.webrtc_config.signaling_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "WebRTC signaling"
  }

  ingress {
    from_port   = 49152
    to_port     = 65535
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "WebRTC P2P communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "webrtc-sg-${var.environment}"
  })
}

# CloudFront distribution for global content delivery
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  price_class         = var.cloudfront_price_class
  comment             = "TALD UNIA content distribution - ${var.environment}"
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.content.bucket_regional_domain_name
    origin_id   = "content-origin"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "content-origin"
    viewer_protocol_policy = "redirect-to-https"
    compress              = true

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # WebSocket support for real-time communication
  ordered_cache_behavior {
    path_pattern           = "/ws/*"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "content-origin"
    viewer_protocol_policy = "https-only"
    compress              = false

    forwarded_values {
      query_string = true
      headers      = ["Sec-WebSocket-Key", "Sec-WebSocket-Version"]
      cookies {
        forward = "all"
      }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
    minimum_protocol_version       = "TLSv1.3"
  }

  tags = local.tags
}

# CloudFront origin access identity
resource "aws_cloudfront_origin_access_identity" "main" {
  comment = "TALD UNIA content access identity"
}

# Route53 configuration for DNS management
resource "aws_route53_zone" "main" {
  name = var.route53_domain

  tags = local.tags
}

# VPC endpoints for AWS services
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${data.aws_region.current.name}.s3"

  tags = merge(local.tags, {
    Name = "s3-endpoint-${var.environment}"
  })
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${data.aws_region.current.name}.dynamodb"

  tags = merge(local.tags, {
    Name = "dynamodb-endpoint-${var.environment}"
  })
}

# Network ACL for enhanced security
resource "aws_network_acl" "gaming" {
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  ingress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = var.vpc_cidr
    from_port  = var.webrtc_config.signaling_port
    to_port    = var.webrtc_config.signaling_port
  }

  ingress {
    protocol   = "udp"
    rule_no    = 200
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 49152
    to_port    = 65535
  }

  egress {
    protocol   = "-1"
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 0
    to_port    = 0
  }

  tags = merge(local.tags, {
    Name = "gaming-nacl-${var.environment}"
  })
}

# Data source for current AWS region
data "aws_region" "current" {}

# Outputs for other modules
output "vpc_id" {
  value       = module.vpc.vpc_id
  description = "VPC ID for service deployment"
}

output "private_subnet_ids" {
  value       = module.vpc.private_subnets
  description = "Private subnet IDs for secure service deployment"
}

output "webrtc_subnet_ids" {
  value       = module.vpc.intra_subnets
  description = "WebRTC subnet IDs for P2P gaming traffic"
}

output "cloudfront_distribution_id" {
  value       = aws_cloudfront_distribution.main.id
  description = "CloudFront distribution ID for DNS configuration"
}