# AWS Provider Configuration
# Version: hashicorp/aws ~> 5.0
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Local variables for resource naming and tagging
locals {
  name_prefix = "tald-unia-${var.environment}"
  common_tags = {
    Project     = "TALD-UNIA"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Component   = "Security"
  }
}

# KMS Key for AES-256-GCM encryption
resource "aws_kms_key" "encryption_key" {
  description              = "TALD UNIA encryption key for secure data storage"
  deletion_window_in_days  = var.kms_config.deletion_window_days
  key_usage               = var.kms_config.key_usage
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  enable_key_rotation     = var.kms_config.key_rotation_enabled

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-encryption-key"
  })
}

# Cognito User Pool for Authentication
resource "aws_cognito_user_pool" "main" {
  name = "${local.name_prefix}-user-pool"

  mfa_configuration = var.cognito_config.mfa_configuration
  
  # Hardware token MFA configuration
  software_token_mfa_configuration {
    enabled = var.cognito_config.hardware_token_enabled
  }

  # Password policy
  password_policy {
    minimum_length    = var.cognito_config.minimum_password_length
    require_lowercase = var.cognito_config.require_lowercase
    require_numbers   = var.cognito_config.require_numbers
    require_symbols   = var.cognito_config.require_symbols
    require_uppercase = var.cognito_config.require_uppercase
  }

  # OAuth configuration
  user_pool_add_ons {
    advanced_security_mode = "ENFORCED"
  }

  # JWT token configuration
  user_pool_add_ons {
    advanced_security_mode = "ENFORCED"
  }

  # Security monitoring
  user_pool_add_ons {
    advanced_security_mode = "ENFORCED"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-user-pool"
  })
}

# Security Group for Network Security
resource "aws_security_group" "main" {
  name        = "${local.name_prefix}-security-group"
  description = "Security group for TALD UNIA gaming platform"
  vpc_id      = data.aws_vpc.main.id

  # WebRTC UDP ports
  dynamic "ingress" {
    for_each = var.security_group_config.webrtc_ports
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "udp"
      cidr_blocks = var.security_group_config.allowed_cidr_blocks
    }
  }

  # HTTPS ingress
  ingress {
    from_port   = var.security_group_config.https_port
    to_port     = var.security_group_config.https_port
    protocol    = "tcp"
    cidr_blocks = var.security_group_config.allowed_cidr_blocks
  }

  # Egress rules
  dynamic "egress" {
    for_each = var.security_group_config.enable_egress ? [1] : []
    content {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = var.security_group_config.egress_cidr_blocks
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-security-group"
  })
}

# CloudWatch Alarms for Security Monitoring
resource "aws_cloudwatch_metric_alarm" "failed_authentication" {
  alarm_name          = "${local.name_prefix}-failed-auth-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.monitoring_config.alert_evaluation_period
  metric_name         = "FailedAuthenticationAttempts"
  namespace           = "TALD/Security"
  period              = 300
  statistic           = "Sum"
  threshold           = var.monitoring_config.failed_auth_threshold
  alarm_description   = "Alert on excessive failed authentication attempts"

  alarm_actions = [aws_sns_topic.security_alerts.arn]
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-failed-auth-alarm"
  })
}

# SNS Topic for Security Alerts
resource "aws_sns_topic" "security_alerts" {
  name = "${local.name_prefix}-security-alerts"
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-security-alerts"
  })
}

# CloudWatch Log Group for Security Audit Logs
resource "aws_cloudwatch_log_group" "security_audit" {
  name              = "/tald-unia/${var.environment}/security-audit"
  retention_in_days = var.monitoring_config.log_retention_days

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-security-audit-logs"
  })
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# Data source for VPC
data "aws_vpc" "main" {
  tags = {
    Name = "${local.name_prefix}-vpc"
  }
}

# Outputs
output "kms_key_arn" {
  value       = aws_kms_key.encryption_key.arn
  description = "ARN of the KMS encryption key"
}

output "cognito_user_pool_id" {
  value       = aws_cognito_user_pool.main.id
  description = "ID of the Cognito user pool"
}

output "security_group_id" {
  value       = aws_security_group.main.id
  description = "ID of the security group"
}