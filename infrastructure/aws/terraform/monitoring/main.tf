# AWS Provider configuration
# Version: ~> 4.0
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# Local variables for resource naming and tagging
locals {
  name_prefix = "tald-unia-${var.environment}"
  common_tags = {
    Project             = "TALD-UNIA"
    Environment         = var.environment
    ManagedBy          = "Terraform"
    SecurityCompliance = "GDPR"
    CostCenter         = "Gaming-Platform"
  }
}

# Managed Prometheus Workspace
resource "aws_prometheus_workspace" "prometheus_workspace" {
  alias               = "prometheus_workspace"
  workspace_name      = "${local.name_prefix}-prometheus"
  retention_in_days   = var.prometheus_retention_period
  tags                = local.common_tags

  logging {
    log_group_arn = aws_cloudwatch_log_group.monitoring_logs.arn
  }
}

# Managed Grafana Workspace
resource "aws_grafana_workspace" "grafana_workspace" {
  name                  = "${local.name_prefix}-grafana"
  account_access_type   = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type       = "SERVICE_MANAGED"
  data_sources         = ["PROMETHEUS"]
  role_arn             = aws_iam_role.grafana_role.arn
  
  vpc_configuration {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.grafana_sg.id]
  }

  tags = local.common_tags
}

# CloudWatch Log Group for Monitoring
resource "aws_cloudwatch_log_group" "monitoring_logs" {
  name              = "/aws/vendedlogs/${local.name_prefix}-monitoring"
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

# Performance Monitoring Alarms
resource "aws_cloudwatch_metric_alarm" "lidar_latency" {
  alarm_name          = "${local.name_prefix}-lidar-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.metric_alarms.lidar_latency.evaluation_periods
  metric_name         = "LidarProcessingLatency"
  namespace           = "TALD/LidarMetrics"
  period              = var.metric_alarms.lidar_latency.period
  statistic           = "Average"
  threshold           = var.metric_alarms.lidar_latency.threshold
  alarm_description   = "LiDAR processing latency exceeds 50ms threshold"
  alarm_actions       = [aws_sns_topic.critical_alarms.arn]
  tags                = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "fleet_trust" {
  alarm_name          = "${local.name_prefix}-fleet-trust"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "FleetTrustScore"
  namespace           = "TALD/SecurityMetrics"
  period              = 300
  statistic           = "Average"
  threshold           = var.security_monitoring_config.fleet_trust_threshold
  alarm_description   = "Fleet trust score below threshold"
  alarm_actions       = [aws_sns_topic.security_alarms.arn]
  tags                = local.common_tags
}

# SNS Topics for Alerts
resource "aws_sns_topic" "critical_alarms" {
  name = "${local.name_prefix}-critical-alarms"
  tags = local.common_tags
}

resource "aws_sns_topic" "security_alarms" {
  name = "${local.name_prefix}-security-alarms"
  tags = local.common_tags
}

# IAM Role for Grafana
resource "aws_iam_role" "grafana_role" {
  name = "${local.name_prefix}-grafana-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "grafana.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for Prometheus Access
resource "aws_iam_role_policy" "prometheus_policy" {
  name = "${local.name_prefix}-prometheus-policy"
  role = aws_iam_role.grafana_role.id

  policy = data.aws_iam_policy_document.prometheus_policy.json
}

# Prometheus Policy Document
data "aws_iam_policy_document" "prometheus_policy" {
  statement {
    effect = "Allow"
    actions = [
      "aps:QueryMetrics",
      "aps:GetLabels",
      "aps:GetSeries",
      "aps:GetMetricMetadata"
    ]
    resources = [aws_prometheus_workspace.prometheus_workspace.arn]
    
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.region]
    }
  }
}

# Security Group for Grafana
resource "aws_security_group" "grafana_sg" {
  name        = "${local.name_prefix}-grafana-sg"
  description = "Security group for Grafana workspace"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# Additional Performance Monitoring Alarms
resource "aws_cloudwatch_metric_alarm" "point_cloud_generation" {
  alarm_name          = "${local.name_prefix}-point-cloud-generation"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.metric_alarms.point_cloud_generation.evaluation_periods
  metric_name         = "PointCloudGenerationRate"
  namespace           = "TALD/LidarMetrics"
  period              = var.metric_alarms.point_cloud_generation.period
  statistic           = "Average"
  threshold           = var.metric_alarms.point_cloud_generation.threshold
  alarm_description   = "Point cloud generation rate below 1M points/second"
  alarm_actions       = [aws_sns_topic.critical_alarms.arn]
  tags                = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "game_fps" {
  alarm_name          = "${local.name_prefix}-game-fps"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.metric_alarms.game_fps.evaluation_periods
  metric_name         = "GameFrameRate"
  namespace           = "TALD/GameMetrics"
  period              = var.metric_alarms.game_fps.period
  statistic           = "Average"
  threshold           = var.metric_alarms.game_fps.threshold
  alarm_description   = "Game frame rate below 58 FPS"
  alarm_actions       = [aws_sns_topic.critical_alarms.arn]
  tags                = local.common_tags
}