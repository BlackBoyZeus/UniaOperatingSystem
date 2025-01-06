# AWS Database Infrastructure for TALD UNIA Gaming Platform
# Version: 1.0

# Configure AWS provider
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
  dynamodb_tags = {
    Service     = "DynamoDB"
    Project     = "TALD-UNIA"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
  
  redis_tags = {
    Service     = "ElastiCache"
    Project     = "TALD-UNIA"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# DynamoDB Tables
resource "aws_dynamodb_table" "gaming_tables" {
  for_each = toset(var.dynamodb_table_names)

  name           = "tald-unia-${each.value}-${var.environment}"
  billing_mode   = var.dynamodb_billing_mode
  hash_key       = "id"
  stream_enabled = true
  stream_view_type = "NEW_AND_OLD_IMAGES"

  # Enable point-in-time recovery
  point_in_time_recovery {
    enabled = true
  }

  # Server-side encryption with AWS managed key
  server_side_encryption {
    enabled = true
  }

  # Table attributes
  attribute {
    name = "id"
    type = "S"
  }

  # Global tables configuration
  dynamic "replica" {
    for_each = var.dynamodb_replica_regions
    content {
      region_name = replica.value
    }
  }

  # Auto scaling configuration for read capacity
  dynamic "read_scaling" {
    for_each = var.dynamodb_billing_mode == "PROVISIONED" ? [1] : []
    content {
      policy_name = "read-scaling-policy"
      target_tracking_scaling_policy_configuration {
        target_value = 70.0
        scale_in_cooldown  = 60
        scale_out_cooldown = 60
      }
    }
  }

  # Auto scaling configuration for write capacity
  dynamic "write_scaling" {
    for_each = var.dynamodb_billing_mode == "PROVISIONED" ? [1] : []
    content {
      policy_name = "write-scaling-policy"
      target_tracking_scaling_policy_configuration {
        target_value = 70.0
        scale_in_cooldown  = 60
        scale_out_cooldown = 60
      }
    }
  }

  # TTL configuration for session data
  ttl {
    attribute_name = "ttl"
    enabled       = true
  }

  tags = merge(local.dynamodb_tags, {
    TableName = each.value
  })

  lifecycle {
    prevent_destroy = true
  }
}

# Redis Cluster for Session Management
resource "aws_elasticache_replication_group" "gaming_session" {
  replication_group_id          = "tald-unia-session-${var.environment}"
  replication_group_description = "TALD UNIA gaming session management cluster"
  
  node_type                     = var.redis_cluster_config["node_type"]
  port                         = tonumber(var.redis_cluster_config["port"])
  parameter_group_family       = var.redis_cluster_config["parameter_group_family"]
  engine_version              = var.redis_cluster_config["engine_version"]
  
  num_cache_clusters          = tonumber(var.redis_cluster_config["num_shards"])
  automatic_failover_enabled  = tobool(var.redis_cluster_config["automatic_failover"])
  multi_az_enabled           = tobool(var.redis_cluster_config["multi_az"])
  
  at_rest_encryption_enabled = tobool(var.redis_cluster_config["at_rest_encryption"])
  transit_encryption_enabled = tobool(var.redis_cluster_config["transit_encryption"])
  
  snapshot_retention_limit   = 7
  snapshot_window           = "04:00-05:00"
  maintenance_window        = var.redis_maintenance_window
  
  auto_minor_version_upgrade = true
  
  # Enhanced monitoring
  notification_topic_arn    = aws_sns_topic.cache_notifications.arn
  
  tags = local.redis_tags
}

# Redis Parameter Group
resource "aws_elasticache_parameter_group" "gaming_session" {
  family = var.redis_cluster_config["parameter_group_family"]
  name   = "tald-unia-params-${var.environment}"
  
  parameter {
    name  = "maxmemory-policy"
    value = "volatile-lru"
  }
  
  parameter {
    name  = "maxmemory-samples"
    value = "5"
  }
  
  parameter {
    name  = "notify-keyspace-events"
    value = "Ex"
  }
  
  tags = local.redis_tags
}

# SNS Topic for Cache Notifications
resource "aws_sns_topic" "cache_notifications" {
  name = "tald-unia-cache-notifications-${var.environment}"
  
  tags = local.redis_tags
}

# CloudWatch Alarms for DynamoDB
resource "aws_cloudwatch_metric_alarm" "dynamodb_throttles" {
  for_each = toset(var.dynamodb_table_names)
  
  alarm_name          = "tald-unia-${each.value}-throttles-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "ThrottledRequests"
  namespace          = "AWS/DynamoDB"
  period             = "300"
  statistic          = "Sum"
  threshold          = "10"
  alarm_description  = "DynamoDB throttled requests monitor"
  
  dimensions = {
    TableName = aws_dynamodb_table.gaming_tables[each.value].name
  }
  
  tags = local.dynamodb_tags
}

# Outputs
output "dynamodb_tables" {
  description = "DynamoDB table information"
  value = {
    table_arns       = { for k, v in aws_dynamodb_table.gaming_tables : k => v.arn }
    table_names      = { for k, v in aws_dynamodb_table.gaming_tables : k => v.name }
    table_stream_arns = { for k, v in aws_dynamodb_table.gaming_tables : k => v.stream_arn }
  }
}

output "elasticache_cluster" {
  description = "Redis cluster information"
  value = {
    cluster_endpoint = aws_elasticache_replication_group.gaming_session.primary_endpoint_address
    cluster_id       = aws_elasticache_replication_group.gaming_session.id
    cluster_status   = aws_elasticache_replication_group.gaming_session.status
  }
}