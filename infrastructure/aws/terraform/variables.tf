# Terraform version constraint
terraform {
  required_version = ">= 1.0"
}

# Environment variable
variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  
  validation {
    condition     = can(regex("^(development|staging|production)$", var.environment))
    error_message = "Environment must be development, staging, or production"
  }
}

# VPC Configuration
variable "vpc_config" {
  description = "VPC configuration for network infrastructure"
  type = object({
    cidr_block         = string
    private_subnets    = list(string)
    public_subnets     = list(string)
    enable_nat_gateway = bool
    single_nat_gateway = bool
  })
  
  default = {
    cidr_block         = "10.0.0.0/16"
    private_subnets    = ["10.0.1.0/24", "10.0.2.0/24"]
    public_subnets     = ["10.0.101.0/24", "10.0.102.0/24"]
    enable_nat_gateway = true
    single_nat_gateway = false
  }
}

# ECS Configuration
variable "ecs_config" {
  description = "ECS cluster configuration for game services"
  type = object({
    cluster_name                = string
    capacity_providers          = list(string)
    container_insights          = bool
    service_discovery_namespace = string
  })
  
  default = {
    cluster_name                = "tald-unia"
    capacity_providers          = ["FARGATE", "FARGATE_SPOT"]
    container_insights          = true
    service_discovery_namespace = "tald.local"
  }
}

# DynamoDB Configuration
variable "dynamodb_config" {
  description = "DynamoDB configuration for game state and user data"
  type = object({
    billing_mode            = string
    read_capacity          = number
    write_capacity         = number
    point_in_time_recovery = bool
    stream_enabled         = bool
  })
  
  default = {
    billing_mode            = "PROVISIONED"
    read_capacity          = 50
    write_capacity         = 50
    point_in_time_recovery = true
    stream_enabled         = true
  }
}

# ElastiCache Configuration
variable "elasticache_config" {
  description = "ElastiCache Redis configuration for session management"
  type = object({
    node_type              = string
    num_cache_nodes        = number
    parameter_group_family = string
    engine_version        = string
    port                  = number
  })
  
  default = {
    node_type              = "cache.t3.medium"
    num_cache_nodes        = 2
    parameter_group_family = "redis6.x"
    engine_version        = "6.x"
    port                  = 6379
  }
}

# CloudFront Configuration
variable "cloudfront_config" {
  description = "CloudFront CDN configuration for content delivery"
  type = object({
    price_class               = string
    minimum_protocol_version  = string
    default_ttl              = number
    max_ttl                  = number
  })
  
  default = {
    price_class               = "PriceClass_100"
    minimum_protocol_version  = "TLSv1.2_2021"
    default_ttl              = 86400    # 24 hours
    max_ttl                  = 31536000 # 1 year
  }
}

# Monitoring Configuration
variable "monitoring_config" {
  description = "Monitoring and observability configuration"
  type = object({
    retention_in_days               = number
    prometheus_evaluation_interval  = string
    grafana_admin_password         = string
    alert_email_endpoints          = list(string)
  })
  
  default = {
    retention_in_days               = 30
    prometheus_evaluation_interval  = "1m"
    grafana_admin_password         = null
    alert_email_endpoints          = []
  }

  validation {
    condition     = var.monitoring_config.retention_in_days >= 1 && var.monitoring_config.retention_in_days <= 365
    error_message = "Retention period must be between 1 and 365 days"
  }
}