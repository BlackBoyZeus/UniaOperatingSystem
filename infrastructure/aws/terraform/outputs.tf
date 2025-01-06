# Terraform outputs for TALD UNIA platform infrastructure
# Defines all exportable values from AWS infrastructure deployment

# VPC and Networking Outputs
output "vpc_id" {
  description = "ID of the created VPC for TALD UNIA platform"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of private subnet IDs for secure service deployment"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of public subnet IDs for internet-facing components"
  value       = module.vpc.public_subnets
}

# ECS Cluster Outputs
output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster for container orchestration"
  value       = module.ecs_cluster.cluster_arn
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster for service deployment"
  value       = module.ecs_cluster.cluster_name
}

# Database Outputs
output "dynamodb_table_arns" {
  description = "ARNs of the created DynamoDB tables for data persistence"
  value       = module.dynamodb_tables[*].table_arn
}

output "redis_endpoint" {
  description = "Redis cluster endpoint for session management and caching"
  value       = module.elasticache_redis.endpoint
  sensitive   = true
}

# CDN Outputs
output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name for content delivery"
  value       = module.cloudfront.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID for cache invalidation"
  value       = module.cloudfront.distribution_id
}

# Monitoring Outputs
output "monitoring_endpoints" {
  description = "Comprehensive monitoring service endpoints for platform observability"
  value = {
    prometheus = aws_prometheus_workspace.tald.prometheus_endpoint
    grafana    = aws_grafana_workspace.tald.endpoint
    cloudwatch = aws_cloudwatch_log_group.app_logs.arn
  }
  sensitive = false
}

# Security Outputs
output "security_group_ids" {
  description = "Security group IDs for service access control"
  value = {
    redis = aws_security_group.redis.id
    ecs   = module.ecs_cluster.security_group_id
  }
}

# Service Discovery Outputs
output "service_discovery_namespace" {
  description = "Service discovery namespace for internal service communication"
  value       = aws_service_discovery_private_dns_namespace.tald.name
}

# Log Group Outputs
output "log_group_names" {
  description = "CloudWatch Log Group names for application and service logs"
  value = {
    application = aws_cloudwatch_log_group.app_logs.name
    ecs        = aws_cloudwatch_log_group.ecs_cluster.name
  }
}

# VPC Endpoint Outputs
output "vpc_endpoints" {
  description = "VPC endpoint IDs for AWS service access"
  value = {
    s3        = module.vpc.s3_endpoint_id
    dynamodb  = module.vpc.dynamodb_endpoint_id
  }
}

# Tags Output
output "resource_tags" {
  description = "Common tags applied to all resources"
  value = {
    Environment = var.environment
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
  }
}