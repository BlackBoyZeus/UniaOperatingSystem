# TALD UNIA Infrastructure - Main Terraform Configuration
# Terraform version: >= 1.0
# AWS Provider version: ~> 4.0

# Data sources for availability zones and account info
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module for network infrastructure
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "tald-unia-${var.environment}"
  cidr = var.vpc_config.cidr_block

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.vpc_config.private_subnets
  public_subnets  = var.vpc_config.public_subnets

  enable_nat_gateway     = var.vpc_config.enable_nat_gateway
  single_nat_gateway     = var.vpc_config.single_nat_gateway
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  # VPC Endpoints for AWS services
  enable_s3_endpoint       = true
  enable_dynamodb_endpoint = true

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
  }
}

# ECS Cluster for containerized services
module "ecs_cluster" {
  source  = "terraform-aws-modules/ecs/aws"
  version = "~> 4.0"

  cluster_name = var.ecs_config.cluster_name

  cluster_configuration = {
    execute_command_configuration = {
      logging = "OVERRIDE"
      log_configuration = {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs_cluster.name
      }
    }
  }

  capacity_providers = var.ecs_config.capacity_providers

  default_capacity_provider_strategy = [{
    capacity_provider = "FARGATE"
    weight           = 1
    base            = 1
  }]

  # Auto-scaling configuration
  autoscaling_capacity_providers = {
    FARGATE = {
      auto_scaling_group_arn         = "FARGATE"
      managed_termination_protection = "ENABLED"
      
      managed_scaling = {
        maximum_scaling_step_size = 10
        minimum_scaling_step_size = 1
        status                   = "ENABLED"
        target_capacity          = 70
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
  }
}

# DynamoDB tables for game state and user data
module "dynamodb_tables" {
  source  = "terraform-aws-modules/dynamodb-table/aws"
  version = "~> 3.0"

  for_each = {
    game_state = {
      hash_key = "session_id"
      range_key = "timestamp"
    }
    user_data = {
      hash_key = "user_id"
      range_key = "data_type"
    }
  }

  name           = "tald-unia-${each.key}-${var.environment}"
  billing_mode   = var.dynamodb_config.billing_mode
  read_capacity  = var.dynamodb_config.read_capacity
  write_capacity = var.dynamodb_config.write_capacity

  hash_key  = each.value.hash_key
  range_key = each.value.range_key

  attributes = [
    {
      name = each.value.hash_key
      type = "S"
    },
    {
      name = each.value.range_key
      type = "S"
    }
  ]

  point_in_time_recovery_enabled = var.dynamodb_config.point_in_time_recovery
  stream_enabled                = var.dynamodb_config.stream_enabled
  stream_view_type             = "NEW_AND_OLD_IMAGES"

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
  }
}

# ElastiCache Redis cluster for session management
module "elasticache_redis" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 3.0"

  cluster_id           = "tald-unia-${var.environment}"
  engine              = "redis"
  node_type           = var.elasticache_config.node_type
  num_cache_nodes     = var.elasticache_config.num_cache_nodes
  parameter_group_family = var.elasticache_config.parameter_group_family
  engine_version      = var.elasticache_config.engine_version
  port                = var.elasticache_config.port

  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.redis.id]

  maintenance_window = "sun:05:00-sun:06:00"
  snapshot_window   = "04:00-05:00"

  multi_az_enabled = true
  automatic_failover_enabled = true

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
  }
}

# CloudFront distribution for content delivery
module "cloudfront" {
  source  = "terraform-aws-modules/cloudfront/aws"
  version = "~> 3.0"

  aliases = ["content.tald-unia.com"]
  comment = "TALD UNIA Content Distribution - ${var.environment}"
  enabled = true
  
  price_class = var.cloudfront_config.price_class
  
  default_cache_behavior = {
    target_origin_id       = "tald-unia-origin"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress              = true
    default_ttl           = var.cloudfront_config.default_ttl
    max_ttl               = var.cloudfront_config.max_ttl
  }

  viewer_certificate = {
    acm_certificate_arn      = var.cloudfront_certificate_arn
    minimum_protocol_version = var.cloudfront_config.minimum_protocol_version
    ssl_support_method       = "sni-only"
  }

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
  }
}

# CloudWatch Log Group for application logs
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/tald-unia/${var.environment}"
  retention_in_days = var.monitoring_config.retention_in_days

  tags = {
    Environment = var.environment
    Project     = "TALD-UNIA"
  }
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "ecs_cluster_arn" {
  description = "ECS Cluster ARN"
  value       = module.ecs_cluster.cluster_arn
}

output "cloudfront_distribution_id" {
  description = "CloudFront Distribution ID"
  value       = module.cloudfront.distribution_id
}