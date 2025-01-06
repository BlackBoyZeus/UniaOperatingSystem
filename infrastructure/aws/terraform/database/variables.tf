# Terraform variables for AWS database resources
# Version: 1.0

# Environment variable (imported from root module)
variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
}

# Region variable (imported from root module)
variable "region" {
  description = "AWS region for database resources"
  type        = string
}

# DynamoDB table names
variable "dynamodb_table_names" {
  description = "List of DynamoDB table names for the gaming platform"
  type        = list(string)
  default     = [
    "user_data",      # User profiles and preferences
    "game_states",    # Game progress and states
    "fleet_management", # Device fleet coordination
    "scan_data"       # LiDAR scan data storage
  ]
}

# DynamoDB billing mode
variable "dynamodb_billing_mode" {
  description = "DynamoDB billing mode (PROVISIONED or PAY_PER_REQUEST)"
  type        = string
  default     = "PROVISIONED"
}

# DynamoDB replica regions
variable "dynamodb_replica_regions" {
  description = "List of regions for DynamoDB global tables"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-northeast-1"]
}

# DynamoDB auto-scaling configuration
variable "dynamodb_auto_scaling_config" {
  description = "Auto-scaling configuration for DynamoDB tables"
  type = map(object({
    min_capacity = number
    max_capacity = number
    target_utilization = number
  }))
  default = {
    read_capacity = {
      min_capacity = 5
      max_capacity = 100
      target_utilization = 70
    }
    write_capacity = {
      min_capacity = 5
      max_capacity = 100
      target_utilization = 70
    }
  }
}

# DynamoDB backup configuration
variable "dynamodb_backup_config" {
  description = "Backup configuration for DynamoDB tables"
  type = map(string)
  default = {
    point_in_time_recovery = "true"
    backup_retention_days = "35"
  }
}

# Redis cluster configuration
variable "redis_cluster_config" {
  description = "Configuration for Redis cluster deployment"
  type = map(string)
  default = {
    node_type = "cache.t3.medium"
    num_shards = "2"
    replicas_per_shard = "1"
    port = "6379"
    engine_version = "6.x"
    parameter_group_family = "redis6.x"
    automatic_failover = "true"
    multi_az = "true"
    transit_encryption = "true"
    at_rest_encryption = "true"
  }
}

# Redis parameter group settings
variable "redis_parameter_group_settings" {
  description = "Redis parameter group configuration"
  type = map(string)
  default = {
    maxmemory_policy = "volatile-lru"
    maxmemory_samples = "5"
    timeout = "300"
    notify_keyspace_events = "Ex"
  }
}

# Redis maintenance window
variable "redis_maintenance_window" {
  description = "Preferred maintenance window for Redis cluster"
  type = string
  default = "sun:05:00-sun:06:00"
}

# Redis security group rules
variable "redis_security_group_rules" {
  description = "Security group rules for Redis cluster"
  type = list(map(string))
  default = [
    {
      type = "ingress"
      from_port = "6379"
      to_port = "6379"
      protocol = "tcp"
      description = "Redis access from VPC"
    }
  ]
}

# Redis backup retention
variable "redis_backup_retention" {
  description = "Number of days to retain Redis backups"
  type = number
  default = 35
}