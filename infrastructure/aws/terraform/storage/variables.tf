# Environment variable for deployment environment identification
variable "environment" {
  type        = string
  default     = "dev"
  description = "Environment name for storage resource deployment"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Primary AWS region for storage deployment
variable "region" {
  type        = string
  default     = "us-west-2"
  description = "Primary AWS region for storage deployment"
  
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}$", var.region))
    error_message = "Region must be a valid AWS region identifier."
  }
}

# List of replica regions for multi-region replication
variable "replica_regions" {
  type        = list(string)
  default     = ["us-east-1", "eu-west-1"]
  description = "List of AWS regions for multi-region replication"
  
  validation {
    condition     = length(var.replica_regions) >= 2
    error_message = "At least two replica regions are required for redundancy."
  }
}

# Storage configuration settings
variable "storage_settings" {
  type = map(object({
    retention_period = number
    partition_config = object({
      scan_data_partition_interval = string
      session_partition_interval   = string
    })
    versioning = bool
    replication_rules = object({
      enabled = bool
      priority = number
      destination_storage_class = string
    })
  }))
  
  default = {
    lidar_storage = {
      retention_period = 90
      partition_config = {
        scan_data_partition_interval = "MONTH"
        session_partition_interval   = "WEEK"
      }
      versioning = true
      replication_rules = {
        enabled = true
        priority = 1
        destination_storage_class = "STANDARD_IA"
      }
    }
    game_assets = {
      retention_period = 365
      partition_config = {
        scan_data_partition_interval = "NONE"
        session_partition_interval   = "NONE"
      }
      versioning = true
      replication_rules = {
        enabled = true
        priority = 2
        destination_storage_class = "STANDARD"
      }
    }
  }
  
  description = "Storage settings including retention and partition configurations"
  
  validation {
    condition     = alltrue([for k, v in var.storage_settings : v.retention_period >= 30])
    error_message = "Retention period must be at least 30 days for all storage types."
  }
}

# DynamoDB table configuration settings
variable "dynamodb_settings" {
  type = map(object({
    billing_mode = string
    read_capacity = number
    write_capacity = number
    ttl_enabled = bool
    ttl_attribute = string
    point_in_time_recovery = bool
    stream_enabled = bool
    stream_view_type = string
  }))
  
  default = {
    session_table = {
      billing_mode = "PROVISIONED"
      read_capacity = 50
      write_capacity = 50
      ttl_enabled = true
      ttl_attribute = "expiry_time"
      point_in_time_recovery = true
      stream_enabled = true
      stream_view_type = "NEW_AND_OLD_IMAGES"
    }
    user_data_table = {
      billing_mode = "PROVISIONED"
      read_capacity = 20
      write_capacity = 20
      ttl_enabled = false
      ttl_attribute = ""
      point_in_time_recovery = true
      stream_enabled = true
      stream_view_type = "NEW_AND_OLD_IMAGES"
    }
  }
  
  description = "DynamoDB table configuration settings"
  
  validation {
    condition     = alltrue([for k, v in var.dynamodb_settings : contains(["PROVISIONED", "PAY_PER_REQUEST"], v.billing_mode)])
    error_message = "DynamoDB billing mode must be either PROVISIONED or PAY_PER_REQUEST."
  }
}

# Backup configuration settings
variable "backup_settings" {
  type = object({
    enabled = bool
    schedule = string
    retention_days = number
    cold_storage_after = number
  })
  
  default = {
    enabled = true
    schedule = "cron(0 5 ? * * *)"  # Daily at 5 AM UTC
    retention_days = 30
    cold_storage_after = 90
  }
  
  description = "Backup configuration settings for storage resources"
  
  validation {
    condition     = var.backup_settings.retention_days >= 7
    error_message = "Backup retention period must be at least 7 days."
  }
}