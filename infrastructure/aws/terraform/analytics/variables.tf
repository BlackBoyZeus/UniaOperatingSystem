# Core environment variable with validation
variable "environment" {
  description = "Deployment environment identifier (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# AWS region variable with validation
variable "region" {
  description = "AWS region for analytics infrastructure deployment"
  type        = string
  default     = "us-west-2"

  validation {
    condition     = can(regex("^(us|eu|ap|sa|ca|me|af)-(north|south|east|west|central)-[1-3]$", var.region))
    error_message = "Must be a valid AWS region identifier."
  }
}

# Kinesis stream configuration with comprehensive settings
variable "kinesis_stream_config" {
  description = "Configuration map for Kinesis data streams with time-based partitioning"
  type = map(object({
    name             = string
    shard_count      = number
    retention_period = number
    encryption_type  = string
    tags            = map(string)
  }))

  validation {
    condition = alltrue([
      for k, v in var.kinesis_stream_config : (
        length(v.name) >= 1 &&
        length(v.name) <= 128 &&
        v.shard_count >= 1 &&
        v.shard_count <= 200 &&
        v.retention_period >= 24 &&
        v.retention_period <= 8760 &&
        contains(["NONE", "KMS"], v.encryption_type)
      )
    ])
    error_message = "Invalid Kinesis stream configuration. Check name length, shard count (1-200), retention period (24-8760 hours), and encryption type (NONE/KMS)."
  }

  default = {
    game_events = {
      name             = "game-events"
      shard_count      = 10
      retention_period = 48
      encryption_type  = "KMS"
      tags = {
        Environment = "dev"
        Purpose     = "game-analytics"
      }
    }
  }
}

# Analytics application configuration with processing settings
variable "analytics_app_config" {
  description = "Configuration map for Kinesis Analytics applications with multi-region support"
  type = map(object({
    name              = string
    input_stream      = string
    output_stream     = string
    processing_units  = number
    parallelism       = number
    error_threshold   = number
    tags             = map(string)
  }))

  validation {
    condition = alltrue([
      for k, v in var.analytics_app_config : (
        length(v.name) >= 1 &&
        length(v.name) <= 128 &&
        length(v.input_stream) > 0 &&
        length(v.output_stream) > 0 &&
        v.processing_units >= 1 &&
        v.processing_units <= 64 &&
        v.parallelism >= 1 &&
        v.parallelism <= 64 &&
        v.error_threshold >= 0 &&
        v.error_threshold <= 100
      )
    ])
    error_message = "Invalid Analytics application configuration. Check name length, stream names, processing units (1-64), parallelism (1-64), and error threshold (0-100)."
  }

  default = {
    game_analytics = {
      name             = "game-analytics"
      input_stream     = "game-events"
      output_stream    = "processed-events"
      processing_units = 4
      parallelism     = 2
      error_threshold = 5
      tags = {
        Environment = "dev"
        Purpose     = "real-time-analytics"
      }
    }
  }
}

# Time-based partitioning configuration
variable "partition_config" {
  description = "Configuration for time-based data partitioning"
  type = object({
    scan_data_retention_days = number
    partition_interval      = string
    enable_archiving       = bool
  })

  default = {
    scan_data_retention_days = 30
    partition_interval      = "MONTHLY"
    enable_archiving       = true
  }

  validation {
    condition = (
      var.partition_config.scan_data_retention_days >= 1 &&
      var.partition_config.scan_data_retention_days <= 365 &&
      contains(["HOURLY", "DAILY", "WEEKLY", "MONTHLY"], var.partition_config.partition_interval)
    )
    error_message = "Invalid partition configuration. Check retention days (1-365) and partition interval (HOURLY/DAILY/WEEKLY/MONTHLY)."
  }
}

# Multi-region deployment configuration
variable "multi_region_config" {
  description = "Configuration for multi-region analytics deployment"
  type = object({
    primary_region   = string
    replica_regions  = list(string)
    replication_mode = string
  })

  default = {
    primary_region   = "us-west-2"
    replica_regions  = ["us-east-1", "eu-west-1"]
    replication_mode = "SYNCHRONOUS"
  }

  validation {
    condition = (
      length(var.multi_region_config.replica_regions) <= 5 &&
      contains(["SYNCHRONOUS", "ASYNCHRONOUS"], var.multi_region_config.replication_mode)
    )
    error_message = "Invalid multi-region configuration. Maximum 5 replica regions allowed, replication mode must be SYNCHRONOUS or ASYNCHRONOUS."
  }
}