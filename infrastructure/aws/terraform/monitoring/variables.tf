# Environment variable with validation
variable "environment" {
  type        = string
  description = "Deployment environment (e.g., dev, staging, prod)"
  validation {
    condition     = can(regex("^(dev|staging|prod)$", var.environment))
    error_message = "Environment must be dev, staging, or prod"
  }
}

# AWS region with default value
variable "region" {
  type        = string
  description = "AWS region for monitoring infrastructure deployment"
  default     = "us-west-2"
}

# Prometheus retention configuration
variable "prometheus_retention_period" {
  type        = number
  description = "Retention period in days for Prometheus metrics"
  default     = 15
  validation {
    condition     = var.prometheus_retention_period >= 1 && var.prometheus_retention_period <= 90
    error_message = "Prometheus retention period must be between 1 and 90 days"
  }
}

# Grafana admin credentials
variable "grafana_admin_password" {
  type        = string
  description = "Initial admin password for Grafana workspace"
  sensitive   = true
}

# Metric alarm configurations
variable "metric_alarms" {
  type = map(object({
    threshold           = number
    evaluation_periods = number
    period             = number
  }))
  description = "Configuration for CloudWatch metric alarms"
  default = {
    lidar_latency = {
      threshold           = 50  # 50ms max latency
      evaluation_periods = 3
      period             = 60
    }
    fleet_latency = {
      threshold           = 50  # 50ms max P2P latency
      evaluation_periods = 3
      period             = 60
    }
    game_fps = {
      threshold           = 58  # Minimum 58 FPS (16.6ms frame time)
      evaluation_periods = 3
      period             = 60
    }
    failed_auth_attempts = {
      threshold           = 5   # Max 5 failed attempts
      evaluation_periods = 1
      period             = 300
    }
    memory_usage = {
      threshold           = 3800  # 3.8GB RAM threshold
      evaluation_periods = 3
      period             = 60
    }
    point_cloud_generation = {
      threshold           = 1000000  # 1M points/second
      evaluation_periods = 3
      period             = 60
    }
  }
}

# Log retention configuration
variable "log_retention_days" {
  type        = number
  description = "Retention period in days for CloudWatch log groups"
  default     = 30
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period"
  }
}

# Security monitoring configuration
variable "security_monitoring_config" {
  type = object({
    network_anomaly_threshold       = number
    system_integrity_check_interval = number
    fleet_trust_threshold          = number
    update_staleness_threshold     = number
  })
  description = "Configuration for security monitoring thresholds"
  default = {
    network_anomaly_threshold       = 2    # Standard deviations for network anomaly detection
    system_integrity_check_interval = 300  # 5 minutes interval for system integrity checks
    fleet_trust_threshold          = 80   # Minimum 80% trust score
    update_staleness_threshold     = 7    # Maximum 7 days behind on updates
  }
}

# Resource tagging
variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to monitoring resources"
  default     = {}
}