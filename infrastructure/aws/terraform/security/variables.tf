# Terraform AWS Security Variables Configuration
# Version: 1.0
# Provider version: hashicorp/terraform ~> 1.0

# KMS encryption key configuration variables
variable "kms_config" {
  description = "Configuration for KMS encryption key settings"
  type = object({
    key_rotation_enabled  = bool
    deletion_window_days = number
    key_usage           = string
    algorithm           = string
  })
  
  default = {
    key_rotation_enabled  = true
    deletion_window_days = 7
    key_usage           = "ENCRYPT_DECRYPT"
    algorithm           = "AES_256_GCM"
  }

  validation {
    condition     = var.kms_config.deletion_window_days >= 7 && var.kms_config.deletion_window_days <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days"
  }
}

# Cognito user pool configuration variables
variable "cognito_config" {
  description = "Configuration for Cognito user pool settings"
  type = object({
    mfa_configuration      = string
    token_validity_hours   = number
    allowed_oauth_flows    = list(string)
    hardware_token_enabled = bool
    minimum_password_length = number
    require_symbols        = bool
    require_numbers        = bool
    require_uppercase      = bool
    require_lowercase      = bool
  })

  default = {
    mfa_configuration      = "ON"
    token_validity_hours   = 24
    allowed_oauth_flows    = ["code", "implicit"]
    hardware_token_enabled = true
    minimum_password_length = 12
    require_symbols        = true
    require_numbers        = true
    require_uppercase      = true
    require_lowercase      = true
  }

  validation {
    condition     = contains(["OFF", "ON", "OPTIONAL"], var.cognito_config.mfa_configuration)
    error_message = "MFA configuration must be OFF, ON, or OPTIONAL"
  }
}

# Security group configuration variables
variable "security_group_config" {
  description = "Configuration for security group settings"
  type = object({
    allowed_cidr_blocks = list(string)
    webrtc_ports       = list(number)
    https_port         = number
    enable_egress      = bool
    egress_cidr_blocks = list(string)
  })

  default = {
    allowed_cidr_blocks = ["0.0.0.0/0"]
    webrtc_ports       = [50000, 50001, 50002]
    https_port         = 443
    enable_egress      = true
    egress_cidr_blocks = ["0.0.0.0/0"]
  }

  validation {
    condition     = length(var.security_group_config.webrtc_ports) > 0
    error_message = "At least one WebRTC port must be specified"
  }
}

# Security monitoring configuration variables
variable "monitoring_config" {
  description = "Configuration for security monitoring and alerts"
  type = object({
    failed_auth_threshold    = number
    alert_evaluation_period = number
    enable_anomaly_detection = bool
    log_retention_days      = number
    enable_audit_logs       = bool
  })

  default = {
    failed_auth_threshold    = 5
    alert_evaluation_period = 300
    enable_anomaly_detection = true
    log_retention_days      = 90
    enable_audit_logs       = true
  }

  validation {
    condition     = var.monitoring_config.log_retention_days >= 90
    error_message = "Log retention must be at least 90 days for security compliance"
  }
}