# Core Terraform functionality for variable definitions
terraform {
  required_version = ">=1.0.0"
}

# Deployment environment variable with validation
variable "environment" {
  type        = string
  description = "Deployment environment (dev/staging/prod)"

  validation {
    condition     = can(regex("^(dev|staging|prod)$", var.environment))
    error_message = "Environment must be dev, staging, or prod"
  }
}

# Domain name variable with DNS format validation
variable "domain_name" {
  type        = string
  description = "Base domain name for CDN endpoint configuration"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\\.[a-z]{2,}$", var.domain_name))
    error_message = "Domain name must be a valid DNS name"
  }
}

# CloudFront price class variable with predefined options
variable "price_class" {
  type        = string
  description = "CloudFront distribution price class for global edge location coverage"
  default     = "PriceClass_100"

  validation {
    condition     = can(regex("^PriceClass_(100|200|All)$", var.price_class))
    error_message = "Price class must be PriceClass_100, PriceClass_200, or PriceClass_All"
  }
}

# Cache TTL settings with validation for proper ordering
variable "cache_ttl" {
  type = object({
    min_ttl     = number
    default_ttl = number
    max_ttl     = number
  })
  description = "Cache TTL settings for optimizing content delivery and reducing origin load"
  default = {
    min_ttl     = 0
    default_ttl = 3600    # 1 hour
    max_ttl     = 86400   # 24 hours
  }

  validation {
    condition     = var.cache_ttl.min_ttl >= 0 && var.cache_ttl.default_ttl >= var.cache_ttl.min_ttl && var.cache_ttl.max_ttl >= var.cache_ttl.default_ttl
    error_message = "TTL values must be in ascending order: min_ttl <= default_ttl <= max_ttl"
  }
}

# SSL/TLS protocol version with security-focused defaults
variable "ssl_protocol_version" {
  type        = string
  description = "Minimum TLS version for secure content delivery"
  default     = "TLSv1.3"

  validation {
    condition     = can(regex("^TLSv1\\.[23]$", var.ssl_protocol_version))
    error_message = "SSL protocol version must be TLSv1.2 or TLSv1.3"
  }
}

# Origin shield region for improved cache hit ratio
variable "origin_shield_region" {
  type        = string
  description = "AWS region for CloudFront origin shield to improve cache hit ratio"
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-\\d$", var.origin_shield_region))
    error_message = "Origin shield region must be a valid AWS region format"
  }
}

# Resource tagging for infrastructure management
variable "tags" {
  type        = map(string)
  description = "Resource tags for CDN infrastructure management and cost allocation"
  default = {
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
    Environment = "var.environment"
  }
}