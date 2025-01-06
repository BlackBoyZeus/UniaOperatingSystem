# Terraform version constraints and provider requirements for TALD UNIA platform
# This file defines version constraints for Terraform core and required providers
# to ensure consistent infrastructure deployment across environments

terraform {
  # Minimum required Terraform version
  # Required for advanced features like custom variable validation and improved provider configuration
  required_version = ">= 1.0"

  # Required provider configurations
  required_providers {
    # AWS provider for core infrastructure services
    # Version ~> 4.0 required for ECS Fargate, Lambda, and other AWS service features
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }

    # Random provider for generating unique identifiers
    # Used for resource naming and randomization needs
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }

    # Null provider for resource dependencies and triggers
    # Used for managing resource creation order and dependencies
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}