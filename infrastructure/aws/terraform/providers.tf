# Terraform configuration block with required providers
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    null = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # S3 backend configuration for state management
  backend "s3" {
    bucket               = "tald-unia-terraform-state-${var.environment}"
    key                  = "terraform.tfstate"
    region              = "us-east-1"
    encrypt             = true
    dynamodb_table      = "tald-unia-terraform-locks-${var.environment}"
    kms_key_id          = "${var.state_encryption_key_arn}"
    workspace_key_prefix = "workspaces"
    acl                 = "private"
  }
}

# Main AWS provider configuration for primary region
provider "aws" {
  region              = var.aws_region
  allowed_account_ids = var.allowed_account_ids
  
  default_tags {
    tags = {
      Environment        = var.environment
      Project           = "TALD-UNIA"
      ManagedBy         = "Terraform"
      Component         = "Gaming-Platform"
      SecurityZone      = "Production"
      DataClassification = "Confidential"
    }
  }

  assume_role {
    role_arn     = var.terraform_role_arn
    session_name = "TerraformProviderSession"
    external_id  = var.terraform_external_id
  }
}

# Secondary AWS provider for global services (CloudFront, IAM, etc.)
provider "aws" {
  alias               = "global"
  region             = "us-east-1"
  allowed_account_ids = var.allowed_account_ids
  
  default_tags {
    tags = {
      Environment        = var.environment
      Project           = "TALD-UNIA"
      ManagedBy         = "Terraform"
      Component         = "Global-Services"
      SecurityZone      = "Production"
      DataClassification = "Confidential"
    }
  }
}

# Random provider for generating unique identifiers
provider "random" {}

# Null provider for dependency management
provider "null" {}