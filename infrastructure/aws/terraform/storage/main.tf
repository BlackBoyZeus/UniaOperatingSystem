# AWS Provider configuration
# AWS Provider version ~> 5.0
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Local variables for common resource tagging
locals {
  common_tags = {
    Project     = "TALD-UNIA"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# LiDAR Data S3 Bucket
resource "aws_s3_bucket" "lidar_data" {
  bucket = "tald-unia-lidar-${var.environment}-${var.region}"
  tags   = merge(local.common_tags, { StorageType = "LiDAR" })
}

resource "aws_s3_bucket_versioning" "lidar_versioning" {
  bucket = aws_s3_bucket.lidar_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lidar_encryption" {
  bucket = aws_s3_bucket.lidar_data.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = data.terraform_remote_state.kms.outputs.key_arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "lidar_lifecycle" {
  bucket = aws_s3_bucket.lidar_data.id

  rule {
    id     = "archive_old_scans"
    status = "Enabled"

    transition {
      days          = var.storage_settings.lidar_storage.retention_period
      storage_class = "GLACIER"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }
  }
}

# Game Assets S3 Bucket
resource "aws_s3_bucket" "game_assets" {
  bucket = "tald-unia-assets-${var.environment}-${var.region}"
  tags   = merge(local.common_tags, { StorageType = "GameAssets" })
}

resource "aws_s3_bucket_versioning" "assets_versioning" {
  bucket = aws_s3_bucket.game_assets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_cors_configuration" "assets_cors" {
  bucket = aws_s3_bucket.game_assets.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    max_age_seconds = 3600
  }
}

resource "aws_s3_bucket_accelerate_configuration" "assets_transfer" {
  bucket = aws_s3_bucket.game_assets.id
  status = "Enabled"
}

# Session Management DynamoDB Table
resource "aws_dynamodb_table" "sessions" {
  name           = "tald-unia-sessions-${var.environment}"
  billing_mode   = var.dynamodb_settings.session_table.billing_mode
  read_capacity  = var.dynamodb_settings.session_table.read_capacity
  write_capacity = var.dynamodb_settings.session_table.write_capacity
  hash_key       = "session_id"
  range_key      = "timestamp"

  attribute {
    name = "session_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  ttl {
    attribute_name = var.dynamodb_settings.session_table.ttl_attribute
    enabled        = var.dynamodb_settings.session_table.ttl_enabled
  }

  point_in_time_recovery {
    enabled = var.dynamodb_settings.session_table.point_in_time_recovery
  }

  stream_enabled   = var.dynamodb_settings.session_table.stream_enabled
  stream_view_type = var.dynamodb_settings.session_table.stream_view_type

  replica {
    region_name = var.replica_regions[0]
  }

  replica {
    region_name = var.replica_regions[1]
  }

  tags = merge(local.common_tags, { StorageType = "SessionData" })
}

# Auto-scaling for DynamoDB
resource "aws_appautoscaling_target" "dynamodb_table_read_target" {
  max_capacity       = 100
  min_capacity       = var.dynamodb_settings.session_table.read_capacity
  resource_id        = "table/${aws_dynamodb_table.sessions.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_target" "dynamodb_table_write_target" {
  max_capacity       = 100
  min_capacity       = var.dynamodb_settings.session_table.write_capacity
  resource_id        = "table/${aws_dynamodb_table.sessions.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}

# Backup configurations
resource "aws_backup_vault" "storage_backup" {
  name = "tald-unia-backup-${var.environment}"
  tags = local.common_tags
}

resource "aws_backup_plan" "storage_backup" {
  name = "tald-unia-backup-plan-${var.environment}"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.storage_backup.name
    schedule          = var.backup_settings.schedule

    lifecycle {
      cold_storage_after = var.backup_settings.cold_storage_after
      delete_after       = var.backup_settings.retention_days
    }
  }

  tags = local.common_tags
}

# Outputs
output "lidar_bucket" {
  value = {
    id                          = aws_s3_bucket.lidar_data.id
    arn                         = aws_s3_bucket.lidar_data.arn
    bucket_regional_domain_name = aws_s3_bucket.lidar_data.bucket_regional_domain_name
  }
}

output "game_assets_bucket" {
  value = {
    id                = aws_s3_bucket.game_assets.id
    arn               = aws_s3_bucket.game_assets.arn
    website_endpoint  = aws_s3_bucket.game_assets.website_endpoint
  }
}

output "session_table" {
  value = {
    id         = aws_dynamodb_table.sessions.id
    arn        = aws_dynamodb_table.sessions.arn
    stream_arn = aws_dynamodb_table.sessions.stream_arn
  }
}