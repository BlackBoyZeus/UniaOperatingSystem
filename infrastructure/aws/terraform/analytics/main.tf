# AWS Provider configuration with version constraint
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Local variables for common resource naming and tagging
locals {
  name_prefix = "${var.environment}-tald-analytics"
  common_tags = {
    Project            = "TALD-UNIA"
    Environment        = var.environment
    ManagedBy         = "Terraform"
    Component         = "Analytics"
    DataClassification = "Sensitive"
    CostCenter        = "Gaming-Analytics"
  }
}

# KMS key for data encryption
resource "aws_kms_key" "analytics_key" {
  description             = "KMS key for TALD UNIA analytics data encryption"
  deletion_window_in_days = 7
  enable_key_rotation    = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-kms-key"
  })
}

# Game events Kinesis stream
resource "aws_kinesis_stream" "game_events" {
  name                = "${local.name_prefix}-game-events"
  shard_count         = var.kinesis_stream_config["game_events"].shard_count
  retention_period    = var.kinesis_stream_config["game_events"].retention_period
  encryption_type     = "KMS"
  kms_key_id         = aws_kms_key.analytics_key.arn

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
    "WriteProvisionedThroughputExceeded",
    "ReadProvisionedThroughputExceeded",
    "IteratorAgeMilliseconds"
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }

  tags = merge(local.common_tags, var.kinesis_stream_config["game_events"].tags)
}

# LiDAR data Kinesis stream
resource "aws_kinesis_stream" "lidar_data" {
  name                = "${local.name_prefix}-lidar-data"
  shard_count         = 20  # Higher shard count for LiDAR data volume
  retention_period    = 48  # 48 hours retention for high-volume data
  encryption_type     = "KMS"
  kms_key_id         = aws_kms_key.analytics_key.arn

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
    "WriteProvisionedThroughputExceeded",
    "ReadProvisionedThroughputExceeded",
    "IteratorAgeMilliseconds"
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }

  tags = merge(local.common_tags, {
    DataType = "LiDAR"
  })
}

# Game analytics application
resource "aws_kinesis_analytics_application" "game_analytics" {
  name = "${local.name_prefix}-game-analytics"

  inputs {
    name_prefix = "SOURCE_SQL_STREAM"
    
    kinesis_stream {
      resource_arn = aws_kinesis_stream.game_events.arn
      role_arn     = aws_iam_role.kinesis_analytics_role.arn
    }

    parallelism {
      count = var.analytics_app_config["game_analytics"].parallelism
    }

    schema_version = "1.0"

    processing_configuration {
      lambda {
        resource_arn = aws_lambda_function.analytics_processor.arn
        role_arn     = aws_iam_role.kinesis_analytics_role.arn
      }
    }
  }

  cloudwatch_logging_options {
    log_stream_arn = aws_cloudwatch_log_stream.analytics_logs.arn
    role_arn       = aws_iam_role.kinesis_analytics_role.arn
  }

  tags = merge(local.common_tags, var.analytics_app_config["game_analytics"].tags)
}

# IAM role for Kinesis Analytics
resource "aws_iam_role" "kinesis_analytics_role" {
  name = "${local.name_prefix}-analytics-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "kinesisanalytics.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# CloudWatch Log Group for analytics
resource "aws_cloudwatch_log_group" "analytics_logs" {
  name              = "/aws/kinesis-analytics/${local.name_prefix}"
  retention_in_days = 30

  tags = local.common_tags
}

# CloudWatch Log Stream
resource "aws_cloudwatch_log_stream" "analytics_logs" {
  name           = "analytics-logs"
  log_group_name = aws_cloudwatch_log_group.analytics_logs.name
}

# Lambda function for analytics processing
resource "aws_lambda_function" "analytics_processor" {
  filename         = "analytics_processor.zip"
  function_name    = "${local.name_prefix}-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "index.handler"
  runtime         = "nodejs18.x"
  timeout         = 300
  memory_size     = 1024

  environment {
    variables = {
      ENVIRONMENT = var.environment
      STREAM_NAME = aws_kinesis_stream.game_events.name
    }
  }

  tags = local.common_tags
}

# IAM role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${local.name_prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Outputs
output "game_events_stream" {
  value = {
    stream_arn   = aws_kinesis_stream.game_events.arn
    stream_name  = aws_kinesis_stream.game_events.name
    shard_count  = aws_kinesis_stream.game_events.shard_count
  }
  description = "Game events Kinesis stream details"
}

output "lidar_data_stream" {
  value = {
    stream_arn      = aws_kinesis_stream.lidar_data.arn
    stream_name     = aws_kinesis_stream.lidar_data.name
    retention_period = aws_kinesis_stream.lidar_data.retention_period
  }
  description = "LiDAR data Kinesis stream details"
}

output "analytics_applications" {
  value = {
    app_arns = [aws_kinesis_analytics_application.game_analytics.arn]
    processing_units = {
      game_analytics = var.analytics_app_config["game_analytics"].processing_units
    }
    monitoring_config = {
      log_group  = aws_cloudwatch_log_group.analytics_logs.name
      log_stream = aws_cloudwatch_log_stream.analytics_logs.name
    }
  }
  description = "Kinesis Analytics applications details"
}