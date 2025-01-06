# TALD UNIA ECS Infrastructure Configuration
# Provider: hashicorp/aws ~> 5.0
# Last Updated: 2024

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

locals {
  common_tags = {
    Project      = "TALD-UNIA"
    Environment  = "${var.environment}"
    ManagedBy    = "Terraform"
    LastUpdated  = "${timestamp()}"
  }
}

# CloudWatch Log Group for ECS Cluster
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/aws/ecs/${var.cluster_name}"
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

# ECS Cluster with Enhanced Monitoring
resource "aws_ecs_cluster" "tald_unia" {
  name = var.cluster_name

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      log_configuration {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs_logs.name
      }
    }
  }

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
    base             = 1
  }

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight           = 4
  }

  tags = local.common_tags
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.cluster_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Role Policy Attachments
resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Fleet Management Service Task Definition
resource "aws_ecs_task_definition" "fleet_service" {
  family                   = "${var.cluster_name}-fleet"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = var.task_cpu["fleet"]
  memory                  = var.task_memory["fleet"]
  execution_role_arn      = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "fleet-service"
      image = "${data.aws_ecr_repository.fleet_service.repository_url}:latest"
      cpu   = var.task_cpu["fleet"]
      memory = var.task_memory["fleet"]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "fleet"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = local.common_tags
}

# Game Service Task Definition
resource "aws_ecs_task_definition" "game_service" {
  family                   = "${var.cluster_name}-game"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = var.task_cpu["game"]
  memory                  = var.task_memory["game"]
  execution_role_arn      = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "game-service"
      image = "${data.aws_ecr_repository.game_service.repository_url}:latest"
      cpu   = var.task_cpu["game"]
      memory = var.task_memory["game"]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "game"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8081/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = local.common_tags
}

# Analytics Service Task Definition
resource "aws_ecs_task_definition" "analytics_service" {
  family                   = "${var.cluster_name}-analytics"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = var.task_cpu["analytics"]
  memory                  = var.task_memory["analytics"]
  execution_role_arn      = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "analytics-service"
      image = "${data.aws_ecr_repository.analytics_service.repository_url}:latest"
      cpu   = var.task_cpu["analytics"]
      memory = var.task_memory["analytics"]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "analytics"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8082/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = local.common_tags
}

# Service Auto-Scaling Policies
resource "aws_appautoscaling_target" "ecs_target" {
  for_each           = var.service_scaling
  max_capacity       = each.value.max
  min_capacity       = each.value.min
  resource_id        = "service/${aws_ecs_cluster.tald_unia.name}/${each.key}-service"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_policy" {
  for_each           = var.service_scaling
  name               = "${each.key}-scaling-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[each.key].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[each.key].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[each.key].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = each.value.cpu_threshold
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# Outputs
output "cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.tald_unia.arn
  sensitive   = false
}

output "service_arns" {
  description = "ARNs of the ECS services"
  value = {
    fleet     = aws_ecs_task_definition.fleet_service.arn
    game      = aws_ecs_task_definition.game_service.arn
    analytics = aws_ecs_task_definition.analytics_service.arn
  }
}

output "task_definition_arns" {
  description = "ARNs of the ECS task definitions"
  value = {
    fleet     = aws_ecs_task_definition.fleet_service.arn
    game      = aws_ecs_task_definition.game_service.arn
    analytics = aws_ecs_task_definition.analytics_service.arn
  }
}