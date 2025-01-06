# AWS ECS Cluster Variables Configuration
# Version: 1.0
# Provider: hashicorp/aws ~> 5.0

variable "cluster_name" {
  type        = string
  description = "Name of the ECS cluster for TALD UNIA services"
  default     = "tald-unia-cluster"
}

variable "task_cpu" {
  type        = map(number)
  description = "CPU units allocation for each service type (1024 = 1 vCPU)"
  default = {
    fleet     = 2048  # 2 vCPU for Fleet Management Services
    game      = 4096  # 4 vCPU for Game Services
    analytics = 2048  # 2 vCPU for Analytics Services
  }
}

variable "task_memory" {
  type        = map(number)
  description = "Memory allocation in MB for each service type"
  default = {
    fleet     = 4096  # 4GB for Fleet Management Services
    game      = 8192  # 8GB for Game Services
    analytics = 6144  # 6GB for Analytics Services
  }
}

variable "service_scaling" {
  type = map(object({
    min           = number
    max           = number
    cpu_threshold = number
  }))
  description = "Service auto-scaling configuration for each service type including minimum and maximum instances and CPU utilization threshold percentage"
  default = {
    fleet = {
      min           = 2   # Minimum 2 instances for high availability
      max           = 10  # Maximum 10 instances for peak load
      cpu_threshold = 70  # Scale up when CPU utilization reaches 70%
    }
    game = {
      min           = 3   # Minimum 3 instances for game service redundancy
      max           = 15  # Maximum 15 instances for high player load
      cpu_threshold = 80  # Scale up when CPU utilization reaches 80%
    }
    analytics = {
      min           = 1   # Minimum 1 instance for analytics processing
      max           = 5   # Maximum 5 instances for peak analytics load
      cpu_threshold = 75  # Scale up when CPU utilization reaches 75%
    }
  }
}