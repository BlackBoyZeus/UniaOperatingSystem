# Configure Terraform and required providers
terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Define local variables for reuse
locals {
  cdn_domain      = "cdn.${var.domain_name}"
  s3_origin_id    = "GameAssetsOrigin"
  lidar_origin_id = "LiDARDataOrigin"
  security_headers = {
    "Strict-Transport-Security" = "max-age=31536000; includeSubDomains; preload"
    "X-Content-Type-Options"   = "nosniff"
    "X-Frame-Options"          = "DENY"
    "X-XSS-Protection"         = "1; mode=block"
  }
}

# Create Origin Access Identity for secure S3 access
resource "aws_cloudfront_origin_access_identity" "cdn_oai" {
  comment = "OAI for ${var.environment} TALD UNIA CDN access"
}

# Create cache policy for game assets
resource "aws_cloudfront_cache_policy" "game_assets" {
  name        = "tald-unia-game-assets-${var.environment}"
  comment     = "Cache policy for TALD UNIA game assets"
  min_ttl     = var.cache_ttl.min_ttl
  default_ttl = var.cache_ttl.default_ttl
  max_ttl     = var.cache_ttl.max_ttl

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    cookies_config {
      cookie_behavior = "none"
    }

    headers_config {
      header_behavior = "whitelist"
      headers {
        items = ["Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"]
      }
    }

    query_strings_config {
      query_string_behavior = "whitelist"
      query_strings {
        items = ["v"]
      }
    }
  }
}

# Create CloudFront distribution
resource "aws_cloudfront_distribution" "cdn" {
  enabled             = true
  is_ipv6_enabled    = true
  comment            = "TALD UNIA CDN - ${var.environment}"
  price_class        = var.price_class
  aliases            = [local.cdn_domain]
  web_acl_id         = var.waf_acl_arn
  wait_for_deployment = false

  # Game assets origin configuration
  origin {
    domain_name = data.terraform_remote_state.storage.outputs.game_assets_bucket.bucket_regional_domain_name
    origin_id   = local.s3_origin_id

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.cdn_oai.cloudfront_access_identity_path
    }

    origin_shield {
      enabled              = true
      origin_shield_region = var.origin_shield_region
    }

    custom_header {
      name  = "X-Origin-Verify"
      value = random_uuid.origin_verify.result
    }
  }

  # LiDAR data origin configuration
  origin {
    domain_name = data.terraform_remote_state.storage.outputs.lidar_data_bucket.bucket_regional_domain_name
    origin_id   = local.lidar_origin_id

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.cdn_oai.cloudfront_access_identity_path
    }

    origin_shield {
      enabled              = true
      origin_shield_region = var.origin_shield_region
    }
  }

  # Default cache behavior for game assets
  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD", "OPTIONS"]
    target_origin_id       = local.s3_origin_id
    viewer_protocol_policy = "redirect-to-https"
    compress              = true

    cache_policy_id          = aws_cloudfront_cache_policy.game_assets.id
    origin_request_policy_id = aws_cloudfront_origin_request_policy.game_assets.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id

    function_association {
      event_type   = "viewer-response"
      function_arn = aws_cloudfront_function.security_headers.arn
    }
  }

  # LiDAR data cache behavior
  ordered_cache_behavior {
    path_pattern           = "/lidar/*"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = local.lidar_origin_id
    viewer_protocol_policy = "https-only"
    compress              = true

    cache_policy_id          = aws_cloudfront_cache_policy.lidar_data.id
    origin_request_policy_id = aws_cloudfront_origin_request_policy.lidar_data.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id
  }

  # SSL/TLS configuration
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.cdn.arn
    minimum_protocol_version = var.ssl_protocol_version
    ssl_support_method       = "sni-only"
  }

  # Custom error responses
  custom_error_response {
    error_code            = 403
    response_code         = 404
    response_page_path    = "/404.html"
    error_caching_min_ttl = 10
  }

  custom_error_response {
    error_code            = 404
    response_code         = 404
    response_page_path    = "/404.html"
    error_caching_min_ttl = 10
  }

  # Geo restrictions
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = merge(var.tags, {
    Name = "tald-unia-cdn-${var.environment}"
  })
}

# Create security headers response policy
resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name    = "tald-unia-security-headers-${var.environment}"
  comment = "Security headers policy for TALD UNIA CDN"

  security_headers_config {
    strict_transport_security {
      override                   = true
      access_control_max_age_sec = 31536000
      include_subdomains        = true
      preload                   = true
    }

    content_type_options {
      override = true
    }

    frame_options {
      override     = true
      frame_option = "DENY"
    }

    xss_protection {
      override   = true
      mode_block = true
      protection = true
    }
  }
}

# Export CloudFront distribution details
output "cloudfront_distribution_id" {
  value       = aws_cloudfront_distribution.cdn.id
  description = "CloudFront distribution ID"
}

output "cloudfront_domain_name" {
  value       = aws_cloudfront_distribution.cdn.domain_name
  description = "CloudFront distribution domain name"
}

output "origin_access_identity" {
  value = {
    iam_arn = aws_cloudfront_origin_access_identity.cdn_oai.iam_arn
    path    = aws_cloudfront_origin_access_identity.cdn_oai.cloudfront_access_identity_path
  }
  description = "CloudFront Origin Access Identity details"
}