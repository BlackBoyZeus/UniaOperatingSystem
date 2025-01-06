# TALD UNIA LiDAR Hardware Driver Makefile
# Version: 1.0.0
#
# Builds FreeBSD kernel module for LiDAR hardware driver supporting:
# - 30Hz continuous scanning
# - 0.01cm resolution
# - 5-meter effective range
# - Real-time point cloud processing
# - Safety features and thermal monitoring

# FreeBSD v9.0 kernel module name
KMOD=	lidar

# Source files
SRCS=	lidar_hw.c \
	lidar_calibration.c

# Compiler flags
CFLAGS+=	-O2 \
		-DLIDAR_SCAN_FREQ=30 \
		-DLIDAR_RESOLUTION=0.01 \
		-DLIDAR_RANGE=5000 \
		-DLIDAR_SAFETY_CHECKS \
		-DLIDAR_THERMAL_MONITOR

# Warning level
WARNS=	2

# Debug flags (enabled for development)
DEBUG_FLAGS=	-g \
		-DLIDAR_DEBUG \
		-DLIDAR_DEBUG_SYMBOLS

# Safety feature flags
SAFETY_FLAGS=	-DLIDAR_SAFETY_CHECKS \
		-DLIDAR_THERMAL_MONITOR \
		-DLIDAR_SAFETY_INTERLOCKS

# Include FreeBSD kernel module build rules
.include <bsd.kmod.mk>