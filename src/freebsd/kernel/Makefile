# TALD UNIA Kernel Module Makefile
# FreeBSD 9.0 Base
# Optimized for performance and security with hardware-specific tuning

.include <bsd.own.mk>

# Module name
KMOD=   tald

# Source files
SRCS=   gpu_driver.c \
        lidar_driver.c \
        memory_protection.c \
        mesh_network.c \
        power_mgmt.c \
        secure_boot.c \
        tald_core.c

# Compiler optimization flags
CFLAGS= -O2 \
        -pipe \
        -fno-strict-aliasing \
        -mcpu=native \
        -march=native \
        -D_KERNEL \
        -DKLD_MODULE \
        -nostdinc \
        -fstack-protector-strong \
        -fPIE \
        -pie \
        -Wl,-z,noexecstack \
        -Wl,-z,relro \
        -Wl,-z,now \
        -I.

# C++ specific flags
CXXFLAGS= ${CFLAGS} \
          -fno-rtti \
          -fno-exceptions \
          -fvisibility=hidden

# Linker flags for security hardening
LDFLAGS= -L/usr/lib \
         -lm \
         -Wl,-z,relro \
         -Wl,-z,now \
         -fPIE \
         -pie

# Include paths for dependencies
INCS=   -I${.CURDIR}/../../include \
        -I${.CURDIR}/../../drivers/gpu \
        -I${.CURDIR}/../../drivers/lidar \
        -I${.CURDIR}/../../drivers/network \
        -I${.CURDIR}/../../security

# Debug flags when DEBUG is defined
.if defined(DEBUG)
CFLAGS+= -g -DDEBUG -DKLD_DEBUG
.endif

# Module signing key path
SIGNING_KEY?= /etc/ssl/private/kernel_signing_key.pem

# Dependency tracking
.for src in ${SRCS}
CLEANFILES+= ${src:R}.d
.endfor

# Default target
all: module_sign

# Build and sign the kernel module
module_sign: ${KMOD}.ko
	@echo "Signing kernel module with ${SIGNING_KEY}"
	@if [ -f ${SIGNING_KEY} ]; then \
		kmodsign sha512 ${SIGNING_KEY} ${KMOD}.ko ${KMOD}.ko.signed && \
		mv ${KMOD}.ko.signed ${KMOD}.ko; \
	else \
		echo "Warning: Signing key not found at ${SIGNING_KEY}"; \
	fi

# Clean build artifacts
clean: 
	rm -f *.o *.ko *.ko.signed *.d .depend
	rm -f ${CLEANFILES}
	rm -f machine x86

# Install the module
install: all
	@echo "Installing ${KMOD}.ko to /boot/modules"
	@mkdir -p /boot/modules
	@install -o root -g wheel -m 555 ${KMOD}.ko /boot/modules/
	@if [ -f ${KMOD}.ko.symbols ]; then \
		install -o root -g wheel -m 444 ${KMOD}.ko.symbols /boot/modules/; \
	fi
	@kldxref /boot/modules

# Hardware-specific optimizations
.if ${MACHINE_ARCH} == "amd64"
CFLAGS+= -mavx2 -msse4.2
.endif

# LiDAR processing optimizations
.if defined(LIDAR_OPT)
CFLAGS+= -DLIDAR_PROCESSING_30HZ -DPOINT_CLOUD_OPT
.endif

# Mesh networking optimizations
.if defined(MESH_OPT)
CFLAGS+= -DMESH_32_DEVICES -DLOW_LATENCY_NET
.endif

# Security hardening
CFLAGS+= -fstack-protector-all \
         -D_FORTIFY_SOURCE=2 \
         -DKLD_SECURITY_POLICY \
         -DASLR_ENABLE \
         -DDEP_ENABLE

# Performance monitoring hooks
.if defined(PERF_MONITOR)
CFLAGS+= -DPERF_HOOKS -DLATENCY_TRACKING
.endif

# Battery optimization
.if defined(POWER_OPT)
CFLAGS+= -DPOWER_SAVE_MODE -DBATTERY_LIFE_4H
.endif

.include <bsd.kmod.mk>