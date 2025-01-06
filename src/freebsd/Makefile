# TALD UNIA Operating System Root Makefile
# Version: 1.0.0
# FreeBSD Build System Version: 9.0

.include <bsd.own.mk>
.include <bsd.kmod.mk>

# Core build directories
SUBDIR=		drivers lib services kernel game

# Installation prefix
PREFIX?=	/usr/local/tald

# Version information
TALD_VERSION?=	1.0.0
MAKE_VERSION?=	FreeBSD 9.0

# Compiler optimization flags for performance targets
CFLAGS+=	-O2 -pipe -fPIC -I${PREFIX}/include
CFLAGS+=	-DNDEBUG -march=native -mtune=native
CFLAGS+=	-fomit-frame-pointer -flto
CFLAGS+=	-DTALD_VERSION=\"${TALD_VERSION}\"

# Linker optimization flags
LDFLAGS+=	-L${PREFIX}/lib -Wl,-O1
LDFLAGS+=	-Wl,--as-needed -Wl,--gc-sections

# Performance target validation
PERF_TARGETS=	\
	scan_latency_50ms \
	network_latency_50ms \
	framerate_60fps \
	fleet_size_32 \
	battery_life_4h

# Default target
all: check-env
	@echo "Building TALD UNIA OS components..."
	@for dir in ${SUBDIR}; do \
		(cd $$dir && ${MAKE} all) || exit $$?; \
	done
	@${MAKE} test
	@echo "Build complete. Run 'make test' for full validation."

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@for dir in ${SUBDIR}; do \
		(cd $$dir && ${MAKE} clean) || exit $$?; \
	done
	@rm -rf ${PREFIX}/tmp/* 
	@rm -f .depend

# Secure installation with validation
install: all
	@echo "Installing TALD UNIA components..."
	@install -d -m 755 ${PREFIX}
	@for dir in ${SUBDIR}; do \
		(cd $$dir && ${MAKE} install) || exit $$?; \
	done
	@${MAKE} verify-install
	@echo "Installation complete at ${PREFIX}"

# Comprehensive testing
test: all
	@echo "Running system tests..."
	@for target in ${PERF_TARGETS}; do \
		(${MAKE} test-$$target) || exit $$?; \
	done
	@echo "All tests passed."

# Environment validation
check-env:
	@echo "Validating build environment..."
	@which cc >/dev/null 2>&1 || \
		(echo "Error: C compiler not found"; exit 1)
	@which ld >/dev/null 2>&1 || \
		(echo "Error: Linker not found"; exit 1)
	@test -d /usr/include || \
		(echo "Error: System headers not found"; exit 1)

# Performance test targets
test-scan_latency_50ms:
	@echo "Testing LiDAR scan latency..."
	@cd drivers/lidar && ${MAKE} test-latency
	@[ $$? -eq 0 ] || (echo "Failed: Scan latency > 50ms"; exit 1)

test-network_latency_50ms:
	@echo "Testing network latency..."
	@cd drivers/network && ${MAKE} test-latency
	@[ $$? -eq 0 ] || (echo "Failed: Network latency > 50ms"; exit 1)

test-framerate_60fps:
	@echo "Testing GPU performance..."
	@cd drivers/gpu && ${MAKE} test-fps
	@[ $$? -eq 0 ] || (echo "Failed: Frame rate < 60 FPS"; exit 1)

test-fleet_size_32:
	@echo "Testing fleet management..."
	@cd services && ${MAKE} test-fleet
	@[ $$? -eq 0 ] || (echo "Failed: Fleet size < 32 devices"; exit 1)

test-battery_life_4h:
	@echo "Testing power consumption..."
	@cd services && ${MAKE} test-power
	@[ $$? -eq 0 ] || (echo "Failed: Battery life < 4 hours"; exit 1)

# Installation verification
verify-install:
	@echo "Verifying installation..."
	@test -x ${PREFIX}/bin/tald || \
		(echo "Error: Main binary not installed"; exit 1)
	@test -d ${PREFIX}/lib || \
		(echo "Error: Libraries not installed"; exit 1)
	@test -d ${PREFIX}/modules || \
		(echo "Error: Kernel modules not installed"; exit 1)

# Dependency management
depend:
	@echo "Updating dependencies..."
	@for dir in ${SUBDIR}; do \
		(cd $$dir && ${MAKE} depend) || exit $$?; \
	done

.include <bsd.subdir.mk>