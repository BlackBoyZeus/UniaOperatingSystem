# TALD UNIA Network Driver Kernel Module Makefile
# Implements mesh networking with 32-device fleet support and WebRTC P2P communication
# Target latency: <50ms for P2P communication

.include <bsd.own.mk>

# Kernel module name
KMOD=		tald_network

# Source files
SRCS=		mesh_protocol.c \
		webrtc_native.c

# Required kernel files
SRCS+=		opt_inet.h \
		opt_inet6.h \
		device_if.h \
		bus_if.h \
		vnode_if.h

# Compiler flags
CFLAGS+=	-O2 \
		-g \
		-DKLD_MODULE \
		-DDEBUG \
		-fPIC \
		-Wall \
		-Wextra \
		-Werror \
		-DMAX_FLEET_SIZE=32 \
		-DMAX_LATENCY_MS=50 \
		-I/usr/local/include \
		-I${.CURDIR} \
		-I${SYSDIR}

# Linker flags
LDFLAGS+=	-L/usr/local/lib \
		-lwebrtc \
		-lssl \
		-lcrypto \
		-pthread

# Module installation directory
KMODDIR=	/boot/modules

# Files to clean
CLEANFILES+=	*.o *.ko *.d .depend

# Ensure proper module signing
SIGNING_KEY?=	/etc/ssl/tald/module.key
SIGNING_CERT?=	/etc/ssl/tald/module.crt

# Build targets
all: ${KMOD}.ko
	@echo "Build complete: ${KMOD}.ko"
	@echo "Fleet size limit: 32 devices"
	@echo "Target latency: <50ms"

# Clean target
clean:
	rm -f ${CLEANFILES}
	@echo "Cleaned build artifacts"

# Install target with proper permissions and signing
install: ${KMOD}.ko
	# Stop module if loaded
	-/sbin/kldunload ${KMOD} 2>/dev/null || true
	
	# Install module with correct permissions
	${INSTALL} -o root -g wheel -m 555 ${KMOD}.ko ${KMODDIR}/
	
	# Sign module if keys available
	if [ -f ${SIGNING_KEY} ] && [ -f ${SIGNING_CERT} ]; then \
		/usr/sbin/kmodsign sha256 ${SIGNING_KEY} ${SIGNING_CERT} \
			${KMODDIR}/${KMOD}.ko; \
	fi
	
	# Update dependencies
	/usr/sbin/kldxref ${KMODDIR}
	@echo "Installed ${KMOD}.ko to ${KMODDIR}"

# Include FreeBSD kernel module build system
.include <bsd.kmod.mk>