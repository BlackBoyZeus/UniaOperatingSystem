# TALD UNIA Security Library Makefile
# Provides hardware-backed security features including TPM integration,
# key management, and secure storage
#
# External dependencies:
# - tpm2-tss v3.2.0: TPM2 Software Stack
# - openssl v3.0.0: Cryptographic operations

.include <bsd.own.mk>

LIB=		security
SHLIB_MAJOR=	1
SHLIB_MINOR=	0

# Source files
SRCS=		key_management.c \
		secure_storage.c \
		tpm_interface.c

# Compiler flags with security hardening
CFLAGS+=	-O2 \
		-pipe \
		-fstack-protector-strong \
		-fno-strict-aliasing \
		-D_FORTIFY_SOURCE=2 \
		-DTPM_VERSION=2 \
		-DNDEBUG

# Include paths
CPPFLAGS+=	-I${.CURDIR} \
		-I${.CURDIR}/../include \
		-I/usr/local/include/tpm2-tss

# Library dependencies
LDADD+=		-L${.CURDIR}/../lib \
		-L/usr/local/lib \
		-ltpm2-tss \
		-lcrypto

# Enable all warnings
WARNS=		6

# Version script for symbol visibility
CLEANFILES+=	version.map
VERSION_MAP=	${.CURDIR}/version.map

# Generate version script
version.map:
	@echo "LIBSECURITY_1.0 {" > ${VERSION_MAP}
	@echo "  global:" >> ${VERSION_MAP}
	@echo "    key_*;" >> ${VERSION_MAP}
	@echo "    secure_*;" >> ${VERSION_MAP}
	@echo "    tpm_*;" >> ${VERSION_MAP}
	@echo "  local: *;" >> ${VERSION_MAP}
	@echo "};" >> ${VERSION_MAP}

# Install headers
INCS=		key_management.h \
		secure_storage.h \
		tpm_interface.h
INCSDIR=	${PREFIX}/include/security

# Man pages
MAN=		libsecurity.3 \
		key_management.3 \
		secure_storage.3 \
		tpm_interface.3

# Build shared library
.if !defined(LIBRARIES_ONLY)
LIBDIR=		${PREFIX}/lib
SHLIBDIR=	${PREFIX}/lib
.endif

# Additional build targets
all: version.map
	@${MAKE} -f ${.CURDIR}/Makefile ${.TARGETS}

# Clean target
clean:
	rm -f ${CLEANFILES} *.o *.so* *.a *.d

# Install target
beforeinstall:
	${INSTALL} -d ${DESTDIR}${INCSDIR}
	${INSTALL} -d ${DESTDIR}${LIBDIR}

.include <bsd.lib.mk>