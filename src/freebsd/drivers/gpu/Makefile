# TALD UNIA GPU Driver Makefile
# Version: 1.0.0
# Supports FreeBSD 9.0+ with Vulkan 1.3, SPIR-V Tools 2023.2, and Shaderc 2023.7

# Compiler and flags
CC=		clang
CFLAGS=		-O2 -pipe -fPIC -flto -march=native \
		-I/usr/local/include \
		-I. \
		-DNDEBUG \
		-DVK_USE_PLATFORM_FREEBSD_KHR \
		-D_KERNEL \
		-DKLD_MODULE \
		-DVULKAN_DRIVER_VERSION=\"1.0.0\" \
		-DSHADER_COMPILER_VERSION=\"1.0.0\"

LDFLAGS=	-L/usr/local/lib \
		-lvulkan \
		-lspirv-tools \
		-lshaderc \
		-pthread \
		-flto

# Source files
SRCS=		shader_compiler.c \
		vulkan_driver.c

# Object files
OBJS=		${SRCS:.c=.o}

# Target kernel module
KMOD=		gpu_driver
TARGET=		${KMOD}.ko

# Parallel build jobs based on CPU cores
MAKE_JOBS_NUMBER=	$(shell sysctl -n hw.ncpu)

# Default target
all: ${TARGET}

# Build kernel module
${TARGET}: ${OBJS}
	@echo "Building ${TARGET}..."
	${CC} -shared ${CFLAGS} ${LDFLAGS} -o ${TARGET} ${OBJS}
	@echo "Generating module dependencies..."
	depmod -a
	@echo "Verifying module symbols..."
	nm -D ${TARGET}

# Pattern rule for object files
%.o: %.c
	@echo "Compiling $<..."
	${CC} ${CFLAGS} -c $< -o $@
	@echo "Generating dependencies for $<..."
	${CC} ${CFLAGS} -MM $< > $*.d

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f ${OBJS}
	rm -f ${TARGET}
	rm -f *.d
	rm -f *~
	rm -f *.ltrans.*
	rm -f *.sym
	@echo "Clean complete"

# Install the kernel module
install: ${TARGET}
	@echo "Installing ${TARGET}..."
	@if kldstat | grep -q ${KMOD}; then \
		kldunload ${KMOD}; \
	fi
	install -o root -g wheel -m 0555 ${TARGET} /boot/modules/
	depmod -a
	@echo "Installation complete"
	@echo "To load the module: kldload ${KMOD}"

# Include generated dependency files
-include $(SRCS:.c=.d)

# Phony targets
.PHONY: all clean install

# Build with parallel jobs
.MAKEFLAGS: -j${MAKE_JOBS_NUMBER}