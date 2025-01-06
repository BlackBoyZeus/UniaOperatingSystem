# TALD UNIA Core Library Makefile
# Version: 1.0.0
# Build configuration for libtald with LiDAR, CUDA and TensorRT integration

# Library name and version
LIB=		libtald.so.1.0
SHLIB_MAJOR=	1
SHLIB_MINOR=	0

# Installation paths
INSTALL_PREFIX=	/usr/local
CUDA_PATH=	/usr/local/cuda
TENSORRT_PATH=	/usr/local/tensorrt

# Compiler and flags
CC?=		clang
AR?=		ar
STRIP?=		strip

# Source files
SRCS=		cuda_wrapper.c \
		lidar_processing.c \
		tensorrt_wrapper.c

OBJS=		$(SRCS:.c=.o)
DEPS=		$(SRCS:.c=.d)

# Include paths
INCS=		-I$(CUDA_PATH)/include \
		-I$(TENSORRT_PATH)/include \
		-I.

# Compiler flags
CFLAGS+=	-O3 \
		-march=native \
		-fPIC \
		-flto \
		-Wall \
		-Wextra \
		-Werror \
		-D_FORTIFY_SOURCE=2 \
		-fstack-protector-strong \
		$(INCS)

# Linker flags
LDFLAGS+=	-L$(CUDA_PATH)/lib64 \
		-L$(TENSORRT_PATH)/lib \
		-Wl,-z,relro,-z,now \
		-Wl,--as-needed \
		-lcudart \
		-lnvinfer \
		-lnvonnxparser

# Default target
all: check $(LIB)

# Generate dependencies
-include $(DEPS)

%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link shared library
$(LIB): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)
	$(CC) -shared -o $@.debug $(OBJS) $(LDFLAGS) -g
	$(STRIP) --only-keep-debug $@.debug -o $@.sym
	$(STRIP) $@
	objcopy --add-gnu-debuglink=$@.sym $@

# Installation
install: all
	install -d $(DESTDIR)$(INSTALL_PREFIX)/lib
	install -d $(DESTDIR)$(INSTALL_PREFIX)/include/tald
	install -d $(DESTDIR)/usr/lib/debug$(INSTALL_PREFIX)/lib
	install -m 755 $(LIB) $(DESTDIR)$(INSTALL_PREFIX)/lib/
	ln -sf $(LIB) $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIB:.1.0=)
	install -m 644 *.h $(DESTDIR)$(INSTALL_PREFIX)/include/tald/
	install -m 644 $(LIB).sym $(DESTDIR)/usr/lib/debug$(INSTALL_PREFIX)/lib/
	ldconfig -m $(DESTDIR)$(INSTALL_PREFIX)/lib

# Environment check
check:
	@command -v $(CC) >/dev/null 2>&1 || { echo "Error: $(CC) not found"; exit 1; }
	@test -d $(CUDA_PATH) || { echo "Error: CUDA not found at $(CUDA_PATH)"; exit 1; }
	@test -d $(TENSORRT_PATH) || { echo "Error: TensorRT not found at $(TENSORRT_PATH)"; exit 1; }

# Clean build artifacts
clean:
	rm -f $(OBJS) $(DEPS) $(LIB) $(LIB).debug $(LIB).sym
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIB)*
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/include/tald/*
	rm -f $(DESTDIR)/usr/lib/debug$(INSTALL_PREFIX)/lib/$(LIB).sym

.PHONY: all check clean install