# TALD UNIA Core Services Makefile
# Version: 1.0.0
# Requires: FreeBSD Make 9.0+, cc 12.0+

SHELL = /bin/sh

# Compiler and flags
CC = cc
CFLAGS = -O2 -pipe -fPIC -Wall -Wextra -fstack-protector-strong -D_FORTIFY_SOURCE=2 \
         -I../include -I../kernel -I../lib/libsecurity -I../lib/libtald
LDFLAGS = -L../lib/libsecurity -L../lib/libtald -Wl,-z,relro,-z,now \
          -lsecurity -ltald -lcuda -lvulkan -lpthread

# Installation paths
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
ETCDIR = $(PREFIX)/etc/tald
VERSION = 1.0.0

# Service binaries
SERVICES = device_manager fleet_coordinator game_engine tald_init
SRCS = $(SERVICES:=.c)
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)
BINS = $(SERVICES)

# Default target
.PHONY: all
all: check $(BINS)

# Include dependency files
-include $(DEPS)

# Pattern rules for building
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

%: %.o
	$(CC) $< $(LDFLAGS) -o $@
	@sha256 $@ > $@.sha256

# Installation target
.PHONY: install
install: all
	@echo "Installing TALD UNIA services..."
	@install -d -m 755 $(DESTDIR)$(BINDIR)
	@install -d -m 755 $(DESTDIR)$(ETCDIR)
	@for service in $(BINS); do \
		install -s -m 4750 $$service $(DESTDIR)$(BINDIR)/tald_$$service; \
		install -m 644 $$service.sha256 $(DESTDIR)$(ETCDIR)/$$service.sha256; \
	done
	@echo "Version $(VERSION)" > $(DESTDIR)$(ETCDIR)/version
	@echo "Installation complete"

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJS) $(DEPS) $(BINS) *.sha256
	@echo "Clean complete"

# Dependency check target
.PHONY: check
check:
	@echo "Checking build requirements..."
	@command -v $(CC) >/dev/null 2>&1 || \
		{ echo "Error: $(CC) compiler not found"; exit 1; }
	@$(CC) --version | grep -q "clang version" || \
		{ echo "Error: Clang compiler required"; exit 1; }
	@for lib in cuda vulkan; do \
		pkg-config --exists $$lib || \
		{ echo "Error: $$lib development files not found"; exit 1; }; \
	done
	@echo "Build requirements satisfied"

# Individual service targets
device_manager: device_manager.o
	$(CC) $< $(LDFLAGS) -o $@
	@sha256 $@ > $@.sha256

fleet_coordinator: fleet_coordinator.o
	$(CC) $< $(LDFLAGS) -o $@
	@sha256 $@ > $@.sha256

game_engine: game_engine.o
	$(CC) $< $(LDFLAGS) -o $@
	@sha256 $@ > $@.sha256

tald_init: tald_init.o
	$(CC) $< $(LDFLAGS) -o $@
	@sha256 $@ > $@.sha256

# Help target
.PHONY: help
help:
	@echo "TALD UNIA Core Services Build System"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all services (default)"
	@echo "  install  - Install services to $(PREFIX)"
	@echo "  clean    - Remove build artifacts"
	@echo "  check    - Verify build requirements"
	@echo "  help     - Show this help message"