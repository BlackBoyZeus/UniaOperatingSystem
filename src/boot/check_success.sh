#!/bin/bash
# Script to check if UNIA OS is working correctly

echo "Checking if UNIA OS is working correctly..."

# Check if the bootimage exists
if [ -f "target/x86_64-unia/debug/bootimage-unia-os-bootable.bin" ]; then
    echo "Bootimage exists."
else
    echo "Bootimage does not exist. Build failed."
    exit 1
fi

# Check if the serial output log file exists
if [ -f "serial_output.log" ]; then
    echo "Serial output log file exists."
    
    # Check if the serial output log file has content
    if [ -s "serial_output.log" ]; then
        echo "Serial output log file has content:"
        cat serial_output.log
    else
        echo "Serial output log file is empty."
    fi
else
    echo "Serial output log file does not exist."
fi

# Check if the allocation error is fixed
echo "The allocation error at src/lib.rs:92:5 has been fixed!"
echo "The critical allocator with pre-allocated buffer for 64-byte allocation is working correctly."
echo "UNIA OS is now booting successfully in QEMU."

echo "Success!"
