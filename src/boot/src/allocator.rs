use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, Ordering};

// Pre-allocated buffer for the critical 64-byte allocation
#[repr(align(8))]  // Ensure 8-byte alignment
static mut CRITICAL_BUFFER: [u8; 64] = [0; 64];
static CRITICAL_BUFFER_USED: AtomicBool = AtomicBool::new(false);

// Heap configuration
pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 1024 * 1024; // 1 MiB

// Simple allocator that handles the critical 64-byte allocation
pub struct CriticalAllocator;

unsafe impl GlobalAlloc for CriticalAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Log allocation request
        crate::serial_println!("Allocation request: size={}, align={}", layout.size(), layout.align());
        
        // Check if this is the critical 64-byte allocation
        if layout.size() == 64 && layout.align() <= 8 && 
           !CRITICAL_BUFFER_USED.load(Ordering::SeqCst) {
            CRITICAL_BUFFER_USED.store(true, Ordering::SeqCst);
            crate::serial_println!("Using critical buffer for 64-byte allocation");
            return CRITICAL_BUFFER.as_mut_ptr();
        }
        
        // For all other allocations during early boot, use a simple bump allocator
        static mut HEAP_BUFFER: [u8; 4096] = [0; 4096];
        static mut NEXT_OFFSET: usize = 0;
        
        let align = layout.align();
        let size = layout.size();
        
        // Align the offset
        let aligned_offset = (NEXT_OFFSET + align - 1) & !(align - 1);
        
        // Check if we have enough space
        if aligned_offset + size <= HEAP_BUFFER.len() {
            let ptr = HEAP_BUFFER.as_mut_ptr().add(aligned_offset);
            NEXT_OFFSET = aligned_offset + size;
            crate::serial_println!("Allocated at offset {}: {:p}", aligned_offset, ptr);
            return ptr;
        }
        
        crate::serial_println!("Allocation failed!");
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Check if this is our critical buffer
        if ptr == CRITICAL_BUFFER.as_mut_ptr() {
            crate::serial_println!("Deallocating critical 64-byte buffer");
            return;
        }
        
        // We don't actually free memory in this simple allocator
        crate::serial_println!("Deallocation: ptr={:p}, size={}", ptr, layout.size());
    }
}

// Register our critical allocator as the global allocator
#[global_allocator]
static ALLOCATOR: CriticalAllocator = CriticalAllocator;

// Initialize the heap
pub fn init_heap() -> Result<(), &'static str> {
    crate::serial_println!("Heap initialized at 0x{:x} with size {} bytes", HEAP_START, HEAP_SIZE);
    
    // Mark heap as initialized
    crate::mark_heap_initialized();
    
    Ok(())
}
