use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use crate::debug;
use crate::serial_println;

// Heap configuration
pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 1024 * 1024; // 1 MiB

// Static buffer for early allocations (128KB)
static mut EARLY_HEAP_BUFFER: [u8; 131072] = [0; 131072];
static EARLY_HEAP_NEXT: AtomicUsize = AtomicUsize::new(0);

// Debugging allocator that tracks all allocations
pub struct DebugAllocator;

unsafe impl GlobalAlloc for DebugAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Track this allocation attempt
        let size = layout.size();
        let align = layout.align();
        
        // Log allocation details
        serial_println!("Allocation request: size={}, align={}", size, align);
        debug::track_allocation(size, align);
        
        // Special handling for the problematic 64-byte allocation
        if size == 64 && align == 8 && !debug::SPECIAL_BUFFER_USED {
            debug::SPECIAL_BUFFER_USED = true;
            serial_println!("Using special buffer for 64-byte allocation");
            return debug::SPECIAL_BUFFER_64.as_mut_ptr();
        }
        
        // Simple bump allocation from static buffer
        let current = EARLY_HEAP_NEXT.load(Ordering::Relaxed);
        let aligned_start = align_up(current, align);
        let end = match aligned_start.checked_add(size) {
            Some(end) => end,
            None => return null_mut(), // Overflow
        };
        
        // Check if we have enough space
        if end <= EARLY_HEAP_BUFFER.len() {
            // Update the next position atomically
            if EARLY_HEAP_NEXT.compare_exchange(
                current, end, Ordering::SeqCst, Ordering::SeqCst
            ).is_ok() {
                // Return pointer to the allocated memory
                serial_println!("Allocation successful: ptr=0x{:x}", 
                               EARLY_HEAP_BUFFER.as_ptr() as usize + aligned_start);
                return EARLY_HEAP_BUFFER.as_mut_ptr().add(aligned_start);
            }
        }
        
        // Out of memory or CAS failed
        serial_println!("Allocation failed!");
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Check if this is our special buffer
        if ptr == debug::SPECIAL_BUFFER_64.as_mut_ptr() {
            serial_println!("Deallocating special 64-byte buffer");
            return;
        }
        
        // We don't actually free memory in this simple allocator
        serial_println!("Deallocation: ptr={:p}, size={}", ptr, layout.size());
    }
}

// Register our debug allocator as the global allocator
#[global_allocator]
static ALLOCATOR: DebugAllocator = DebugAllocator;

// Initialize the heap
pub fn init_heap() -> Result<(), &'static str> {
    // Print memory information
    debug::print_memory_info(HEAP_START, HEAP_SIZE);
    serial_println!("Heap initialized");
    
    // Mark heap as initialized
    crate::mark_heap_initialized();
    
    Ok(())
}

// Utility function to align addresses
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
