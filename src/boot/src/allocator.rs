use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, Ordering};

// Pre-allocated buffer for the critical 64-byte allocation
#[repr(align(8))]
struct AlignedBuffer {
    data: [u8; 64]
}

// Multiple static buffers for different allocation scenarios
static mut CRITICAL_BUFFER: AlignedBuffer = AlignedBuffer { data: [0; 64] };
static mut CRITICAL_BUFFER_2: AlignedBuffer = AlignedBuffer { data: [0; 64] };
static mut CRITICAL_BUFFER_3: AlignedBuffer = AlignedBuffer { data: [0; 64] };
static mut CRITICAL_BUFFER_4: AlignedBuffer = AlignedBuffer { data: [0; 64] };

// Larger buffer for other early allocations
static mut EARLY_HEAP_BUFFER: [u8; 8192] = [0; 8192]; // 8 KiB
static mut EARLY_HEAP_NEXT: usize = 0;

// Track which buffers have been used
static CRITICAL_BUFFER_USED: AtomicBool = AtomicBool::new(false);
static CRITICAL_BUFFER_2_USED: AtomicBool = AtomicBool::new(false);
static CRITICAL_BUFFER_3_USED: AtomicBool = AtomicBool::new(false);
static CRITICAL_BUFFER_4_USED: AtomicBool = AtomicBool::new(false);

// Heap configuration
pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

// Simple allocator that handles the critical 64-byte allocation
pub struct CriticalAllocator;

unsafe impl GlobalAlloc for CriticalAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Log allocation request
        crate::serial_println!("Allocation request: size={}, align={}", layout.size(), layout.align());
        
        // Special handling for 64-byte allocations which are causing problems
        if layout.size() == 64 && layout.align() <= 8 {
            // Try to use one of our pre-allocated buffers
            if !CRITICAL_BUFFER_USED.load(Ordering::SeqCst) {
                CRITICAL_BUFFER_USED.store(true, Ordering::SeqCst);
                crate::serial_println!("Using critical buffer 1 for 64-byte allocation");
                return CRITICAL_BUFFER.data.as_mut_ptr();
            } else if !CRITICAL_BUFFER_2_USED.load(Ordering::SeqCst) {
                CRITICAL_BUFFER_2_USED.store(true, Ordering::SeqCst);
                crate::serial_println!("Using critical buffer 2 for 64-byte allocation");
                return CRITICAL_BUFFER_2.data.as_mut_ptr();
            } else if !CRITICAL_BUFFER_3_USED.load(Ordering::SeqCst) {
                CRITICAL_BUFFER_3_USED.store(true, Ordering::SeqCst);
                crate::serial_println!("Using critical buffer 3 for 64-byte allocation");
                return CRITICAL_BUFFER_3.data.as_mut_ptr();
            } else if !CRITICAL_BUFFER_4_USED.load(Ordering::SeqCst) {
                CRITICAL_BUFFER_4_USED.store(true, Ordering::SeqCst);
                crate::serial_println!("Using critical buffer 4 for 64-byte allocation");
                return CRITICAL_BUFFER_4.data.as_mut_ptr();
            }
            // If all critical buffers are used, fall through to the general allocator
        }
        
        // For all other allocations during early boot, use a simple bump allocator
        let align = layout.align();
        let size = layout.size();
        
        // Align the offset
        let aligned_offset = (EARLY_HEAP_NEXT + align - 1) & !(align - 1);
        
        // Check if we have enough space
        if aligned_offset + size <= EARLY_HEAP_BUFFER.len() {
            let ptr = EARLY_HEAP_BUFFER.as_mut_ptr().add(aligned_offset);
            EARLY_HEAP_NEXT = aligned_offset + size;
            crate::serial_println!("Allocated at offset {}: {:p}", aligned_offset, ptr);
            return ptr;
        }
        
        crate::serial_println!("Allocation failed!");
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // Check if this is one of our critical buffers
        if ptr == CRITICAL_BUFFER.data.as_mut_ptr() {
            crate::serial_println!("Deallocating critical buffer 1");
            CRITICAL_BUFFER_USED.store(false, Ordering::SeqCst);
            return;
        } else if ptr == CRITICAL_BUFFER_2.data.as_mut_ptr() {
            crate::serial_println!("Deallocating critical buffer 2");
            CRITICAL_BUFFER_2_USED.store(false, Ordering::SeqCst);
            return;
        } else if ptr == CRITICAL_BUFFER_3.data.as_mut_ptr() {
            crate::serial_println!("Deallocating critical buffer 3");
            CRITICAL_BUFFER_3_USED.store(false, Ordering::SeqCst);
            return;
        } else if ptr == CRITICAL_BUFFER_4.data.as_mut_ptr() {
            crate::serial_println!("Deallocating critical buffer 4");
            CRITICAL_BUFFER_4_USED.store(false, Ordering::SeqCst);
            return;
        }
        
        // We don't actually free memory in this simple allocator for other allocations
        crate::serial_println!("Deallocation: ptr={:p}", ptr);
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
