use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use x86_64::{
    structures::paging::{
        mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
    },
    VirtAddr,
};

// Static buffer for early allocations (128KB should be enough for boot)
static mut EARLY_HEAP_BUFFER: [u8; 131072] = [0; 131072];
static EARLY_HEAP_NEXT: AtomicUsize = AtomicUsize::new(0);

// Heap configuration
pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 1024 * 1024; // 1 MiB

// Simple allocator that uses a static buffer
pub struct SimpleAllocator;

unsafe impl GlobalAlloc for SimpleAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Simple bump allocation from static buffer
        let align = layout.align();
        let size = layout.size();
        
        // Get current position and align it
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
                return EARLY_HEAP_BUFFER.as_mut_ptr().add(aligned_start);
            }
        }
        
        // Out of memory or CAS failed
        null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // No deallocation support in this simple allocator
    }
}

// Register our simple allocator as the global allocator
#[global_allocator]
static ALLOCATOR: SimpleAllocator = SimpleAllocator;

// Initialize the heap by mapping pages
pub fn init_heap(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError<Size4KiB>> {
    crate::println!("Mapping heap pages...");
    
    let page_range = {
        let heap_start = VirtAddr::new(HEAP_START as u64);
        let heap_end = heap_start + HEAP_SIZE - 1u64;
        let heap_start_page = Page::containing_address(heap_start);
        let heap_end_page = Page::containing_address(heap_end);
        Page::range_inclusive(heap_start_page, heap_end_page)
    };

    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe { mapper.map_to(page, frame, flags, frame_allocator)?.flush() };
    }

    crate::println!("Heap pages mapped successfully");
    
    // Mark heap as initialized
    crate::mark_heap_initialized();
    crate::println!("Heap initialization complete");

    Ok(())
}

// Utility function to align addresses
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
