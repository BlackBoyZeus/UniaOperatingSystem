use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use linked_list_allocator::LockedHeap;
use spin::Mutex;
use x86_64::{
    structures::paging::{mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB},
    VirtAddr,
};

// Increase heap size to 1 MiB
pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 1024 * 1024; // 1 MiB

// Bump allocator for early boot
pub struct BumpAllocator {
    heap_start: usize,
    heap_end: usize,
    next: usize,
    allocations: usize,
}

impl BumpAllocator {
    pub const fn new() -> Self {
        BumpAllocator {
            heap_start: 0,
            heap_end: 0,
            next: 0,
            allocations: 0,
        }
    }

    pub unsafe fn init(&mut self, heap_start: usize, heap_size: usize) {
        self.heap_start = heap_start;
        self.heap_end = heap_start + heap_size;
        self.next = heap_start;
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut next = self.next;
        
        // Align the allocation address
        let alloc_start = align_up(next, layout.align());
        let alloc_end = match alloc_start.checked_add(layout.size()) {
            Some(end) => end,
            None => return null_mut(), // Overflow
        };

        if alloc_end > self.heap_end {
            // Out of memory
            null_mut()
        } else {
            // Update the next allocation position
            next = alloc_end;
            self.next = next;
            self.allocations += 1;
            alloc_start as *mut u8
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // No deallocation support in bump allocator
    }
}

// Early boot allocator
static BUMP_ALLOCATOR: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::new());

// Main allocator for after heap initialization
#[global_allocator]
static ALLOCATOR: BumpAllocator = BumpAllocator::new();

// Initialize the early boot allocator
pub fn init_early_allocator() {
    unsafe {
        // Initialize the global allocator directly
        let mut allocator = ALLOCATOR;
        allocator.init(HEAP_START, HEAP_SIZE);
    }
    crate::println!("Early allocator initialized at 0x{:x} with size {} bytes", HEAP_START, HEAP_SIZE);
}

// Initialize the main heap allocator
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

// Locked wrapper (unchanged)
pub struct Locked<A> {
    inner: spin::Mutex<A>,
}

impl<A> Locked<A> {
    pub const fn new(inner: A) -> Self {
        Locked {
            inner: spin::Mutex::new(inner),
        }
    }

    pub fn lock(&self) -> spin::MutexGuard<A> {
        self.inner.lock()
    }
}

// Utility function to align addresses
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

// Locked wrapper (unchanged)
pub struct Locked<A> {
    inner: spin::Mutex<A>,
}

impl<A> Locked<A> {
    pub const fn new(inner: A) -> Self {
        Locked {
            inner: spin::Mutex::new(inner),
        }
    }

    pub fn lock(&self) -> spin::MutexGuard<A> {
        self.inner.lock()
    }
}
