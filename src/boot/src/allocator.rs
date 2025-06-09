use linked_list_allocator::LockedHeap;
use x86_64::{
    structures::paging::{
        mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
    },
    VirtAddr,
};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::alloc::{GlobalAlloc, Layout};

pub const HEAP_START: usize = 0x_4444_4444_0000;
pub const HEAP_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

// Safe wrapper around the allocator
pub struct SafeAllocator {
    heap: LockedHeap,
    initialized: AtomicBool,
}

impl SafeAllocator {
    pub const fn new() -> Self {
        Self {
            heap: LockedHeap::empty(),
            initialized: AtomicBool::new(false),
        }
    }
    
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
    
    pub unsafe fn init(&self, heap_start: *mut u8, heap_size: usize) {
        self.heap.lock().init(heap_start, heap_size);
        self.initialized.store(true, Ordering::Release);
    }
}

unsafe impl GlobalAlloc for SafeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.is_initialized() {
            // Use emergency allocation for early boot
            return emergency_alloc(layout.size(), layout.align());
        }
        
        self.heap.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !self.is_initialized() {
            // Handle emergency deallocation
            emergency_dealloc(ptr);
            return;
        }
        
        self.heap.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: SafeAllocator = SafeAllocator::new();

// Emergency allocation system for early boot
const EMERGENCY_HEAP_SIZE: usize = 32 * 1024; // 32 KiB
static mut EMERGENCY_HEAP: [u8; EMERGENCY_HEAP_SIZE] = [0; EMERGENCY_HEAP_SIZE];
static EMERGENCY_OFFSET: AtomicUsize = AtomicUsize::new(0);

struct EmergencyAllocation {
    ptr: *mut u8,
    size: usize,
    in_use: AtomicBool,
}

impl EmergencyAllocation {
    const fn new() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            size: 0,
            in_use: AtomicBool::new(false),
        }
    }
}

// Track emergency allocations for proper cleanup
static mut EMERGENCY_ALLOCATIONS: [EmergencyAllocation; 32] = [
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
    EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(), EmergencyAllocation::new(),
];

fn emergency_alloc(size: usize, align: usize) -> *mut u8 {
    crate::serial_println!("Emergency allocation: size={}, align={}", size, align);
    
    let aligned_size = (size + align - 1) & !(align - 1);
    let current_offset = EMERGENCY_OFFSET.load(Ordering::Acquire);
    let aligned_offset = (current_offset + align - 1) & !(align - 1);
    
    if aligned_offset + aligned_size > EMERGENCY_HEAP_SIZE {
        crate::serial_println!("Emergency allocation failed: out of memory");
        return core::ptr::null_mut();
    }
    
    // Try to update the offset
    if EMERGENCY_OFFSET.compare_exchange(
        current_offset,
        aligned_offset + aligned_size,
        Ordering::AcqRel,
        Ordering::Relaxed
    ).is_err() {
        // Another thread updated the offset, try again
        return emergency_alloc(size, align);
    }
    
    unsafe {
        let ptr = EMERGENCY_HEAP.as_mut_ptr().add(aligned_offset);
        
        // Record this allocation
        for allocation in &mut EMERGENCY_ALLOCATIONS {
            if !allocation.in_use.load(Ordering::Acquire) {
                if allocation.in_use.compare_exchange(
                    false, true, Ordering::AcqRel, Ordering::Relaxed
                ).is_ok() {
                    allocation.ptr = ptr;
                    allocation.size = aligned_size;
                    break;
                }
            }
        }
        
        // Zero the memory
        core::ptr::write_bytes(ptr, 0, aligned_size);
        crate::serial_println!("Emergency allocation successful: ptr={:p}", ptr);
        ptr
    }
}

fn emergency_dealloc(ptr: *mut u8) {
    unsafe {
        for allocation in &mut EMERGENCY_ALLOCATIONS {
            if allocation.ptr == ptr && allocation.in_use.load(Ordering::Acquire) {
                allocation.in_use.store(false, Ordering::Release);
                allocation.ptr = core::ptr::null_mut();
                allocation.size = 0;
                crate::serial_println!("Emergency deallocation: ptr={:p}", ptr);
                break;
            }
        }
    }
}

pub fn init_heap(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError<Size4KiB>> {
    crate::serial_println!("Mapping heap pages...");
    
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

    unsafe {
        ALLOCATOR.init(HEAP_START as *mut u8, HEAP_SIZE);
    }
    
    crate::serial_println!("Heap initialized at 0x{:x} with size {} bytes", HEAP_START, HEAP_SIZE);
    crate::mark_heap_initialized();
    
    Ok(())
}
