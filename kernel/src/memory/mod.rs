use bootloader::bootinfo::{MemoryMap, MemoryRegionType};
use x86_64::{
    structures::paging::{
        FrameAllocator, Mapper, OffsetPageTable, Page, PageTable,
        PhysFrame, Size4KiB, PageTableFlags,
    },
    PhysAddr, VirtAddr,
};
use spin::Mutex;

mod allocator;
mod frame;
mod heap;
mod page;
mod vmem;

pub use allocator::KernelAllocator;
pub use frame::FrameTracker;
pub use heap::HeapAllocator;
pub use page::PageTracker;
pub use vmem::VirtualMemoryManager;

/// Modern memory management system for UNIA OS
pub struct MemoryManager {
    page_table: Mutex<OffsetPageTable<'static>>,
    frame_allocator: Mutex<frame::BitmapFrameAllocator>,
    heap_allocator: Mutex<heap::HeapAllocator>,
    vmem_manager: Mutex<vmem::VirtualMemoryManager>,
}

impl MemoryManager {
    pub fn new(boot_info: &'static bootloader::BootInfo) -> Self {
        let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
        let page_table = unsafe { Self::init_page_table(phys_mem_offset) };
        let frame_allocator = frame::BitmapFrameAllocator::new(&boot_info.memory_map);
        let heap_allocator = heap::HeapAllocator::new();
        let vmem_manager = vmem::VirtualMemoryManager::new();

        Self {
            page_table: Mutex::new(page_table),
            frame_allocator: Mutex::new(frame_allocator),
            heap_allocator: Mutex::new(heap_allocator),
            vmem_manager: Mutex::new(vmem_manager),
        }
    }

    pub fn init(&mut self) {
        // Initialize heap
        self.init_heap();

        // Set up kernel address space
        self.setup_kernel_space();

        // Initialize virtual memory manager
        self.vmem_manager.lock().init();
    }

    /// Initialize the kernel heap
    fn init_heap(&mut self) {
        use x86_64::structures::paging::mapper::MapToError;

        const HEAP_START: usize = 0x_4444_4444_0000;
        const HEAP_SIZE: usize = 100 * 1024 * 1024; // 100 MiB

        let heap_start = VirtAddr::new(HEAP_START as u64);
        let heap_end = heap_start + HEAP_SIZE - 1u64;
        let heap_pages = {
            let heap_start_page = Page::containing_address(heap_start);
            let heap_end_page = Page::containing_address(heap_end);
            Page::range_inclusive(heap_start_page, heap_end_page)
        };

        // Map heap pages
        for page in heap_pages {
            let frame = self.frame_allocator
                .lock()
                .allocate_frame()
                .expect("Failed to allocate heap frame");

            let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
            unsafe {
                self.page_table
                    .lock()
                    .map_to(page, frame, flags, &mut *self.frame_allocator.lock())
                    .expect("Failed to map heap page")
                    .flush();
            }
        }

        // Initialize heap allocator
        unsafe {
            self.heap_allocator
                .lock()
                .init(HEAP_START, HEAP_SIZE);
        }
    }

    /// Set up kernel address space
    fn setup_kernel_space(&mut self) {
        // Identity map the first 1MB for BIOS
        for i in 0..256 {
            let frame = PhysFrame::containing_address(PhysAddr::new(i * 4096));
            let page = Page::containing_address(VirtAddr::new(i * 4096));
            let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

            unsafe {
                self.page_table
                    .lock()
                    .map_to(page, frame, flags, &mut *self.frame_allocator.lock())
                    .expect("Failed to identity map low memory")
                    .flush();
            }
        }
    }

    /// Initialize page table
    unsafe fn init_page_table(phys_mem_offset: VirtAddr) -> OffsetPageTable<'static> {
        let level_4_table = active_level_4_table(phys_mem_offset);
        OffsetPageTable::new(level_4_table, phys_mem_offset)
    }
}

/// Returns a mutable reference to the active level 4 page table
unsafe fn active_level_4_table(phys_mem_offset: VirtAddr) -> &'static mut PageTable {
    use x86_64::registers::control::Cr3;

    let (level_4_table_frame, _) = Cr3::read();
    let phys = level_4_table_frame.start_address();
    let virt = phys_mem_offset + phys.as_u64();
    let page_table_ptr: *mut PageTable = virt.as_mut_ptr();

    &mut *page_table_ptr
}
