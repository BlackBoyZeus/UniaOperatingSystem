use bootloader::bootinfo::{MemoryMap, MemoryRegionType};
use x86_64::{
    structures::paging::{
        FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PhysFrame, Size4KiB,
        PageTableFlags, MappedPageTable,
    },
    PhysAddr, VirtAddr, registers::control::Cr3,
};

/// Initialize memory management
/// Returns (OffsetPageTable, BootInfoFrameAllocator)
pub fn init_memory(
    physical_memory_offset: VirtAddr,
    memory_map: &'static MemoryMap
) -> Result<(OffsetPageTable<'static>, BootInfoFrameAllocator), &'static str> {
    // Validate physical memory offset
    if physical_memory_offset.as_u64() == 0 {
        return Err("Invalid physical memory offset");
    }

    // Get the current page table
    let (level_4_table_frame, _) = Cr3::read();
    let phys = level_4_table_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    
    // Create page table
    let page_table = unsafe {
        let page_table_ptr: *mut PageTable = virt.as_mut_ptr();
        &mut *page_table_ptr
    };

    // Create offset page table
    let mut mapper = unsafe {
        OffsetPageTable::new(page_table, physical_memory_offset)
    };

    // Create frame allocator
    let mut frame_allocator = unsafe {
        BootInfoFrameAllocator::init(memory_map)
    };

    // Identity map the first 1MB
    for i in 0..256 {
        let page = Page::containing_address(VirtAddr::new(i * Size4KiB::SIZE));
        let frame = PhysFrame::containing_address(PhysAddr::new(i * Size4KiB::SIZE));
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

        unsafe {
            mapper.map_to(page, frame, flags, &mut frame_allocator)
                .map_err(|_| "Failed to identity map low memory")?
                .flush();
        }
    }

    Ok((mapper, frame_allocator))
}

pub struct BootInfoFrameAllocator {
    memory_map: &'static MemoryMap,
    next: usize,
}

impl BootInfoFrameAllocator {
    /// Create a FrameAllocator from the passed memory map.
    ///
    /// # Safety
    /// The caller must guarantee that the passed memory map is valid
    /// and that the frames marked as `USABLE` are actually unused.
    pub unsafe fn init(memory_map: &'static MemoryMap) -> Self {
        BootInfoFrameAllocator {
            memory_map,
            next: 0,
        }
    }

    /// Returns an iterator over the usable frames specified in the memory map.
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> {
        // Get usable regions from memory map
        let regions = self.memory_map.iter();
        let usable_regions = regions.filter(|r| r.region_type == MemoryRegionType::Usable);
        
        // Map each region to its address range
        let addr_ranges = usable_regions.map(|r| r.range.start_addr()..r.range.end_addr());
        
        // Transform to an iterator of frame start addresses
        let frame_addresses = addr_ranges.flat_map(|r| r.step_by(4096));
        
        // Create `PhysFrame` types from the start addresses
        frame_addresses.map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let frame = self.usable_frames().nth(self.next);
        self.next += 1;
        frame
    }
}
