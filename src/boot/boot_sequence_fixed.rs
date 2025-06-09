use bootloader::BootInfo;
use x86_64::{
    VirtAddr,
    structures::paging::{PageTableFlags, Page, PhysFrame, Size4KiB},
};
use crate::{memory_fixed as memory, serial_println};

pub fn initialize_memory(boot_info: &'static BootInfo) -> Result<(), &'static str> {
    // Calculate physical memory offset
    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    serial_println!("Physical memory offset: {:?}", phys_mem_offset);

    // Initialize page table
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    
    // Initialize frame allocator
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };

    // Map the first 2MB of memory for kernel use
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
    
    for i in 0..512 {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(i * 4096));
        match memory::map_page(page, flags, &mut mapper, &mut frame_allocator) {
            Ok(_) => {},
            Err(e) => {
                serial_println!("Failed to map page at {:?}: {}", page, e);
                return Err("Memory mapping failed");
            }
        }
    }

    serial_println!("Memory initialization complete");
    serial_println!("Total frames available: {}", frame_allocator.total_frames());
    serial_println!("Frames allocated: {}", frame_allocator.allocated_frame_count());

    Ok(())
}
