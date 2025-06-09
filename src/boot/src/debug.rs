use core::sync::atomic::{AtomicUsize, Ordering};
use crate::serial_println;

// Allocation tracking counters
pub static ALLOCATION_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

// Special buffer for the problematic 64-byte allocation
pub static mut SPECIAL_BUFFER_64: [u8; 64] = [0; 64];
pub static mut SPECIAL_BUFFER_USED: bool = false;

// Print information about the current execution context
pub fn print_debug_info(message: &str) {
    serial_println!("DEBUG: {}", message);
}

// Track an allocation attempt
pub fn track_allocation(size: usize, align: usize) {
    let count = ALLOCATION_COUNTER.fetch_add(1, Ordering::SeqCst);
    let total_bytes = ALLOCATION_BYTES.fetch_add(size, Ordering::SeqCst);
    
    serial_println!("ALLOC #{}: size={}, align={}, total_bytes={}", 
                   count, size, align, total_bytes);
    
    // Special handling for the problematic 64-byte allocation
    if size == 64 && align == 8 {
        serial_println!("*** CRITICAL: 64-byte allocation detected! ***");
    }
}

// Print register state for debugging
pub fn print_register_state() {
    serial_println!("=== REGISTER STATE ===");
    
    let rax: u64;
    let rbx: u64;
    let rcx: u64;
    let rdx: u64;
    let rsp: u64;
    let rbp: u64;
    
    unsafe {
        core::arch::asm!("mov {}, rax", out(reg) rax);
        core::arch::asm!("mov {}, rbx", out(reg) rbx);
        core::arch::asm!("mov {}, rcx", out(reg) rcx);
        core::arch::asm!("mov {}, rdx", out(reg) rdx);
        core::arch::asm!("mov {}, rsp", out(reg) rsp);
        core::arch::asm!("mov {}, rbp", out(reg) rbp);
    }
    
    serial_println!("RAX: 0x{:016x}  RBX: 0x{:016x}", rax, rbx);
    serial_println!("RCX: 0x{:016x}  RDX: 0x{:016x}", rcx, rdx);
    serial_println!("RSP: 0x{:016x}  RBP: 0x{:016x}", rsp, rbp);
}

// Print memory layout information
pub fn print_memory_info(heap_start: usize, heap_size: usize) {
    serial_println!("=== MEMORY INFO ===");
    serial_println!("Heap start: 0x{:x}", heap_start);
    serial_println!("Heap size: {} bytes", heap_size);
    serial_println!("Heap end: 0x{:x}", heap_start + heap_size);
}
