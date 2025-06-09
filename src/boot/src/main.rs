#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{
    allocator, boot_sequence, console, hlt_loop, memory, println, task::{executor::Executor, keyboard, Task}, ai, network, game
};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize the OS components first
    unia_os_bootable::init();
    
    // Initialize memory management immediately
    let phys_mem_offset = boot_info.physical_memory_offset;
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };
    
    // Now it's safe to print
    println!("UNIA OS Kernel Starting...");
    println!("Memory management initialized");
    
    // Initialize the heap
    println!("Initializing heap...");
    match allocator::init_heap(&mut mapper, &mut frame_allocator) {
        Ok(_) => println!("Heap initialization successful"),
        Err(e) => {
            println!("Heap initialization failed: {:?}", e);
            panic!("Failed to initialize heap");
        }
    }
    
    // Run the boot sequence animation
    println!("Starting boot sequence...");
    boot_sequence::run_boot_sequence();
    println!("Boot sequence completed");
    
    // Initialize console after heap is ready
    println!("Initializing console...");
    console::init_console();
    println!("Console initialized");
    
    // Create an executor for async tasks
    println!("Creating task executor...");
    let mut executor = Executor::new();
    println!("Task executor created");
    
    // Spawn keyboard task
    println!("Spawning keyboard task...");
    executor.spawn(Task::new(keyboard::print_keypresses()));
    
    // Spawn demo tasks
    println!("Spawning demo tasks...");
    executor.spawn(Task::new(ai::run_ai_demo()));
    executor.spawn(Task::new(network::run_network_demo()));
    executor.spawn(Task::new(game::run_game_demo()));
    
    // Run the executor
    println!("UNIA OS Ready! Starting task executor...");
    executor.run();
    
    // This should never be reached
    hlt_loop();
}

/// This function is called on panic.
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("\n\nKERNEL PANIC: {}", info);
    
    // Print additional debug information
    println!("System halted.");
    
    hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    unia_os_bootable::test_panic_handler(info)
}
