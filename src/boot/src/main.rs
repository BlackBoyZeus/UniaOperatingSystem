#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{
    allocator, boot_sequence, console, hlt_loop, memory, println,
    task::{executor::Executor, keyboard, Task}, ai, network, game
};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    println!("UNIA OS Kernel Starting...");

    // Initialize basic interrupts and GDT
    unia_os_bootable::init();

    // Initialize early allocator for boot sequence
    allocator::init_early_allocator();
    println!("Early allocator initialized");

    // Run the boot sequence animation
    boot_sequence::run_boot_sequence();

    // Initialize memory management
    let phys_mem_offset = boot_info.physical_memory_offset;
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };

    // Initialize the main heap allocator
    println!("Initializing heap...");
    allocator::init_heap(&mut mapper, &mut frame_allocator)
        .expect("Heap initialization failed");
    println!("Heap initialized successfully");

    // Initialize console after heap is ready
    console::init_console();

    // Create an executor for async tasks
    println!("Creating task executor...");
    let mut executor = Executor::new();

    // Spawn tasks
    executor.spawn(Task::new(keyboard::print_keypresses()));
    executor.spawn(Task::new(ai::run_ai_demo()));
    executor.spawn(Task::new(network::run_network_demo()));
    executor.spawn(Task::new(game::run_game_demo()));

    // Run the executor
    println!("UNIA OS Ready!");
    executor.run();

    // This should never be reached
    hlt_loop();
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("\n\nKERNEL PANIC: {}", info);
    println!("System halted.");
    hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    unia_os_bootable::test_panic_handler(info)
}
