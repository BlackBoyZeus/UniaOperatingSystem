#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{
    allocator,
    boot_sequence,
    memory::{self, BootInfoFrameAllocator},
    println,
    task::{executor::Executor, keyboard, Task},
    ui::dashboard::init_dashboard,
};
use x86_64::VirtAddr;

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Run the UNIA OS boot sequence
    boot_sequence::run_boot_sequence();
    
    println!("UNIA OS Bootable Experience");
    println!("---------------------------");
    println!("Initializing...");

    // Initialize memory management
    unia_os_bootable::init();
    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe { BootInfoFrameAllocator::init(&boot_info.memory_map) };
    
    // Initialize heap allocation
    allocator::init_heap(&mut mapper, &mut frame_allocator).expect("Heap initialization failed");

    // Initialize UI dashboard
    println!("Initializing UI dashboard...");
    init_dashboard();

    // Create async executor and tasks
    let mut executor = Executor::new();
    executor.spawn(Task::new(keyboard::print_keypresses()));
    executor.spawn(Task::new(unia_os_bootable::ai::run_ai_demo()));
    executor.spawn(Task::new(unia_os_bootable::network::run_network_demo()));
    executor.spawn(Task::new(unia_os_bootable::game::run_game_demo()));

    // Run the executor
    println!("UNIA OS initialized successfully!");
    println!("Press any key to interact with the dashboard...");
    executor.run();
}

/// This function is called on panic.
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("{}", info);
    unia_os_bootable::hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    unia_os_bootable::test_panic_handler(info)
}
