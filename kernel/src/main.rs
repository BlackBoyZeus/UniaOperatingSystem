#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![feature(abi_x86_interrupt)]
#![feature(const_mut_refs)]
#![feature(async_fn_in_trait)]
#![test_runner(crate::test::test_runner)]
#![reexport_test_harness_main = "test_main"]

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;

mod arch;
mod drivers;
mod fs;
mod graphics;
mod memory;
mod net;
mod security;
mod sync;
mod task;
mod time;
mod virt;

use crate::arch::x86_64::{gdt, idt, paging};
use crate::memory::MemoryManager;
use crate::task::scheduler::Scheduler;

// Global kernel state
pub static KERNEL: spin::Once<KernelState> = spin::Once::new();

pub struct KernelState {
    memory_manager: MemoryManager,
    scheduler: Scheduler,
    network_stack: net::NetworkStack,
    security_manager: security::SecurityManager,
    fs_manager: fs::FileSystemManager,
    device_manager: drivers::DeviceManager,
}

impl KernelState {
    fn new(boot_info: &'static BootInfo) -> Self {
        let memory_manager = MemoryManager::new(boot_info);
        let scheduler = Scheduler::new();
        let network_stack = net::NetworkStack::new();
        let security_manager = security::SecurityManager::new();
        let fs_manager = fs::FileSystemManager::new();
        let device_manager = drivers::DeviceManager::new();

        Self {
            memory_manager,
            scheduler,
            network_stack,
            security_manager,
            fs_manager,
            device_manager,
        }
    }

    fn init(&mut self) {
        // Initialize architecture-specific features
        gdt::init();
        idt::init();
        paging::init();

        // Initialize core kernel subsystems
        self.memory_manager.init();
        self.scheduler.init();
        self.network_stack.init();
        self.security_manager.init();
        self.fs_manager.init();
        self.device_manager.init();
    }
}

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize kernel state
    let mut kernel = KernelState::new(boot_info);
    kernel.init();
    KERNEL.call_once(|| kernel);

    // Display boot message
    graphics::display::clear_screen();
    graphics::display::print_at(
        "UNIA OS v2.0 - Modern Microkernel",
        0,
        0,
        graphics::Color::Green,
    );

    // Start scheduler
    KERNEL.get().unwrap().scheduler.start();

    // Main kernel loop
    loop {
        x86_64::instructions::hlt();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    graphics::display::print_at(
        "KERNEL PANIC!",
        0,
        2,
        graphics::Color::Red,
    );
    if let Some(location) = info.location() {
        graphics::display::print_at(
            &format!("at {}:{}", location.file(), location.line()),
            0,
            3,
            graphics::Color::White,
        );
    }
    loop {
        x86_64::instructions::hlt();
    }
}
