use crate::{println, ui, serial_println};
use conquer_once::spin::OnceCell;
use core::{
    pin::Pin,
    task::{Context, Poll},
};
use crossbeam_queue::ArrayQueue;
use futures_util::{
    stream::{Stream, StreamExt},
    task::AtomicWaker,
};
use pc_keyboard::{layouts, DecodedKey, HandleControl, Keyboard, ScancodeSet1};

static SCANCODE_QUEUE: OnceCell<ArrayQueue<u8>> = OnceCell::uninit();
static WAKER: AtomicWaker = AtomicWaker::new();

/// Called by the keyboard interrupt handler
///
/// Must not block or allocate.
pub(crate) fn add_scancode(scancode: u8) {
    if let Ok(queue) = SCANCODE_QUEUE.try_get() {
        if let Err(_) = queue.push(scancode) {
            serial_println!("WARNING: scancode queue full; dropping keyboard input");
        } else {
            WAKER.wake();
        }
    } else {
        serial_println!("WARNING: scancode queue uninitialized");
    }
}

pub struct ScancodeStream {
    _private: (),
}

impl ScancodeStream {
    pub fn new() -> Self {
        SCANCODE_QUEUE
            .try_init_once(|| ArrayQueue::new(100))
            .expect("ScancodeStream::new should only be called once");
        ScancodeStream { _private: () }
    }
}

impl Stream for ScancodeStream {
    type Item = u8;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<u8>> {
        let queue = SCANCODE_QUEUE
            .try_get()
            .expect("scancode queue not initialized");

        // fast path
        if let Some(scancode) = queue.pop() {
            return Poll::Ready(Some(scancode));
        }

        WAKER.register(&cx.waker());
        match queue.pop() {
            Some(scancode) => {
                WAKER.take();
                Poll::Ready(Some(scancode))
            }
            None => Poll::Pending,
        }
    }
}

pub async fn print_keypresses() {
    let mut scancodes = ScancodeStream::new();
    let mut keyboard = Keyboard::new(ScancodeSet1::new(), layouts::Us104Key, HandleControl::Ignore);

    while let Some(scancode) = scancodes.next().await {
        if let Ok(Some(key_event)) = keyboard.add_byte(scancode) {
            if let Some(key) = keyboard.process_keyevent(key_event) {
                match key {
                    DecodedKey::Unicode(character) => {
                        // Process the key in the UI dashboard
                        ui::dashboard::process_key(character);
                    }
                    DecodedKey::RawKey(key) => {
                        // Handle special keys if needed
                        println!("Special key: {:?}", key);
                    }
                }
            }
        }
    }
}

// Simple keyboard handler for direct polling
pub struct SimpleKeyboard {
    keyboard: Keyboard<layouts::Us104Key, ScancodeSet1>,
}

impl SimpleKeyboard {
    pub fn new() -> Self {
        Self {
            keyboard: Keyboard::new(ScancodeSet1::new(), layouts::Us104Key, HandleControl::Ignore),
        }
    }
    
    pub fn process_next_scancode(&mut self) -> Option<DecodedKey> {
        let queue = match SCANCODE_QUEUE.try_get() {
            Ok(q) => q,
            Err(_) => return None,
        };
        
        let scancode = match queue.pop() {
            Some(s) => s,
            None => return None,
        };
        
        if let Ok(Some(key_event)) = self.keyboard.add_byte(scancode) {
            return self.keyboard.process_keyevent(key_event);
        }
        
        None
    }
}

// Initialize the keyboard handler
pub fn init_keyboard() -> SimpleKeyboard {
    // Initialize the scancode queue if not already done
    let _ = SCANCODE_QUEUE.try_init_once(|| ArrayQueue::new(100));
    
    SimpleKeyboard::new()
}
