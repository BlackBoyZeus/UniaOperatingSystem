use alloc::{collections::BTreeMap, sync::Arc};
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll};
use spin::Mutex;

mod executor;
mod scheduler;
mod task;

pub use executor::Executor;
pub use scheduler::Scheduler;
pub use task::{Task, TaskId, TaskState};

/// Modern task management system for UNIA OS
pub struct TaskManager {
    tasks: Mutex<BTreeMap<TaskId, Arc<Task>>>,
    executor: Mutex<Executor>,
    next_id: AtomicU64,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(BTreeMap::new()),
            executor: Mutex::new(Executor::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Spawn a new task
    pub fn spawn<F>(&self, future: F) -> TaskId 
    where
        F: Future<Output = ()> + 'static + Send,
    {
        let id = TaskId(self.next_id.fetch_add(1, Ordering::SeqCst));
        let task = Arc::new(Task::new(id, future));
        
        self.tasks.lock().insert(id, task.clone());
        self.executor.lock().spawn(task);
        
        id
    }

    /// Run the task scheduler
    pub fn run(&self) -> ! {
        self.executor.lock().run()
    }

    /// Get task by ID
    pub fn get_task(&self, id: TaskId) -> Option<Arc<Task>> {
        self.tasks.lock().get(&id).cloned()
    }

    /// Kill a task
    pub fn kill(&self, id: TaskId) {
        if let Some(task) = self.tasks.lock().remove(&id) {
            task.set_state(TaskState::Dead);
        }
    }

    /// Suspend a task
    pub fn suspend(&self, id: TaskId) {
        if let Some(task) = self.get_task(id) {
            task.set_state(TaskState::Suspended);
        }
    }

    /// Resume a task
    pub fn resume(&self, id: TaskId) {
        if let Some(task) = self.get_task(id) {
            task.set_state(TaskState::Running);
        }
    }

    /// Get number of running tasks
    pub fn running_tasks(&self) -> usize {
        self.tasks
            .lock()
            .values()
            .filter(|task| task.state() == TaskState::Running)
            .count()
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    RealTime = 3,
}

/// Async task wrapper
pub struct AsyncTask<F: Future> {
    future: Pin<Box<F>>,
    priority: Priority,
}

impl<F: Future> AsyncTask<F> {
    pub fn new(future: F, priority: Priority) -> Self {
        Self {
            future: Box::pin(future),
            priority,
        }
    }
}

impl<F: Future> Future for AsyncTask<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

/// Task statistics
#[derive(Debug, Default)]
pub struct TaskStats {
    pub cpu_time: u64,
    pub context_switches: u64,
    pub memory_usage: usize,
    pub io_operations: u64,
}

/// Task capabilities
#[derive(Debug, Clone)]
pub struct TaskCapabilities {
    pub can_allocate: bool,
    pub can_network: bool,
    pub can_filesystem: bool,
    pub max_memory: Option<usize>,
    pub max_threads: Option<usize>,
}
