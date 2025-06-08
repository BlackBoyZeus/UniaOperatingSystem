use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use core::sync::atomic::{AtomicUsize, Ordering};
use lazy_static::lazy_static;
use spin::Mutex;

// Dashboard state
lazy_static! {
    static ref DASHBOARD: Mutex<Dashboard> = Mutex::new(Dashboard::new());
}

static ACTIVE_USERS: AtomicUsize = AtomicUsize::new(1024);
static PROJECTS: AtomicUsize = AtomicUsize::new(57);
static SERVER_UPTIME: AtomicUsize = AtomicUsize::new(9998); // 99.98%

pub struct Dashboard {
    current_view: View,
    stats: Stats,
    activities: Vec<Activity>,
    navigation: Vec<NavItem>,
}

pub struct Stats {
    active_users: usize,
    projects: usize,
    server_uptime: f32,
}

pub struct Activity {
    icon: &'static str,
    message: String,
    timestamp: u64,
}

pub struct NavItem {
    name: &'static str,
    active: bool,
    action: Box<dyn Fn() + Send>,
}

#[derive(PartialEq)]
pub enum View {
    Dashboard,
    Projects,
    Analytics,
    Settings,
}

impl Dashboard {
    pub fn new() -> Self {
        Dashboard {
            current_view: View::Dashboard,
            stats: Stats {
                active_users: ACTIVE_USERS.load(Ordering::Relaxed),
                projects: PROJECTS.load(Ordering::Relaxed),
                server_uptime: SERVER_UPTIME.load(Ordering::Relaxed) as f32 / 100.0,
            },
            activities: vec![
                Activity {
                    icon: "🔄",
                    message: String::from("System initialization complete"),
                    timestamp: 1623456789,
                },
                Activity {
                    icon: "📈",
                    message: String::from("AI subsystem loaded"),
                    timestamp: 1623456790,
                },
                Activity {
                    icon: "🛠️",
                    message: String::from("Mesh networking initialized"),
                    timestamp: 1623456791,
                },
            ],
            navigation: vec![
                NavItem {
                    name: "Dashboard",
                    active: true,
                    action: Box::new(|| set_view(View::Dashboard)),
                },
                NavItem {
                    name: "Projects",
                    active: false,
                    action: Box::new(|| set_view(View::Projects)),
                },
                NavItem {
                    name: "Analytics",
                    active: false,
                    action: Box::new(|| set_view(View::Analytics)),
                },
                NavItem {
                    name: "Settings",
                    active: false,
                    action: Box::new(|| set_view(View::Settings)),
                },
            ],
        }
    }

    pub fn render(&self) {
        // In a real implementation, this would render to a framebuffer or other display mechanism
        crate::println!("=== UNIA OS Dashboard ===");
        crate::println!("View: {:?}", self.current_view);
        crate::println!("");
        
        crate::println!("Stats:");
        crate::println!("- Active Users: {}", self.stats.active_users);
        crate::println!("- Projects: {}", self.stats.projects);
        crate::println!("- System Uptime: {}%", self.stats.server_uptime);
        crate::println!("");
        
        crate::println!("Recent Activity:");
        for activity in &self.activities {
            crate::println!("{} {}", activity.icon, activity.message);
        }
    }

    pub fn add_activity(&mut self, icon: &'static str, message: String) {
        self.activities.insert(0, Activity {
            icon,
            message,
            timestamp: 1623456792, // In a real implementation, this would be the current timestamp
        });
        
        // Keep only the most recent activities
        if self.activities.len() > 10 {
            self.activities.pop();
        }
    }

    pub fn update_stats(&mut self) {
        self.stats.active_users = ACTIVE_USERS.load(Ordering::Relaxed);
        self.stats.projects = PROJECTS.load(Ordering::Relaxed);
        self.stats.server_uptime = SERVER_UPTIME.load(Ordering::Relaxed) as f32 / 100.0;
    }
}

pub fn init_dashboard() {
    let mut dashboard = DASHBOARD.lock();
    dashboard.render();
}

pub fn set_view(view: View) {
    let mut dashboard = DASHBOARD.lock();
    dashboard.current_view = view;
    
    // Update navigation active states
    for item in &mut dashboard.navigation {
        item.active = match (&view, item.name) {
            (View::Dashboard, "Dashboard") => true,
            (View::Projects, "Projects") => true,
            (View::Analytics, "Analytics") => true,
            (View::Settings, "Settings") => true,
            _ => false,
        };
    }
    
    dashboard.render();
}

pub fn add_activity(icon: &'static str, message: String) {
    let mut dashboard = DASHBOARD.lock();
    dashboard.add_activity(icon, message);
    dashboard.render();
}

pub fn update_stats() {
    let mut dashboard = DASHBOARD.lock();
    dashboard.update_stats();
    dashboard.render();
}
