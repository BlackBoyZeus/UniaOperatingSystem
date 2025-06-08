use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use core::sync::atomic::{AtomicUsize, Ordering};
use lazy_static::lazy_static;
use spin::Mutex;
use crate::println;

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
    system_name: String,
    user_name: String,
}

pub struct Stats {
    active_users: usize,
    projects: usize,
    server_uptime: f32,
    cpu_usage: usize,
    memory_usage: usize,
    network_speed: f32,
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
    Games,
    AI,
    Network,
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
                cpu_usage: 24,
                memory_usage: 1200, // MB
                network_speed: 5.4, // Mbps
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
                    name: "Games",
                    active: false,
                    action: Box::new(|| set_view(View::Games)),
                },
                NavItem {
                    name: "AI System",
                    active: false,
                    action: Box::new(|| set_view(View::AI)),
                },
                NavItem {
                    name: "Network",
                    active: false,
                    action: Box::new(|| set_view(View::Network)),
                },
                NavItem {
                    name: "Settings",
                    active: false,
                    action: Box::new(|| set_view(View::Settings)),
                },
            ],
            system_name: String::from("UNIA OS"),
            user_name: String::from("Player"),
        }
    }

    pub fn render(&self) {
        // Clear screen first
        for _ in 0..25 {
            println!("");
        }
        
        // Header
        println!("=== {} Dashboard ===", self.system_name);
        println!("Welcome, {}!", self.user_name);
        println!("View: {:?}", self.current_view);
        println!("");
        
        match self.current_view {
            View::Dashboard => self.render_dashboard(),
            View::Games => self.render_games(),
            View::AI => self.render_ai(),
            View::Network => self.render_network(),
            View::Settings => self.render_settings(),
        }
        
        // Footer with navigation
        println!("");
        println!("Navigation: ");
        for (i, item) in self.navigation.iter().enumerate() {
            if item.active {
                print!("[{}] ", item.name);
            } else {
                print!("{} ", item.name);
            }
            
            if i < self.navigation.len() - 1 {
                print!("| ");
            }
        }
        println!("");
        println!("Press key (1-5) to navigate, ESC for console");
    }
    
    fn render_dashboard(&self) {
        println!("System Stats:");
        println!("- CPU Usage: {}%", self.stats.cpu_usage);
        println!("- Memory: {} MB", self.stats.memory_usage);
        println!("- Network: {} Mbps", self.stats.network_speed);
        println!("");
        
        println!("Recent Activity:");
        for activity in &self.activities {
            println!("{} {}", activity.icon, activity.message);
        }
    }
    
    fn render_games(&self) {
        println!("Games Library:");
        println!("1. UNIA Demo Game");
        println!("   A demonstration of UNIA OS capabilities");
        println!("");
        println!("2. AI Sandbox");
        println!("   Experiment with AI-driven NPCs and procedural generation");
        println!("");
        println!("3. Mesh Network Test");
        println!("   Test the mesh networking capabilities with multiple players");
    }
    
    fn render_ai(&self) {
        println!("AI System Status:");
        println!("- NPC Behavior Trees: Active");
        println!("- Procedural Generation: Ready");
        println!("- Player Modeling: Learning");
        println!("");
        println!("AI Models:");
        println!("1. Basic NPC Behavior (loaded)");
        println!("2. Advanced Procedural Generation (loaded)");
        println!("3. Player Preference Prediction (loading: 78%)");
    }
    
    fn render_network(&self) {
        println!("Network Status:");
        println!("- Status: Online");
        println!("- Mesh Nodes: 3");
        println!("- Bandwidth: {} Mbps", self.stats.network_speed);
        println!("- Latency: 12ms");
        println!("");
        println!("Connected Peers:");
        println!("1. Local System (You)");
        println!("2. Test Node 1 (192.168.1.101)");
        println!("3. Test Node 2 (192.168.1.102)");
    }
    
    fn render_settings(&self) {
        println!("System Settings:");
        println!("1. Display");
        println!("   Resolution: 1920x1080");
        println!("   Refresh Rate: 60Hz");
        println!("");
        println!("2. Audio");
        println!("   Volume: 80%");
        println!("   Output: System Default");
        println!("");
        println!("3. Network");
        println!("   Mesh Networking: Enabled");
        println!("   Max Connections: 32");
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
        
        // Update dynamic stats (in a real implementation, these would come from actual measurements)
        self.stats.cpu_usage = (self.stats.cpu_usage + 1) % 100;
        self.stats.memory_usage = 1200 + (self.stats.cpu_usage * 10);
        self.stats.network_speed = 5.0 + (self.stats.cpu_usage as f32 / 100.0 * 2.0);
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
            (View::Games, "Games") => true,
            (View::AI, "AI System") => true,
            (View::Network, "Network") => true,
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

pub fn process_key(key: char) {
    match key {
        '1' => set_view(View::Dashboard),
        '2' => set_view(View::Games),
        '3' => set_view(View::AI),
        '4' => set_view(View::Network),
        '5' => set_view(View::Settings),
        '\x1B' => crate::console::init_console(), // ESC key for console
        _ => {} // Ignore other keys
    }
}
