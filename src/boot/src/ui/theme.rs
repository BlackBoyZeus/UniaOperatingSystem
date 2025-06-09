// UNIA OS UI Theme

// Color palette
pub const COLOR_BG_DARK: u32 = 0x111827; // bg-gray-900
pub const COLOR_BG_MEDIUM: u32 = 0x1F2937; // bg-gray-800
pub const COLOR_BG_LIGHT: u32 = 0x374151; // bg-gray-700
pub const COLOR_TEXT_PRIMARY: u32 = 0xFFFFFF; // text-white
pub const COLOR_TEXT_SECONDARY: u32 = 0x9CA3AF; // text-gray-400
pub const COLOR_ACCENT: u32 = 0x3B82F6; // blue-500
pub const COLOR_SUCCESS: u32 = 0x10B981; // green-500
pub const COLOR_WARNING: u32 = 0xF59E0B; // amber-500
pub const COLOR_ERROR: u32 = 0xEF4444; // red-500

// Typography
pub const FONT_FAMILY_SANS: &str = "Inter, system-ui, sans-serif";
pub const FONT_SIZE_XS: u8 = 12;
pub const FONT_SIZE_SM: u8 = 14;
pub const FONT_SIZE_BASE: u8 = 16;
pub const FONT_SIZE_LG: u8 = 18;
pub const FONT_SIZE_XL: u8 = 20;
pub const FONT_SIZE_2XL: u8 = 24;
pub const FONT_SIZE_3XL: u8 = 30;

// Spacing
pub const SPACE_1: u8 = 4;
pub const SPACE_2: u8 = 8;
pub const SPACE_3: u8 = 12;
pub const SPACE_4: u8 = 16;
pub const SPACE_5: u8 = 20;
pub const SPACE_6: u8 = 24;
pub const SPACE_8: u8 = 32;
pub const SPACE_10: u8 = 40;
pub const SPACE_12: u8 = 48;
pub const SPACE_16: u8 = 64;

// Border radius
pub const RADIUS_SM: u8 = 2;
pub const RADIUS_BASE: u8 = 4;
pub const RADIUS_MD: u8 = 6;
pub const RADIUS_LG: u8 = 8;
pub const RADIUS_XL: u8 = 12;
pub const RADIUS_2XL: u8 = 16;
pub const RADIUS_FULL: u8 = 9999;

// Shadows
pub const SHADOW_SM: &str = "0 1px 2px 0 rgba(0, 0, 0, 0.05)";
pub const SHADOW_BASE: &str = "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)";
pub const SHADOW_MD: &str = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)";
pub const SHADOW_LG: &str = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)";
pub const SHADOW_XL: &str = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)";

// Transitions
pub const TRANSITION_BASE: &str = "all 0.2s ease";
pub const TRANSITION_FAST: &str = "all 0.1s ease";
pub const TRANSITION_SLOW: &str = "all 0.3s ease";

// Z-index
pub const Z_0: u8 = 0;
pub const Z_10: u8 = 10;
pub const Z_20: u8 = 20;
pub const Z_30: u8 = 30;
pub const Z_40: u8 = 40;
pub const Z_50: u8 = 50;
pub const Z_AUTO: i8 = -1; // Special value
