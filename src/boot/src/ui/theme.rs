// Theme constants for UNIA OS UI
use crate::vga_buffer::Color;

// Colors
pub const COLOR_PRIMARY: Color = Color::LightGreen;
pub const COLOR_SECONDARY: Color = Color::LightBlue;
pub const COLOR_ACCENT: Color = Color::Yellow;
pub const COLOR_BACKGROUND: Color = Color::Black;
pub const COLOR_TEXT: Color = Color::White;
pub const COLOR_TEXT_MUTED: Color = Color::LightGray;
pub const COLOR_ERROR: Color = Color::LightRed;
pub const COLOR_SUCCESS: Color = Color::LightGreen;
pub const COLOR_WARNING: Color = Color::Yellow;
pub const COLOR_INFO: Color = Color::LightCyan;

// Spacing
pub const SPACING_XS: u8 = 1;
pub const SPACING_SM: u8 = 2;
pub const SPACING_MD: u8 = 4;
pub const SPACING_LG: u8 = 6;
pub const SPACING_XL: u8 = 8;

// Border styles
pub const BORDER_NONE: u8 = 0;
pub const BORDER_THIN: u8 = 1;
pub const BORDER_MEDIUM: u8 = 2;
pub const BORDER_THICK: u8 = 3;

// Border types
pub const BORDER_SOLID: u8 = 0;
pub const BORDER_DASHED: u8 = 1;
pub const BORDER_DOTTED: u8 = 2;
pub const BORDER_DOUBLE: u8 = 3;

// Border radius
pub const RADIUS_NONE: u8 = 0;
pub const RADIUS_SM: u8 = 1;
pub const RADIUS_MD: u8 = 2;
pub const RADIUS_LG: u8 = 3;
pub const RADIUS_FULL: u8 = 255; // Maximum value for u8

// Animation speeds
pub const ANIMATION_SLOW: u16 = 1000;
pub const ANIMATION_MEDIUM: u16 = 500;
pub const ANIMATION_FAST: u16 = 250;

// Font sizes
pub const FONT_SIZE_XS: u8 = 1;
pub const FONT_SIZE_SM: u8 = 2;
pub const FONT_SIZE_MD: u8 = 3;
pub const FONT_SIZE_LG: u8 = 4;
pub const FONT_SIZE_XL: u8 = 5;

// Component sizes
pub const BUTTON_HEIGHT: u8 = 3;
pub const BUTTON_MIN_WIDTH: u8 = 10;
pub const INPUT_HEIGHT: u8 = 3;
pub const CARD_MIN_HEIGHT: u8 = 5;
pub const CARD_MIN_WIDTH: u8 = 20;
