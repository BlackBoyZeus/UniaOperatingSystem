# UNIA OS Web Simulation Guide

This guide explains how to use the UNIA OS web simulation, which provides a browser-based representation of the UNIA Operating System interface.

## Overview

The UNIA OS web simulation is a browser-based version of the UNIA OS dashboard that:

- Runs in any modern web browser
- Provides a visual representation of the UNIA OS interface
- Simulates system metrics and activities
- Works on any hardware architecture (x86_64, ARM64, etc.)

This is particularly useful for:
- Users on non-x86_64 hardware (like Apple Silicon Macs)
- Quick demonstrations without full OS installation
- Development and testing of UI components

## Running the Web Simulation

### Using the Convenience Script

We provide a convenience script to easily run the web simulation:

```bash
# Run the web simulation
./run_web_simulation.sh
```

This script will:
1. Start a local web server
2. Automatically select an available port
3. Provide a URL to access the simulation

### Manual Setup

If you prefer to set up the web simulation manually:

```bash
# Navigate to the web directory
cd src/boot/web

# Start a Python HTTP server
python -m http.server 8000
# Or if you have Python 3:
python3 -m http.server 8000

# Then open your browser to http://localhost:8000
```

## Web Simulation Features

The web simulation includes:

### Dashboard Interface
- System overview with performance metrics
- Navigation sidebar
- Activity feed

### Real-time Charts
- CPU usage simulation
- Memory usage simulation
- Network activity simulation

### Simulated System Activities
- System initialization events
- AI subsystem activities
- Networking events
- Game engine status

## Customizing the Web Simulation

You can customize the web simulation by modifying the following files:

- `src/boot/web/index.html`: Main HTML structure
- `src/boot/web/styles.css`: CSS styling (if separated)
- `src/boot/web/script.js`: JavaScript functionality (if separated)

## Limitations

The web simulation has the following limitations compared to the full bootable UNIA OS:

1. **Limited Functionality**: The simulation only shows the UI, without actual OS functionality
2. **No Hardware Access**: Cannot access or control actual hardware
3. **No Real Metrics**: Performance metrics are simulated, not actual system measurements
4. **No AI Processing**: AI features are simulated, not actually running

## Using the Web Simulation for Development

Developers can use the web simulation to:

1. **Test UI Changes**: Quickly iterate on UI design without rebuilding the OS
2. **Demonstrate Features**: Show planned features before implementation
3. **Get Feedback**: Share the UI with users for feedback
4. **Cross-Platform Testing**: Test the UI on different browsers and devices

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   - Error: "Address already in use"
   - Solution: Change the port number (e.g., `python -m http.server 8001`)

2. **Missing Dependencies**:
   - Error: Chart.js or other libraries not loading
   - Solution: Check your internet connection, as these are loaded from CDNs

3. **Browser Compatibility**:
   - Issue: UI looks incorrect in certain browsers
   - Solution: Use a modern browser like Chrome, Firefox, or Safari

## Future Enhancements

We plan to enhance the web simulation with:

1. More interactive elements
2. Better representation of actual OS features
3. WebAssembly integration for more realistic simulation
4. Collaborative features for multiple users
