# Zig MPI Wrapper

A comprehensive Zig wrapper for MPI (Message Passing Interface) that provides idiomatic Zig bindings for parallel computing applications.

TODO: Library has quirks on the API design, still under development.

## Installation

This library requires Zig 0.15+ and an MPI implementation (Microsoft MPI on Windows, OpenMPI/MPICH on Linux/macOS).

### Windows (Microsoft MPI)
1. Install Microsoft MPI from the official Microsoft website
2. Clone this repository
3. Build with `zig build`

### Linux/macOS (OpenMPI/MPICH)
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS with Homebrew  
brew install open-mpi

# Clone and build
git clone <repository-url>
cd zig-mpi
zig build
```

## Usage

### Basic Hello World
```zig
const std = @import("std");
const mpi = @import("mpi");

pub fn main() !void {
    var env = mpi.Environment{};
    try env.init();
    defer env.deinit() catch {};

    const comm = mpi.Communicator.world;
    const my_rank = try comm.getRank();
    const world_size = try comm.getSize();

    std.debug.print("Hello from rank {} of {}\n", .{ my_rank, world_size });
}
```

### Point-to-Point Communication
```zig
// Rank 0 sends data to rank 1
if (my_rank == 0) {
    const data = [_]i32{ 42, 84, 126 };
    try comm.send(i32, &data, mpi.CommParams.dest(1).withTag(100));
} else if (my_rank == 1) {
    var buffer: [3]i32 = undefined;
    _ = try comm.recv(i32, &buffer, mpi.CommParams.source(0).withTag(100));
    std.debug.print("Received: {any}\n", .{buffer});
}
```

### Collective Operations
```zig
// Broadcast value from root to all processes
var shared_value: f64 = if (my_rank == 0) 42.7 else 0.0;
try comm.bcastValue(f64, &shared_value, 0);

// Parallel reduction
const local_value = @as(f64, @floatFromInt(my_rank + 1));
const global_sum = try mpi.convenience.parallelSum(f64, local_value, comm);
```

### Non-blocking Communication
```zig
var request_manager = mpi.RequestManager.init(allocator);
defer request_manager.deinit();

// Start non-blocking operations
const req1 = try comm.isend(f64, &data, mpi.CommParams.dest(1).withTag(200));
const req2 = try comm.irecv(f64, &buffer, mpi.CommParams.source(0).withTag(200));

try request_manager.add(req1);
try request_manager.add(req2);

// Wait for completion
var statuses = try request_manager.waitAll();
defer statuses.deinit();
```

## Examples

The repository includes comprehensive examples demonstrating:

- **01_hello_world.zig** - Basic MPI initialization and rank identification
- **02_point_to_point.zig** - Send/receive operations between processes
- **03_non_blocking.zig** - Asynchronous communication patterns
- **04_collective_ops.zig** - Broadcast, reduce, gather operations
- **05_monte_carlo_pi.zig** - Parallel Monte Carlo π estimation
- **06_process_exchange.zig** - Process data exchange patterns
- **07_benchmark.zig** - Performance measurement tools
- **08_error_handling.zig** - Error handling and recovery patterns
- **09_advanced_patterns.zig** - Complex communication patterns

Run examples with:
```bash
# Single process (for testing)
zig build run-hello

# Multiple processes
mpiexec -n 4 zig-out/bin/mpi-example-hello.exe
```

## API Reference

### Core Types
- `mpi.Environment` - MPI runtime environment management
- `mpi.Communicator` - Communication context
- `mpi.CommParams` - Communication parameters builder
- `mpi.RequestManager` - Non-blocking operation management

### Key Functions
- `send()` / `recv()` - Blocking point-to-point communication
- `isend()` / `irecv()` - Non-blocking point-to-point communication  
- `bcast()` / `bcastValue()` - Broadcast operations
- `reduce()` / `allReduce()` - Reduction operations
- `gather()` / `allGather()` - Gather operations
- `scatter()` - Scatter operations

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and ensure compatibility with multiple MPI implementations.

### Development Setup
1. Install Zig 0.15+ and MPI implementation
2. Run `zig build` to compile library and examples
3. Run `zig build test` to execute test suite
4. Test with multiple MPI implementations when possible

## License

[MIT](https://choosealicense.com/licenses/mit/)
