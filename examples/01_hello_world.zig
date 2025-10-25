// Example 1: Basic MPI hello world
// Demonstrates basic MPI initialization, rank/size queries, and processor name

const std = @import("std");
const mpi = @import("mpi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize MPI environment
    var env = mpi.Environment{};
    try env.init();
    defer env.deinit() catch {};

    // Get communicator and basic info
    const comm = mpi.Communicator.world;
    const my_rank = try comm.getRank();
    const world_size = try comm.getSize();

    // Get processor name
    const processor_name = try comm.getProcessorName(allocator);
    defer allocator.free(processor_name);

    // Print hello message
    std.debug.print("Hello from rank {} of {} on {s}\n", .{ my_rank, world_size, processor_name });

    // Show MPI version and implementation info (only from root)
    if (my_rank == 0) {
        const version = try mpi.Environment.getVersion();
        const implementation = mpi.platform.detectMpiImplementation();

        std.debug.print("MPI Version: {}.{}, Implementation: {}\n", .{ version.version, version.subversion, implementation });
        std.debug.print("Total processes: {}\n", .{world_size});
    }

    // Synchronize all processes before finishing
    try comm.barrier();

    if (my_rank == 0) {
        std.debug.print("All processes synchronized. Hello world complete!\n", .{});
    }
}
