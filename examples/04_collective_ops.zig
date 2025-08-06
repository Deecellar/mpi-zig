// Example 4: Collective operations
// Demonstrates broadcast, reduce, gather operations

const std = @import("std");
const mpi = @import("mpi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var env = mpi.Environment{};
    try env.init();
    defer env.deinit() catch {};

    const comm = mpi.Communicator.world;
    const my_rank = try comm.getRank();
    const world_size = try comm.getSize();

    // === Broadcast example ===
    var shared_value: f64 = 0.0;
    if (my_rank == 0) {
        shared_value = 42.7;
        std.debug.print("Root broadcasting value: {d}\n", .{shared_value});
    }

    try comm.bcastValue(f64, &shared_value, 0);
    std.debug.print("Rank {}: received broadcast value = {d}\n", .{ my_rank, shared_value });

    try comm.barrier();

    // === Reduction examples ===
    const local_contribution = @as(f64, @floatFromInt(my_rank + 1));

    // Using convenience functions for common reductions
    const global_sum = try mpi.convenience.parallelSum(f64, local_contribution, comm);
    const global_max = try mpi.convenience.parallelMax(f64, local_contribution, comm);
    const global_min = try mpi.convenience.parallelMin(f64, local_contribution, comm);

    if (my_rank == 0) {
        const expected_sum = @as(f64, @floatFromInt(world_size * (world_size + 1))) / 2.0;
        std.debug.print("\n--- Reduction Results ---\n", .{});
        std.debug.print("Global sum: {d:.1} (expected: {d:.1})\n", .{ global_sum, expected_sum });
        std.debug.print("Global max: {d:.1}, min: {d:.1}\n", .{ global_max, global_min });
    }

    try comm.barrier();

    // === Gather example ===
    const my_data = [_]i32{my_rank * 10};
    var all_data: ?[]i32 = null;

    if (my_rank == 0) {
        all_data = try allocator.alloc(i32, @intCast(world_size));
        std.debug.print("\n--- Gather Operation ---\n", .{});
        std.debug.print("Gathering data from all processes...\n", .{});
    }
    defer if (all_data) |data| allocator.free(data);

    try comm.gather(i32, &my_data, all_data, 0);

    if (my_rank == 0 and all_data != null) {
        std.debug.print("Gathered data: {any}\n", .{all_data.?});
    }

    try comm.barrier();

    // === AllGather example ===
    const all_gathered: []i32 = try allocator.alloc(i32, @intCast(world_size));
    defer allocator.free(all_gathered);

    const my_value = [_]i32{my_rank * my_rank};
    try comm.allGather(i32, &my_value, all_gathered);

    std.debug.print("Rank {}: AllGather result = {any}\n", .{ my_rank, all_gathered });

    try comm.barrier();

    // === Scatter example ===
    var scatter_data: ?[]i32 = null;
    if (my_rank == 0) {
        scatter_data = try allocator.alloc(i32, @intCast(world_size));
        for (scatter_data.?, 0..) |*value, i| {
            value.* = @as(i32, @intCast(i * 100));
        }
        std.debug.print("\n--- Scatter Operation ---\n", .{});
        std.debug.print("Scattering data: {any}\n", .{scatter_data.?});
    }
    defer if (scatter_data) |data| allocator.free(data);

    var my_scattered: [1]i32 = undefined;
    try comm.scatter(i32, scatter_data, &my_scattered, 0);

    std.debug.print("Rank {}: received scattered value = {}\n", .{ my_rank, my_scattered[0] });

    try comm.barrier();

    if (my_rank == 0) {
        std.debug.print("\nCollective operations example complete!\n", .{});
    }
}
