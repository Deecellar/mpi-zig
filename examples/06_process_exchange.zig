// Example 6: Process pair exchange pattern
// Demonstrates process pairing and data exchange

const std = @import("std");
const mpi = @import("mpi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var env = mpi.Environment{};
    try env.init();
    defer env.deinit() catch {};

    const comm = mpi.Communicator.world;
    const my_rank = try comm.getRank();
    const world_size = try comm.getSize();

    // Check if we have an even number of processes
    if (@mod(world_size, 2) != 0) {
        if (my_rank == 0) {
            std.debug.print("This example requires an even number of processes\n", .{});
            std.debug.print("Current processes: {}, please use an even number\n", .{world_size});
        }
        return;
    }

    if (my_rank == 0) {
        std.debug.print("Starting process exchange with {} processes\n", .{world_size});
        std.debug.print("Each process will exchange data with its pair\n\n", .{});
    }

    try comm.barrier();

    // Determine partner process (pair up adjacent processes)
    const partner = if (@mod(my_rank, 2) == 0) my_rank + 1 else my_rank - 1;

    // Create unique data for this process
    const send_data = [_]i32{
        my_rank * 100, // Base identifier
        my_rank * 10 + 1, // Some calculation
        my_rank * 10 + 2, // Another calculation
        my_rank * my_rank, // Squared value
    };
    var recv_buffer: [4]i32 = undefined;

    std.debug.print("Rank {} preparing to exchange with partner {}\n", .{ my_rank, partner });
    std.debug.print("Rank {} sending data: {any}\n", .{ my_rank, send_data });

    // Use convenience exchange function with timing
    const timer = mpi.Timer.start();
    const status = try mpi.convenience.exchange(i32, comm, &send_data, &recv_buffer, partner, 42);
    const exchange_time = timer.elapsed();

    std.debug.print("Rank {} completed exchange with rank {} in {d:.6}s\n", .{ my_rank, status.getSource(), exchange_time });
    std.debug.print("Rank {} received data: {any}\n", .{ my_rank, recv_buffer });

    try comm.barrier();

    // Verify the exchange worked correctly
    const expected_partner_base = partner * 100;
    if (recv_buffer[0] == expected_partner_base) {
        std.debug.print("Rank {}: ✓ Exchange verification PASSED\n", .{my_rank});
    } else {
        std.debug.print("Rank {}: ✗ Exchange verification FAILED (expected {}, got {})\n", .{ my_rank, expected_partner_base, recv_buffer[0] });
    }

    // Demonstrate ring exchange pattern as well
    try comm.barrier();

    if (my_rank == 0) {
        std.debug.print("\n--- Ring Exchange Pattern ---\n", .{});
    }

    // Ring exchange: send to next process, receive from previous
    const next_rank = @mod(my_rank + 1, world_size);
    const prev_rank = @mod(my_rank + world_size - 1, world_size);

    const ring_send_data = [_]i32{my_rank * 1000};
    var ring_recv_buffer: [1]i32 = undefined;

    // Use separate send/recv for ring exchange (alternative to sendrecv)
    if (@mod(my_rank, 2) == 0) {
        // Even ranks send first, then receive
        try comm.send(i32, &ring_send_data, mpi.CommParams.dest(next_rank).withTag(100));
        _ = try comm.recv(i32, &ring_recv_buffer, mpi.CommParams.source(prev_rank).withTag(100));
    } else {
        // Odd ranks receive first, then send
        _ = try comm.recv(i32, &ring_recv_buffer, mpi.CommParams.source(prev_rank).withTag(100));
        try comm.send(i32, &ring_send_data, mpi.CommParams.dest(next_rank).withTag(100));
    }

    std.debug.print("Rank {}: Ring - sent {} to rank {}, received {} from rank {}\n", .{ my_rank, ring_send_data[0], next_rank, ring_recv_buffer[0], prev_rank });

    // Calculate timing statistics across all processes
    const timing_data = [_]f64{exchange_time};
    const min_time = try mpi.convenience.parallelMin(f64, timing_data[0], comm);
    const max_time = try mpi.convenience.parallelMax(f64, timing_data[0], comm);
    const avg_time = try mpi.convenience.parallelSum(f64, timing_data[0], comm) /
        @as(f64, @floatFromInt(world_size));

    if (my_rank == 0) {
        std.debug.print("\n--- Exchange Timing Statistics ---\n", .{});
        std.debug.print("Min exchange time: {d:.6}s\n", .{min_time});
        std.debug.print("Max exchange time: {d:.6}s\n", .{max_time});
        std.debug.print("Avg exchange time: {d:.6}s\n", .{avg_time});

        if (max_time > 0) {
            const efficiency = (min_time / max_time) * 100.0;
            std.debug.print("Exchange efficiency: {d:.1}%\n", .{efficiency});
        }
    }

    try comm.barrier();

    if (my_rank == 0) {
        std.debug.print("\nProcess exchange patterns complete!\n", .{});
    }
}
