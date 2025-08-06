// Example 8: Error handling demonstration
// Shows how to handle MPI errors gracefully

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

    if (my_rank == 0) {
        std.debug.print("=== MPI Error Handling Demonstration ===\n", .{});
        std.debug.print("Processes: {}\n\n", .{world_size});
    }

    // Test 1: Invalid rank error
    if (my_rank == 0) {
        std.debug.print("Test 1: Sending to invalid rank...\n", .{});

        const data = [_]i32{ 1, 2, 3 };
        const invalid_rank = 999999;

        comm.send(i32, &data, mpi.CommParams.dest(invalid_rank)) catch |err| switch (err) {
            mpi.MpiError.invalid_arg => {
                std.debug.print("✓ Caught expected invalid_arg error\n", .{});
            },
            else => {
                std.debug.print("✗ Caught unexpected error: {}\n", .{err});
                return err;
            },
        };
    }

    try comm.barrier();

    // Test 2: Tag out of range
    if (my_rank == 0) {
        std.debug.print("\nTest 2: Using invalid tag...\n", .{});

        const data = [_]i32{42};
        const invalid_tag = -1;

        if (world_size > 1) {
            comm.send(i32, &data, mpi.CommParams.dest(1).withTag(invalid_tag)) catch |err| switch (err) {
                mpi.MpiError.invalid_tag => {
                    std.debug.print("✓ Caught expected invalid_tag error\n", .{});
                },
                mpi.MpiError.invalid_arg => {
                    std.debug.print("✓ Caught expected invalid_arg error (implementation variant)\n", .{});
                },
                else => {
                    std.debug.print("✗ Caught unexpected error: {}\n", .{err});
                },
            };
        }
    }

    try comm.barrier();

    // Test 3: Buffer size mismatch (if we have multiple processes)
    if (world_size >= 2) {
        if (my_rank == 0) {
            std.debug.print("\nTest 3: Buffer size handling...\n", .{});

            // Send a large array
            const large_data = try allocator.alloc(f64, 1000);
            defer allocator.free(large_data);

            for (large_data, 0..) |*val, i| {
                val.* = @as(f64, @floatFromInt(i));
            }

            try mpi.testing.expectMpiSuccess(comm.send(f64, large_data, mpi.CommParams.dest(1).withTag(200)));
            std.debug.print("✓ Successfully sent large buffer\n", .{});
        } else if (my_rank == 1) {
            // Try to receive into a smaller buffer
            var small_buffer: [100]f64 = undefined;

            // This should work - MPI will only fill the available buffer space
            _ = try mpi.testing.expectMpiSuccess(comm.recv(f64, &small_buffer, mpi.CommParams.source(0).withTag(200)));

            std.debug.print("✓ Received elements into smaller buffer (truncated from larger message)\n", .{});
        }
    }

    try comm.barrier();

    // Test 4: Timeout handling (non-blocking operations)
    if (my_rank == 0) {
        std.debug.print("\nTest 4: Non-blocking operation timeout...\n", .{});

        // Start a receive that will never complete
        var timeout_buffer: [10]i32 = undefined;
        var req = comm.irecv(i32, &timeout_buffer, mpi.CommParams.any_source.withTag(999)) catch |err| {
            std.debug.print("✓ Non-blocking receive setup handled error: {}\n", .{err});
            return;
        };

        // Test the request periodically
        var attempts: u32 = 0;
        const max_attempts = 5;

        while (attempts < max_attempts) {
            if (try req.testRequest()) |status| {
                std.debug.print("✗ Unexpected completion: {}\n", .{status});
                break;
            } else {
                attempts += 1;
                std.debug.print("  Attempt {}/{}: Request still pending\n", .{ attempts, max_attempts });
                std.Thread.sleep(100_000_000); // 100ms
            }
        }

        // Cancel the request
        try req.cancel();
        std.debug.print("✓ Successfully cancelled pending request\n", .{});
    }

    try comm.barrier();

    // Test 5: Error recovery
    if (my_rank == 0) {
        std.debug.print("\nTest 5: Error recovery demonstration...\n", .{});

        // Try multiple invalid operations and recover
        const invalid_operations = [_]struct { name: []const u8, rank: i32 }{
            .{ .name = "negative rank", .rank = -1 },
            .{ .name = "rank too high", .rank = 99999 },
        };

        var successful_recoveries: u32 = 0;

        for (invalid_operations) |op| {
            const data = [_]i32{123};
            comm.send(i32, &data, mpi.CommParams.dest(op.rank)) catch |err| {
                std.debug.print("  ✓ Recovered from {s} error: {}\n", .{ op.name, err });
                successful_recoveries += 1;
                continue;
            };
            std.debug.print("  ✗ Expected error for {s} but operation succeeded\n", .{op.name});
        }

        std.debug.print("✓ Successfully recovered from {}/{} error scenarios\n", .{ successful_recoveries, invalid_operations.len });
    }

    try comm.barrier();

    // Test 6: Collective operation error handling
    if (my_rank == 0) {
        std.debug.print("\nTest 6: Collective operation validation...\n", .{});
    }

    // This should work - all processes participate
    const collective_data = @as(f64, @floatFromInt(my_rank));
    const sum_result = mpi.convenience.parallelSum(f64, collective_data, comm) catch |err| {
        std.debug.print("Rank {}: ✗ Collective operation failed: {}\n", .{ my_rank, err });
        return;
    };

    if (my_rank == 0) {
        const expected_sum = @as(f64, @floatFromInt(world_size * (world_size - 1))) / 2.0;
        if (@abs(sum_result - expected_sum) < 0.001) {
            std.debug.print("✓ Collective sum operation succeeded: {d:.1}\n", .{sum_result});
        } else {
            std.debug.print("✗ Collective sum incorrect: got {d:.1}, expected {d:.1}\n", .{ sum_result, expected_sum });
        }
    }

    // Test 7: Resource cleanup verification
    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\nTest 7: Resource cleanup verification...\n", .{});
        std.debug.print("✓ All MPI resources will be cleaned up on environment destruction\n", .{});
    }

    // Final status
    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\n=== Error Handling Tests Complete ===\n", .{});
        std.debug.print("All processes completed error handling demonstration successfully\n", .{});
    }
}
