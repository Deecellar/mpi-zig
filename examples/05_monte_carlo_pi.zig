// Example 5: Parallel Monte Carlo Pi calculation
// Demonstrates work distribution and parallel computation with timing

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

    // Configuration
    const total_samples: i32 = 10_000_000;

    if (my_rank == 0) {
        std.debug.print("Starting parallel Monte Carlo Pi calculation\n", .{});
        std.debug.print("Total samples: {}, Processes: {}\n", .{ total_samples, world_size });
    }

    // Distribute work evenly among processes
    const work = try comm.distributeWork(total_samples);

    std.debug.print("Rank {}: processing {} samples (from {} to {})\n", .{ my_rank, work.count, work.start, work.end });

    // Synchronize before timing
    try comm.barrier();

    // Monte Carlo computation with timing
    const result = try mpi.convenience.timing.syncTime(comm, computeMonteCarloSlice, .{ work.start, work.end, my_rank });
    const local_hits = result.result;

    if (my_rank == 0) {
        std.debug.print("\n--- Computation Timing ---\n", .{});
        std.debug.print("Max time: {d:.6}s, Min time: {d:.6}s, Avg time: {d:.6}s\n", .{ result.max_time, result.min_time, result.avg_time });
    }

    // Sum up all hits from all processes
    const total_hits = try mpi.convenience.parallelSum(i32, local_hits, comm);

    if (my_rank == 0) {
        const pi_estimate = 4.0 * @as(f64, @floatFromInt(total_hits)) / @as(f64, @floatFromInt(total_samples));
        const pi_error = @abs(pi_estimate - std.math.pi);
        const efficiency = (result.min_time / result.max_time) * 100.0;

        std.debug.print("\n--- Results ---\n", .{});
        std.debug.print("Total hits in circle: {}\n", .{total_hits});
        std.debug.print("Pi estimate: {d:.6}\n", .{pi_estimate});
        std.debug.print("Actual Pi: {d:.6}\n", .{std.math.pi});
        std.debug.print("Error: {d:.6} ({d:.4}%)\n", .{ pi_error, (pi_error / std.math.pi) * 100.0 });
        std.debug.print("Parallel efficiency: {d:.1}%\n", .{efficiency});
    }

    // Demonstrate load balancing check
    var samples_per_proc: ?[]i32 = null;
    if (my_rank == 0) {
        samples_per_proc = try allocator.alloc(i32, @intCast(world_size));
    }
    defer if (samples_per_proc) |data| allocator.free(data);

    const my_work_count = [_]i32{work.count};
    try comm.gather(i32, &my_work_count, samples_per_proc, 0);

    if (my_rank == 0) {
        std.debug.print("\nWork distribution: {any}\n", .{samples_per_proc.?});

        // Check load balance
        var min_work = samples_per_proc.?[0];
        var max_work = samples_per_proc.?[0];
        for (samples_per_proc.?[1..]) |count| {
            min_work = @min(min_work, count);
            max_work = @max(max_work, count);
        }
        const load_balance = @as(f64, @floatFromInt(min_work)) / @as(f64, @floatFromInt(max_work)) * 100.0;
        std.debug.print("Load balance: {d:.1}% (min: {}, max: {})\n", .{ load_balance, min_work, max_work });
    }

    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\nMonte Carlo Pi calculation complete!\n", .{});
    }
}

fn computeMonteCarloSlice(start: i32, end: i32, seed_offset: i32) i32 {
    // Use different seed for each process to avoid correlation
    const seed = @as(u64, @intCast(std.time.timestamp())) +% @as(u64, @intCast(seed_offset));
    var rng = std.Random.DefaultPrng.init(seed);
    var hits: i32 = 0;

    var i = start;
    while (i < end) : (i += 1) {
        // Generate random point in [-1, 1] x [-1, 1] square
        const x = rng.random().float(f64) * 2.0 - 1.0;
        const y = rng.random().float(f64) * 2.0 - 1.0;

        // Check if point is inside unit circle
        if (x * x + y * y <= 1.0) {
            hits += 1;
        }
    }

    return hits;
}
