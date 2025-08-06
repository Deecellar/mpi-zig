// Example 7: Performance benchmarking
// Demonstrates bandwidth and latency measurements

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

    if (world_size < 2) {
        if (my_rank == 0) {
            std.debug.print("This benchmark requires at least 2 processes\n", .{});
        }
        return;
    }

    // Show platform information
    if (my_rank == 0) {
        const optimizations = mpi.platform.getOptimizations();
        std.debug.print("=== MPI Performance Benchmark ===\n", .{});
        std.debug.print("Processes: {}\n", .{world_size});
        std.debug.print("Platform optimizations:\n", .{});
        std.debug.print("  CUDA support: {}\n", .{optimizations.supports_cuda});
        std.debug.print("  RDMA support: {}\n", .{optimizations.supports_rdma});
        std.debug.print("  Max message size: {}\n", .{optimizations.max_message_size});
        std.debug.print("\n", .{});
    }

    try comm.barrier();

    // Latency test (ping-pong with small messages)
    if (my_rank == 0) {
        std.debug.print("--- Latency Test (Ping-Pong) ---\n", .{});
    }

    if (my_rank == 0 or my_rank == 1) {
        try latencyTest(comm, my_rank);
    }

    try comm.barrier();

    // Bandwidth test with different message sizes
    if (my_rank == 0) {
        std.debug.print("\n--- Bandwidth Test ---\n", .{});
        std.debug.print("{s:>12} | {s:>12} | {s:>12} | {s:>12}\n", .{ "Size (B)", "Time (μs)", "Bandwidth", "Rate" });
        std.debug.print("{s:-<12}-+-{s:-<12}-+-{s:-<12}-+-{s:-<12}\n", .{ "", "", "", "" });
    }

    const message_sizes = [_]usize{ 1, 64, 1024, 8192, 65536, 524288, 1048576 };

    for (message_sizes) |msg_size| {
        if (my_rank == 0 or my_rank == 1) {
            try bandwidthTest(comm, my_rank, msg_size, allocator);
        }
        try comm.barrier();
    }

    // Collective operation benchmarks
    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\n--- Collective Operations Benchmark ---\n", .{});
    }

    try collectiveBenchmark(comm, my_rank, world_size);

    // All-to-all communication pattern benchmark
    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\n--- All-to-All Communication ---\n", .{});
    }

    try allToAllBenchmark(comm, my_rank, world_size, allocator);

    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\nBenchmark complete!\n", .{});
    }
}

fn latencyTest(comm: mpi.Communicator, my_rank: i32) !void {
    const iterations = 1000;
    const warmup = 100;
    const data = [_]u8{42};

    if (my_rank == 0) {
        // Warmup
        for (0..warmup) |_| {
            try comm.send(u8, &data, mpi.CommParams.dest(1).withTag(0));
            var ack: [1]u8 = undefined;
            _ = try comm.recv(u8, &ack, mpi.CommParams.source(1).withTag(0));
        }

        // Actual measurement
        const timer = mpi.Timer.start();
        for (0..iterations) |_| {
            try comm.send(u8, &data, mpi.CommParams.dest(1).withTag(0));
            var ack: [1]u8 = undefined;
            _ = try comm.recv(u8, &ack, mpi.CommParams.source(1).withTag(0));
        }
        const total_time = timer.elapsed();

        const round_trip_time = total_time / @as(f64, @floatFromInt(iterations));
        const latency = round_trip_time / 2.0; // One-way latency

        std.debug.print("Round-trip time: {d:.3} μs\n", .{round_trip_time * 1_000_000});
        std.debug.print("One-way latency: {d:.3} μs\n", .{latency * 1_000_000});
    } else if (my_rank == 1) {
        // Warmup
        for (0..warmup) |_| {
            var buffer: [1]u8 = undefined;
            _ = try comm.recv(u8, &buffer, mpi.CommParams.source(0).withTag(0));
            try comm.send(u8, &buffer, mpi.CommParams.dest(0).withTag(0));
        }

        // Echo back received data
        for (0..iterations) |_| {
            var buffer: [1]u8 = undefined;
            _ = try comm.recv(u8, &buffer, mpi.CommParams.source(0).withTag(0));
            try comm.send(u8, &buffer, mpi.CommParams.dest(0).withTag(0));
        }
    }
}

fn bandwidthTest(comm: mpi.Communicator, my_rank: i32, msg_size: usize, allocator: std.mem.Allocator) !void {
    const iterations: u32 = if (msg_size < 1024) 1000 else if (msg_size < 65536) 100 else 10;

    if (my_rank == 0) {
        const send_buffer = try allocator.alloc(u8, msg_size);
        defer allocator.free(send_buffer);

        // Fill with pattern
        for (send_buffer, 0..) |*byte, i| {
            byte.* = @intCast(i % 256);
        }

        const timer = mpi.Timer.start();

        for (0..iterations) |_| {
            try comm.send(u8, send_buffer, mpi.CommParams.dest(1).withTag(0));
            var ack: [1]u8 = undefined;
            _ = try comm.recv(u8, &ack, mpi.CommParams.source(1).withTag(0));
        }

        const total_time = timer.elapsed();
        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const bandwidth_bps = @as(f64, @floatFromInt(msg_size)) / avg_time;
        const bandwidth_mbps = bandwidth_bps / (1024.0 * 1024.0);

        const size_str = if (msg_size >= 1024 * 1024)
            std.fmt.allocPrint(allocator, "{d} MB", .{msg_size / (1024 * 1024)}) catch "N/A"
        else if (msg_size >= 1024)
            std.fmt.allocPrint(allocator, "{d} KB", .{msg_size / 1024}) catch "N/A"
        else
            std.fmt.allocPrint(allocator, "{d} B", .{msg_size}) catch "N/A";
        defer allocator.free(size_str);

        std.debug.print("{s:>12} | {d:>9.3} | {d:>9.2} MB/s | {d:>9.1} msg/s\n", .{ size_str, avg_time * 1_000_000, bandwidth_mbps, 1.0 / avg_time });
    } else if (my_rank == 1) {
        const recv_buffer = try allocator.alloc(u8, msg_size);
        defer allocator.free(recv_buffer);

        for (0..iterations) |_| {
            _ = try comm.recv(u8, recv_buffer, mpi.CommParams.source(0).withTag(0));
            const ack = [_]u8{1};
            try comm.send(u8, &ack, mpi.CommParams.dest(0).withTag(0));
        }
    }
}

fn collectiveBenchmark(comm: mpi.Communicator, my_rank: i32, world_size: i32) !void {
    _ = world_size; // Mark as used for future implementation
    const data_size = 1024;
    const iterations = 100;

    // Broadcast benchmark
    if (my_rank == 0) {
        var bcast_data: [data_size]f64 = undefined;
        for (&bcast_data, 0..) |*val, i| {
            val.* = @as(f64, @floatFromInt(i));
        }

        const timer = mpi.Timer.start();
        for (0..iterations) |_| {
            try comm.bcast(f64, &bcast_data, 0);
        }
        const bcast_time = timer.elapsed() / @as(f64, @floatFromInt(iterations));

        std.debug.print("Broadcast ({} doubles): {d:.3} μs\n", .{ data_size, bcast_time * 1_000_000 });
    } else {
        var bcast_data: [data_size]f64 = undefined;
        for (0..iterations) |_| {
            try comm.bcast(f64, &bcast_data, 0);
        }
    }

    // Barrier benchmark
    try comm.barrier();
    const barrier_timer = mpi.Timer.start();
    for (0..iterations) |_| {
        try comm.barrier();
    }
    const barrier_time = barrier_timer.elapsed() / @as(f64, @floatFromInt(iterations));

    if (my_rank == 0) {
        std.debug.print("Barrier: {d:.3} μs\n", .{barrier_time * 1_000_000});
    }

    // Allreduce benchmark
    const local_sum = @as(f64, @floatFromInt(my_rank));
    const allreduce_timer = mpi.Timer.start();
    for (0..iterations) |_| {
        _ = try mpi.convenience.parallelSum(f64, local_sum, comm);
    }
    const allreduce_time = allreduce_timer.elapsed() / @as(f64, @floatFromInt(iterations));

    if (my_rank == 0) {
        std.debug.print("Allreduce (sum): {d:.3} μs\n", .{allreduce_time * 1_000_000});
    }
}

fn allToAllBenchmark(comm: mpi.Communicator, my_rank: i32, world_size: i32, allocator: std.mem.Allocator) !void {
    const msg_per_proc = 64; // integers per process
    const iterations = 50;

    const send_buffer = try allocator.alloc(i32, @intCast(world_size * msg_per_proc));
    defer allocator.free(send_buffer);
    const recv_buffer = try allocator.alloc(i32, @intCast(world_size * msg_per_proc));
    defer allocator.free(recv_buffer);

    // Fill send buffer
    for (send_buffer, 0..) |*val, i| {
        val.* = my_rank * 1000 + @as(i32, @intCast(i));
    }

    try comm.barrier();
    const timer = mpi.Timer.start();

    for (0..iterations) |_| {
        // Simple all-to-all: send to each process
        for (0..@intCast(world_size)) |dest| {
            if (dest != my_rank) {
                const send_slice = send_buffer[@intCast(dest * msg_per_proc)..@intCast((dest + 1) * msg_per_proc)];
                try comm.send(i32, send_slice, mpi.CommParams.dest(@intCast(dest)).withTag(100));
            }
        }

        // Receive from each process
        for (0..@intCast(world_size)) |src| {
            if (src != my_rank) {
                const recv_slice = recv_buffer[@intCast(src * msg_per_proc)..@intCast((src + 1) * msg_per_proc)];
                _ = try comm.recv(i32, recv_slice, mpi.CommParams.source(@intCast(src)).withTag(100));
            }
        }
    }

    const total_time = timer.elapsed() / @as(f64, @floatFromInt(iterations));
    const total_data = @as(f64, @floatFromInt(world_size * msg_per_proc * @sizeOf(i32)));
    const bandwidth = total_data / total_time / (1024.0 * 1024.0);

    if (my_rank == 0) {
        std.debug.print("All-to-all ({} ints/proc): {d:.3} ms, {d:.2} MB/s aggregate\n", .{ msg_per_proc, total_time * 1000, bandwidth });
    }
}
