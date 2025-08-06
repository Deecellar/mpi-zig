// Example 9: Advanced MPI patterns
// Demonstrates complex communication patterns and optimizations

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
        std.debug.print("=== Advanced MPI Patterns ===\n", .{});
        std.debug.print("Processes: {}\n\n", .{world_size});
    }

    // Pattern 1: Pipeline pattern
    try pipelinePattern(comm, my_rank, world_size, allocator);

    try comm.barrier();

    // Pattern 2: Master-Worker pattern
    if (world_size >= 3) {
        try masterWorkerPattern(comm, my_rank, world_size, allocator);
    } else if (my_rank == 0) {
        std.debug.print("Skipping master-worker pattern (requires at least 3 processes)\n", .{});
    }

    try comm.barrier();

    // Pattern 3: Hierarchical communication
    try hierarchicalPattern(comm, my_rank, world_size, allocator);

    try comm.barrier();

    // Pattern 4: Overlapping computation and communication
    try overlapPattern(comm, my_rank, world_size, allocator);

    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("\nAdvanced MPI patterns demonstration complete!\n", .{});
    }
}

fn pipelinePattern(comm: mpi.Communicator, my_rank: i32, world_size: i32, allocator: std.mem.Allocator) !void {
    _ = allocator;

    if (my_rank == 0) {
        std.debug.print("--- Pipeline Pattern ---\n", .{});
        std.debug.print("Data flows through processes 0 -> 1 -> 2 -> ... -> {}\n", .{world_size - 1});
    }

    // Each process adds its rank to the data and passes it on
    var pipeline_data: i32 = 0;

    const timer = mpi.Timer.start();

    if (my_rank == 0) {
        // Start the pipeline
        pipeline_data = 1000; // Initial value
        std.debug.print("Rank 0: Starting pipeline with value {}\n", .{pipeline_data});

        if (world_size > 1) {
            try comm.send(i32, @ptrCast(&pipeline_data), mpi.CommParams.dest(1).withTag(300));
        }
    } else {
        // Receive from previous, process, and send to next
        const prev_rank = my_rank - 1;
        _ = try comm.recv(i32, @ptrCast(&pipeline_data), mpi.CommParams.source(prev_rank).withTag(300));

        // Process the data (add rank)
        pipeline_data += my_rank * 10;
        std.debug.print("Rank {}: Received {}, processed to {}\n", .{ my_rank, pipeline_data - my_rank * 10, pipeline_data });

        // Send to next process if not the last
        if (my_rank < world_size - 1) {
            try comm.send(i32, @ptrCast(&pipeline_data), mpi.CommParams.dest(my_rank + 1).withTag(300));
        }
    }

    // Last process reports final result
    if (my_rank == world_size - 1) {
        const pipeline_time = timer.elapsed();
        std.debug.print("Rank {}: Final pipeline result = {} (completed in {d:.6}s)\n", .{ my_rank, pipeline_data, pipeline_time });
    }
}

fn masterWorkerPattern(comm: mpi.Communicator, my_rank: i32, world_size: i32, allocator: std.mem.Allocator) !void {
    if (my_rank == 0) {
        std.debug.print("\n--- Master-Worker Pattern ---\n", .{});
        std.debug.print("Master (rank 0) distributes work to {} workers\n", .{world_size - 1});

        // Master process
        const total_tasks = 20;
        const task_data = try allocator.alloc(i32, total_tasks);
        defer allocator.free(task_data);

        // Initialize tasks
        for (task_data, 0..) |*task, i| {
            task.* = @as(i32, @intCast(i * i)); // Task: compute square root
        }

        std.debug.print("Master: Distributing {} tasks to workers\n", .{total_tasks});

        var tasks_sent: i32 = 0;
        var results_received: i32 = 0;

        // Send initial tasks to workers
        var worker: i32 = 1;
        while (worker < world_size and tasks_sent < total_tasks) {
            try comm.send(i32, @ptrCast(&task_data[@intCast(tasks_sent)]), mpi.CommParams.dest(worker).withTag(400));
            std.debug.print("Master: Sent task {} (value={}) to worker {}\n", .{ tasks_sent, task_data[@intCast(tasks_sent)], worker });
            tasks_sent += 1;
            worker += 1;
        }

        // Collect results and send more work
        while (results_received < total_tasks) {
            var result: f64 = 0;
            const status = try comm.recv(f64, @ptrCast(&result), mpi.CommParams.any_source.withTag(401));
            results_received += 1;

            std.debug.print("Master: Received result {d:.2} from worker {}\n", .{ result, status.getSource() });

            // Send more work if available
            if (tasks_sent < total_tasks) {
                try comm.send(i32, @ptrCast(&task_data[@intCast(tasks_sent)]), mpi.CommParams.dest(status.getSource()).withTag(400));
                std.debug.print("Master: Sent task {} to worker {}\n", .{ tasks_sent, status.getSource() });
                tasks_sent += 1;
            } else {
                // Send termination signal
                const termination: i32 = -1;
                try comm.send(i32, @ptrCast(&termination), mpi.CommParams.dest(status.getSource()).withTag(400));
            }
        }

        std.debug.print("Master: All {} tasks completed\n", .{total_tasks});
    } else {
        // Worker process
        while (true) {
            var task: i32 = 0;
            _ = try comm.recv(i32, @ptrCast(&task), mpi.CommParams.source(0).withTag(400));

            if (task == -1) {
                std.debug.print("Worker {}: Received termination signal\n", .{my_rank});
                break;
            }

            // Process task (compute square root)
            const result = @sqrt(@as(f64, @floatFromInt(task)));

            // Simulate some work time
            std.Thread.sleep(10_000_000 * @as(u64, @intCast(my_rank))); // Variable work time

            std.debug.print("Worker {}: Processed task {} -> {d:.2}\n", .{ my_rank, task, result });

            // Send result back
            try comm.send(f64, @ptrCast(&result), mpi.CommParams.dest(0).withTag(401));
        }
    }
}

fn hierarchicalPattern(comm: mpi.Communicator, my_rank: i32, world_size: i32, allocator: std.mem.Allocator) !void {
    _ = allocator; // Mark as used for potential future use
    if (my_rank == 0) {
        std.debug.print("\n--- Hierarchical Communication Pattern ---\n", .{});
        std.debug.print("Two-level hierarchy: local leaders and global coordination\n", .{});
    }

    // Create groups of processes (simulate NUMA domains or nodes)
    const group_size = 2;
    const num_groups = @divTrunc(world_size + group_size - 1, group_size); // Ceiling division
    const my_group = @divTrunc(my_rank, group_size);
    const local_rank = @mod(my_rank, group_size);
    const is_leader = (local_rank == 0);

    std.debug.print("Rank {}: Group {}, Local rank {}, Leader: {}\n", .{ my_rank, my_group, local_rank, is_leader });

    try comm.barrier();

    // Phase 1: Local reduction within each group
    const local_value = @as(f64, @floatFromInt(my_rank * my_rank));
    var group_sum: f64 = local_value;

    // Simple local reduction (in real code, you'd use MPI_Comm_split)
    if (!is_leader) {
        // Non-leaders send to their group leader
        const leader_rank = my_group * group_size;
        try comm.send(f64, @ptrCast(&local_value), mpi.CommParams.dest(leader_rank).withTag(500));
        std.debug.print("Rank {}: Sent {d:.1} to group leader {}\n", .{ my_rank, local_value, leader_rank });
    } else {
        // Leaders collect from their group members
        const members_in_group = @min(group_size, world_size - my_group * group_size);

        for (1..@intCast(members_in_group)) |i| {
            var member_value: f64 = 0;
            const member_rank = my_group * group_size + @as(i32, @intCast(i));
            if (member_rank < world_size) {
                _ = try comm.recv(f64, @ptrCast(&member_value), mpi.CommParams.source(member_rank).withTag(500));
                group_sum += member_value;
                std.debug.print("Leader {}: Received {d:.1} from member {}\n", .{ my_rank, member_value, member_rank });
            }
        }

        std.debug.print("Leader {}: Group {} sum = {d:.1}\n", .{ my_rank, my_group, group_sum });
    }

    try comm.barrier();

    // Phase 2: Global reduction among leaders
    if (is_leader) {
        var global_sum: f64 = group_sum;

        if (my_rank == 0) {
            // Root leader collects from other leaders
            for (1..@intCast(num_groups)) |i| {
                const other_leader = @as(i32, @intCast(i)) * group_size;
                if (other_leader < world_size) {
                    var other_group_sum: f64 = 0;
                    _ = try comm.recv(f64, @ptrCast(&other_group_sum), mpi.CommParams.source(other_leader).withTag(501));
                    global_sum += other_group_sum;
                    std.debug.print("Root leader: Received group sum {d:.1} from leader {}\n", .{ other_group_sum, other_leader });
                }
            }

            std.debug.print("Root leader: Global sum = {d:.1}\n", .{global_sum});

            // Broadcast result back to other leaders
            for (1..@intCast(num_groups)) |i| {
                const other_leader = @as(i32, @intCast(i)) * group_size;
                if (other_leader < world_size) {
                    try comm.send(f64, @ptrCast(&global_sum), mpi.CommParams.dest(other_leader).withTag(502));
                }
            }
        } else {
            // Other leaders send to root and receive result
            try comm.send(f64, @ptrCast(&group_sum), mpi.CommParams.dest(0).withTag(501));
            _ = try comm.recv(f64, @ptrCast(&global_sum), mpi.CommParams.source(0).withTag(502));
            std.debug.print("Leader {}: Received global sum {d:.1}\n", .{ my_rank, global_sum });
        }

        // Broadcast to local group members
        const members_in_group = @min(group_size, world_size - my_group * group_size);
        for (1..@intCast(members_in_group)) |i| {
            const member_rank = my_group * group_size + @as(i32, @intCast(i));
            if (member_rank < world_size) {
                try comm.send(f64, @ptrCast(&global_sum), mpi.CommParams.dest(member_rank).withTag(503));
            }
        }
    } else {
        // Non-leaders receive final result from their leader
        var final_result: f64 = 0;
        const leader_rank = my_group * group_size;
        _ = try comm.recv(f64, @ptrCast(&final_result), mpi.CommParams.source(leader_rank).withTag(503));
        std.debug.print("Rank {}: Final result = {d:.1}\n", .{ my_rank, final_result });
    }
}

fn overlapPattern(comm: mpi.Communicator, my_rank: i32, world_size: i32, allocator: std.mem.Allocator) !void {
    if (my_rank == 0) {
        std.debug.print("\n--- Computation-Communication Overlap ---\n", .{});
        std.debug.print("Demonstrating overlapping computation with communication\n", .{});
    }

    if (world_size < 2) return;

    const data_size = 10000;
    const send_buffer = try allocator.alloc(f64, data_size);
    defer allocator.free(send_buffer);
    const recv_buffer = try allocator.alloc(f64, data_size);
    defer allocator.free(recv_buffer);

    // Initialize data
    for (send_buffer, 0..) |*val, i| {
        val.* = @as(f64, @floatFromInt(my_rank * 1000 + @as(i32, @intCast(i))));
    }

    const timer = mpi.Timer.start();

    if (my_rank == 0) {
        // Start non-blocking send
        var send_req = try comm.isend(f64, send_buffer, mpi.CommParams.dest(1).withTag(600));

        // Do computation while sending
        var computation_result: f64 = 0;
        for (0..1000000) |i| {
            computation_result += @sqrt(@as(f64, @floatFromInt(i)));
        }

        // Wait for send to complete
        _ = try send_req.wait();

        const total_time = timer.elapsed();
        std.debug.print("Rank 0: Overlapped send + computation in {d:.6}s, result = {d:.2}\n", .{ total_time, computation_result });
    } else if (my_rank == 1) {
        // Start non-blocking receive
        var recv_req = try comm.irecv(f64, recv_buffer, mpi.CommParams.source(0).withTag(600));

        // Do computation while receiving
        var computation_result: f64 = 0;
        for (0..500000) |i| {
            computation_result += @sin(@as(f64, @floatFromInt(i)) * 0.001);
        }

        // Wait for receive to complete
        _ = try recv_req.wait();

        const total_time = timer.elapsed();
        std.debug.print("Rank 1: Overlapped recv + computation in {d:.6}s, result = {d:.2}\n", .{ total_time, computation_result });

        // Verify received data
        const expected_first = @as(f64, @floatFromInt(0 * 1000 + 0));
        if (@abs(recv_buffer[0] - expected_first) < 0.001) {
            std.debug.print("Rank 1: ✓ Data verification passed\n", .{});
        } else {
            std.debug.print("Rank 1: ✗ Data verification failed\n", .{});
        }
    }
}
