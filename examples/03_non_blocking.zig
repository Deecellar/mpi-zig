// Example 3: Asynchronous (non-blocking) communication
// Demonstrates isend/irecv with RequestManager

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
            std.debug.print("This example requires at least 2 processes\n", .{});
        }
        return;
    }

    var request_manager = mpi.RequestManager.init(allocator);
    defer request_manager.deinit();

    if (my_rank == 0) {
        const data1 = [_]f64{ 1.1, 2.2, 3.3 };
        const data2 = [_]f64{ 4.4, 5.5, 6.6 };

        std.debug.print("Rank 0: Starting non-blocking sends\n", .{});

        // Start multiple non-blocking sends
        const req1 = try comm.isend(f64, &data1, mpi.CommParams.dest(1).withTag(200));
        const req2 = try comm.isend(f64, &data2, mpi.CommParams.dest(1).withTag(201));

        try request_manager.add(req1);
        try request_manager.add(req2);

        // Do some work while sending
        std.debug.print("Rank 0: Doing computation while sending...\n", .{});
        var sum: f64 = 0.0;
        for (0..1000000) |i| {
            sum += @sqrt(@as(f64, @floatFromInt(i)));
        }

        // Wait for all sends to complete
        var statuses = try request_manager.waitAll();
        defer statuses.deinit();

        std.debug.print("Rank 0: All sends completed, computed sum = {d:.2}\n", .{sum});
    } else if (my_rank == 1) {
        var buffer1: [3]f64 = undefined;
        var buffer2: [3]f64 = undefined;

        std.debug.print("Rank 1: Starting non-blocking receives\n", .{});

        // Start multiple non-blocking receives
        const req1 = try comm.irecv(f64, &buffer1, mpi.CommParams.source(0).withTag(200));
        const req2 = try comm.irecv(f64, &buffer2, mpi.CommParams.source(0).withTag(201));

        try request_manager.add(req1);
        try request_manager.add(req2);

        // Poll for completion
        var poll_count: u32 = 0;
        while (true) {
            poll_count += 1;
            if (try request_manager.testAll()) |statuses| {
                defer statuses.deinit();
                std.debug.print("Rank 1: Received after {} polls - data1={any}, data2={any}\n", .{ poll_count, buffer1, buffer2 });
                break;
            }
            if (poll_count % 1000 == 0) {
                std.debug.print("Rank 1: Still waiting... (poll {})\n", .{poll_count});
            }
            std.Thread.sleep(1000); // 1 microsecond
        }
    }

    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("Non-blocking communication example complete!\n", .{});
    }
}
