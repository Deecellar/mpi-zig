// Example 2: Point-to-point communication
// Demonstrates send/receive operations with timing

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

    if (world_size < 2) {
        if (my_rank == 0) {
            std.debug.print("This example requires at least 2 processes\n", .{});
        }
        return;
    }

    if (my_rank == 0) {
        // Sender - send array and single value
        const message = [_]i32{ 1, 2, 3, 4, 5 };
        const dest = mpi.CommParams.dest(1).withTag(100);

        const timer = mpi.Timer.start();
        try comm.send(i32, &message, dest);
        const send_time = timer.elapsed();

        std.debug.print("Rank 0 sent array {any} in {d:.6} seconds\n", .{ message, send_time });

        // Send a single floating point value
        try comm.sendValue(f64, 3.14159, mpi.CommParams.dest(1).withTag(101));
        std.debug.print("Rank 0 sent pi value\n", .{});
    } else if (my_rank == 1) {
        // Receiver - receive array and single value
        var buffer: [5]i32 = undefined;
        const source = mpi.CommParams.source(0).withTag(100);

        const timer = mpi.Timer.start();
        const status = try comm.recv(i32, &buffer, source);
        const recv_time = timer.elapsed();

        std.debug.print("Rank 1 received {any} from rank {} in {d:.6} seconds\n", .{ buffer, status.getSource(), recv_time });

        // Receive single value
        const result = try comm.recvValue(f64, mpi.CommParams.source(0).withTag(101));
        std.debug.print("Rank 1 received pi = {d:.5}\n", .{result.value});
    }

    // Barrier to ensure clean output
    try comm.barrier();
    if (my_rank == 0) {
        std.debug.print("Point-to-point communication complete!\n", .{});
    }
}
