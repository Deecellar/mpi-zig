// Example 10: Parallel Area Under a Curve Calculation
// Demonstrates numerical integration using the trapezoidal rule

const std = @import("std");
const mpi = @import("mpi");

// The function to integrate. Let's use f(x) = x^2 as an example.
fn f(x: f64) f64 {
    return x * x;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const start_x: f64 = 0.0;
    const end_x: f64 = 324.0;
    const total_steps: i32 = std.math.maxInt(i32);
    const step_size = (end_x - start_x) / @as(f64, @floatFromInt(total_steps));


    // Initialize MPI environment
    var env = mpi.Environment{};
    try env.init();
    defer env.deinit() catch {};

    const comm = mpi.Communicator.world;
    const my_rank = try comm.getRank();
    const world_size = try comm.getSize();

    // Integration parameters


    if (my_rank == 0) {
        std.debug.print("Starting parallel area calculation for f(x) = x^2\n", .{});
        std.debug.print("Range: [{}, {}], Total Steps: {}, Processes: {}\n", .{ start_x, end_x, total_steps, world_size });
    }

    // Distribute the integration steps among processes
    const work = try comm.distributeWork(total_steps);

    // Synchronize before timing the computation
    try comm.barrier();
    const timer = mpi.Timer.start();

    // Each process calculates its partial area using the trapezoidal rule
    var local_area: f64 = 0.0;
    var i = work.start;
    while (i < work.end) : (i += 1) {
        const x1 = start_x + @as(f64, @floatFromInt(i)) * step_size;
        const x2 = start_x + @as(f64, @floatFromInt(i + 1)) * step_size;
        local_area += (f(x1) + f(x2)) / 2.0 * step_size;
    }

    // Use a reduction operation to sum the partial areas from all processes
    const total_area = try mpi.convenience.parallelSum(f64, local_area, comm);

    try comm.barrier();
    const elapsed_time = timer.elapsed();

    // The root process prints the final result and timing information
    if (my_rank == 0) {
        const expected_area = (end_x * end_x * end_x) / 3.0 - (start_x * start_x * start_x) / 3.0; // Integral of x^2 is x^3/3
        const err = @abs(total_area - expected_area);

        std.debug.print("\n--- Results ---\n", .{});
        std.debug.print("Calculation finished in {d:.6}s\n", .{elapsed_time});
        std.debug.print("Calculated Area: {d:.12}\n", .{total_area});
        std.debug.print("Expected Area:   {d:.12}\n", .{expected_area});
        std.debug.print("Error:           {d:.12}\n", .{err});
    }
}