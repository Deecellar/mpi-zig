//! Modern, type-safe MPI bindings for Zig 0.15+
//!
//! This library provides an idiomatic Zig interface to MPI (Message Passing Interface)
//! with automatic error handling, compile-time type safety, and resource management.
//! It follows Zig naming conventions and design patterns while maintaining full
//! MPI functionality for high-performance parallel computing.
//!
//! Key features:
//! - Type-safe communication operations with compile-time datatype mapping
//! - Automatic error conversion from MPI integer codes to Zig error types
//! - RAII-style resource management for requests and environments
//! - High-level convenience functions for common parallel patterns
//! - Platform-specific optimizations and MPI implementation detection
//! - Comprehensive test utilities for MPI application development
//!
//! Basic usage:
//! ```zig
//! var env = mpi.Environment{};
//! try env.init();
//! defer env.deinit() catch {};
//!
//! const comm = mpi.Communicator.world;
//! const rank = try comm.getRank();
//! const size = try comm.getSize();
//!
//! if (rank == 0) {
//!     const data = [_]i32{1, 2, 3};
//!     try comm.send(i32, &data, mpi.CommParams.dest(1));
//! }
//! ```

/// MPI Wrapper for Zig 0.15+ - Modern, type-safe MPI bindings
/// Provides idiomatic Zig interface to MPI with automatic error handling,
/// compile-time type safety, and resource management.
///
/// Follows Zig naming conventions: camelCase functions, snake_case variables, PascalCase types
const std = @import("std");
const builtin = @import("builtin");
const c = @cImport({
    @cInclude("mpi.h");
});

/// MPI-specific error types with detailed categorization.
/// All MPI operations return these errors instead of integer codes.
pub const MpiError = error{
    /// Invalid buffer pointer or buffer configuration
    invalid_buffer,
    /// Count parameter is negative or exceeds implementation limits
    invalid_count,
    /// Unsupported or mismatched MPI datatype
    invalid_datatype,
    /// Tag value outside valid range (typically 0 to MPI_TAG_UB)
    invalid_tag,
    /// Invalid or null communicator handle
    invalid_comm,
    /// Invalid group handle or group operation
    invalid_group,
    /// Invalid reduction operation specified
    invalid_op,
    /// Invalid topology or topology operation
    invalid_topology,
    /// Invalid dimension specification for topology
    invalid_dims,
    /// Generic invalid argument error
    invalid_arg,
    /// Unknown or unclassified MPI error
    unknown,
    /// Message truncated due to insufficient buffer space
    truncate,
    /// Other MPI implementation-specific error
    other,
    /// Internal MPI implementation error
    intern,
    /// Error information available in status object
    in_status,
    /// Operation still pending completion
    pending,
    /// Invalid request handle or request operation
    request,
    /// Root rank mismatch in collective operation
    root_mismatch,
    /// Group mismatch in operation requiring consistent groups
    group_mismatch,
    /// Operation mismatch in collective operation
    op_mismatch,
    /// Topology type mismatch
    topology_mismatch,
    /// Dimension specification mismatch
    dims_mismatch,
    /// Argument type or value mismatch
    arg_mismatch,
    /// MPI initialization failed
    init_failed,
    /// MPI finalization failed
    finalize_failed,
    /// Memory allocation failed during MPI operation
    OutOfMemory,
};

/// Converts MPI C integer error codes to typed Zig errors.
/// Assumes error code is a valid MPI error constant from mpi.h.
/// Returns appropriate MpiError variant for the given code.
fn mpiErrorToZig(err: c_int) MpiError {
    return switch (err) {
        c.MPI_ERR_BUFFER => MpiError.invalid_buffer,
        c.MPI_ERR_COUNT => MpiError.invalid_count,
        c.MPI_ERR_TYPE => MpiError.invalid_datatype,
        c.MPI_ERR_TAG => MpiError.invalid_tag,
        c.MPI_ERR_COMM => MpiError.invalid_comm,
        c.MPI_ERR_RANK => MpiError.invalid_arg,
        c.MPI_ERR_ROOT => MpiError.root_mismatch,
        c.MPI_ERR_GROUP => MpiError.invalid_group,
        c.MPI_ERR_OP => MpiError.invalid_op,
        c.MPI_ERR_TOPOLOGY => MpiError.invalid_topology,
        c.MPI_ERR_DIMS => MpiError.invalid_dims,
        c.MPI_ERR_ARG => MpiError.invalid_arg,
        c.MPI_ERR_UNKNOWN => MpiError.unknown,
        c.MPI_ERR_TRUNCATE => MpiError.truncate,
        c.MPI_ERR_OTHER => MpiError.other,
        c.MPI_ERR_INTERN => MpiError.intern,
        c.MPI_ERR_IN_STATUS => MpiError.in_status,
        c.MPI_ERR_PENDING => MpiError.pending,
        c.MPI_ERR_REQUEST => MpiError.request,
        else => MpiError.unknown,
    };
}

/// Validates MPI operation result and converts errors to Zig error types.
/// Returns void on success, appropriate MpiError on failure.
/// Assumes result is a valid MPI return code from any MPI function.
inline fn checkMpiResult(result: c_int) MpiError!void {
    if (result != c.MPI_SUCCESS) {
        return mpiErrorToZig(result);
    }
}

/// Platform-specific MPI datatype mapping
fn getMpiDatatypeImpl(comptime T: type) c.MPI_Datatype {
    if (builtin.os.tag == .windows) {
        // Windows/MSMPI uses direct constants
        return switch (T) {
            i8 => c.MPI_INT8_T,
            u8 => c.MPI_UINT8_T,
            i16 => c.MPI_INT16_T,
            u16 => c.MPI_UINT16_T,
            i32 => c.MPI_INT32_T,
            u32 => c.MPI_UINT32_T,
            i64 => c.MPI_INT64_T,
            u64 => c.MPI_UINT64_T,
            f32 => c.MPI_FLOAT,
            f64 => c.MPI_DOUBLE,
            c_int => c.MPI_INT,
            bool => c.MPI_C_BOOL,
            else => @compileError("Unsupported MPI datatype for type: " ++ @typeName(T)),
        };
    } else {
        // OpenMPI on Linux uses global variables
        return switch (T) {
            i8 => @ptrCast(&c.ompi_mpi_int8_t),
            u8 => @ptrCast(&c.ompi_mpi_uint8_t),
            i16 => @ptrCast(&c.ompi_mpi_int16_t),
            u16 => @ptrCast(&c.ompi_mpi_uint16_t),
            i32 => @ptrCast(&c.ompi_mpi_int32_t),
            u32 => @ptrCast(&c.ompi_mpi_uint32_t),
            i64 => @ptrCast(&c.ompi_mpi_int64_t),
            u64 => @ptrCast(&c.ompi_mpi_uint64_t),
            f32 => @ptrCast(&c.ompi_mpi_float),
            f64 => @ptrCast(&c.ompi_mpi_double),
            c_int => @ptrCast(&c.ompi_mpi_int),
            bool => @ptrCast(&c.ompi_mpi_c_bool),
            else => @compileError("Unsupported MPI datatype for type: " ++ @typeName(T)),
        };
    }
}

/// Maps Zig types to corresponding MPI datatypes at compile time.
/// Supports all standard integer, floating-point, and boolean types.
/// Assumes type T is supported by the underlying MPI implementation.
/// Compile error if type is not supported.
pub fn getMpiDatatype(comptime T: type) c.MPI_Datatype {
    return getMpiDatatypeImpl(T);
}

/// Thread support levels available in MPI implementations.
/// Determines level of thread safety provided by MPI library.
pub const ThreadSupport = enum(c_int) {
    single = c.MPI_THREAD_SINGLE,
    funneled = c.MPI_THREAD_FUNNELED,
    serialized = c.MPI_THREAD_SERIALIZED,
    multiple = c.MPI_THREAD_MULTIPLE,
};

/// Platform-specific MPI operation mapping
const MpiOps = struct {
    pub fn getMax() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_MAX;
        } else {
            return @ptrCast(&c.ompi_mpi_op_max);
        }
    }
    
    pub fn getMin() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_MIN;
        } else {
            return @ptrCast(&c.ompi_mpi_op_min);
        }
    }
    
    pub fn getSum() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_SUM;
        } else {
            return @ptrCast(&c.ompi_mpi_op_sum);
        }
    }
    
    pub fn getProd() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_PROD;
        } else {
            return @ptrCast(&c.ompi_mpi_op_prod);
        }
    }
    
    pub fn getLand() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_LAND;
        } else {
            return @ptrCast(&c.ompi_mpi_op_land);
        }
    }
    
    pub fn getBand() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_BAND;
        } else {
            return @ptrCast(&c.ompi_mpi_op_band);
        }
    }
    
    pub fn getLor() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_LOR;
        } else {
            return @ptrCast(&c.ompi_mpi_op_lor);
        }
    }
    
    pub fn getBor() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_BOR;
        } else {
            return @ptrCast(&c.ompi_mpi_op_bor);
        }
    }
    
    pub fn getLxor() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_LXOR;
        } else {
            return @ptrCast(&c.ompi_mpi_op_lxor);
        }
    }
    
    pub fn getBxor() c.MPI_Op {
        if (builtin.os.tag == .windows) {
            return c.MPI_BXOR;
        } else {
            return @ptrCast(&c.ompi_mpi_op_bxor);
        }
    }
};

/// MPI reduction operations for collective computations.
/// Each operation defines how values are combined across processes.
pub const Operation = enum {
    max,
    min,
    sum,
    prod,
    land,
    band,
    lor,
    bor,
    lxor,
    bxor,

    /// Converts Operation enum to MPI C constant.
    pub fn toC(self: Operation) c.MPI_Op {
        return switch (self) {
            .max => MpiOps.getMax(),
            .min => MpiOps.getMin(),
            .sum => MpiOps.getSum(),
            .prod => MpiOps.getProd(),
            .land => MpiOps.getLand(),
            .band => MpiOps.getBand(),
            .lor => MpiOps.getLor(),
            .bor => MpiOps.getBor(),
            .lxor => MpiOps.getLxor(),
            .bxor => MpiOps.getBxor(),
        };
    }
};

/// High-precision timing utilities for performance measurement.
/// Uses MPI_Wtime for consistent timing across MPI processes.
pub const Timer = struct {
    start_time: f64,

    /// Creates new timer starting from current time.
    pub fn start() Timer {
        return Timer{ .start_time = c.MPI_Wtime() };
    }

    /// Returns elapsed time in seconds since timer creation.
    pub fn elapsed(self: Timer) f64 {
        return c.MPI_Wtime() - self.start_time;
    }

    /// Resets timer to current time.
    pub fn restart(self: *Timer) void {
        self.start_time = c.MPI_Wtime();
    }

    /// Returns timer resolution in seconds.
    pub fn getResolution() f64 {
        return c.MPI_Wtick();
    }
};

/// Work distribution helper for parallel computations.
/// Distributes work evenly across processes with load balancing.
pub const WorkDistribution = struct {
    start: i32,
    end: i32,
    count: i32,

    /// Distributes total_work items evenly across size processes.
    /// Process rank gets work items from start (inclusive) to end (exclusive).
    /// Assumes total_work >= 0, rank >= 0, size > 0, and rank < size.
    pub fn init(total_work: i32, rank: i32, size: i32) WorkDistribution {
        const work_per_proc = @divTrunc(total_work, size);
        const remainder = @rem(total_work, size);

        const start_idx = rank * work_per_proc + @min(rank, remainder);
        const work_count = work_per_proc + (if (rank < remainder) @as(i32, 1) else @as(i32, 0));
        const end_idx = start_idx + work_count;

        return WorkDistribution{
            .start = start_idx,
            .end = end_idx,
            .count = work_count,
        };
    }
};

/// MPI Status wrapper with convenience methods for status inspection.
/// Provides type-safe access to MPI_Status fields and operations.
pub const Status = struct {
    status: c.MPI_Status,

    /// Returns source rank of received message.
    pub fn getSource(self: Status) i32 {
        return self.status.MPI_SOURCE;
    }

    /// Returns tag of received message.
    pub fn getTag(self: Status) i32 {
        return self.status.MPI_TAG;
    }

    /// Returns error code associated with message reception.
    pub fn getError(self: Status) i32 {
        return self.status.MPI_ERROR;
    }

    /// Returns number of elements received for given datatype.
    /// Assumes T matches the datatype used in the original communication operation.
    pub fn getCount(self: Status, comptime T: type) MpiError!i32 {
        var count: c_int = undefined;
        try checkMpiResult(c.MPI_Get_count(&self.status, getMpiDatatype(T), &count));
        return count;
    }

    /// Checks if the associated operation was cancelled.
    pub fn isCancelled(self: Status) MpiError!bool {
        var flag: c_int = undefined;
        try checkMpiResult(c.MPI_Test_cancelled(&self.status, &flag));
        return flag != 0;
    }
};

/// MPI Request handle for non-blocking operations.
/// Manages lifecycle of asynchronous MPI communications.
pub const Request = struct {
    request: c.MPI_Request,

    pub const null_request = Request{ .request = c.MPI_REQUEST_NULL };

    /// Blocks until the request completes and returns status information.
    pub fn wait(self: *Request) MpiError!Status {
        var status: c.MPI_Status = undefined;
        try checkMpiResult(c.MPI_Wait(&self.request, &status));
        return Status{ .status = status };
    }

    pub fn testRequest(self: *Request) MpiError!?Status {
        var flag: c_int = undefined;
        var status: c.MPI_Status = undefined;
        try checkMpiResult(c.MPI_Test(&self.request, &flag, &status));
        if (flag != 0) {
            return Status{ .status = status };
        }
        return null;
    }

    pub fn cancel(self: *Request) MpiError!void {
        try checkMpiResult(c.MPI_Cancel(&self.request));
    }

    pub fn isComplete(self: *Request) MpiError!bool {
        return (try self.testRequest()) != null;
    }
};

/// Communication parameters for send/receive operations
pub const CommParams = struct {
    dest_or_source: i32,
    tag: i32 = 0,

    pub fn dest(rank: i32) CommParams {
        return CommParams{ .dest_or_source = rank };
    }

    pub fn source(rank: i32) CommParams {
        return CommParams{ .dest_or_source = rank };
    }

    pub fn withTag(self: CommParams, new_tag: i32) CommParams {
        return CommParams{ .dest_or_source = self.dest_or_source, .tag = new_tag };
    }

    pub const any_source = CommParams{ .dest_or_source = c.MPI_ANY_SOURCE };
    pub const any_tag = -1; // Use with withTag
};

/// Platform-specific MPI constants abstraction
const MpiConstants = struct {
    pub fn getCommWorld() c.MPI_Comm {
        if (builtin.os.tag == .windows) {
            // Windows/MSMPI uses direct constants
            return c.MPI_COMM_WORLD;
        } else {
            // OpenMPI on Linux uses global variables
            return @ptrCast(&c.ompi_mpi_comm_world);
        }
    }
    
    pub fn getCommSelf() c.MPI_Comm {
        if (builtin.os.tag == .windows) {
            return c.MPI_COMM_SELF;
        } else {
            return @ptrCast(&c.ompi_mpi_comm_self);
        }
    }
};

/// Main MPI Communicator with improved API
pub const Communicator = struct {
    comm: c.MPI_Comm,

    pub const world = Communicator{ .comm = MpiConstants.getCommWorld() };
    pub const self_comm = Communicator{ .comm = MpiConstants.getCommSelf() };

    /// Get process rank
    pub fn getRank(self: Communicator) MpiError!i32 {
        var process_rank: c_int = undefined;
        try checkMpiResult(c.MPI_Comm_rank(self.comm, &process_rank));
        return process_rank;
    }

    /// Get communicator size
    pub fn getSize(self: Communicator) MpiError!i32 {
        var process_count: c_int = undefined;
        try checkMpiResult(c.MPI_Comm_size(self.comm, &process_count));
        return process_count;
    }

    /// Barrier synchronization
    pub fn barrier(self: Communicator) MpiError!void {
        try checkMpiResult(c.MPI_Barrier(self.comm));
    }

    /// Send data (improved API with CommParams)
    pub fn send(self: Communicator, comptime T: type, data: []const T, params: CommParams) MpiError!void {
        try checkMpiResult(c.MPI_Send(
            data.ptr,
            @intCast(data.len),
            getMpiDatatype(T),
            params.dest_or_source,
            params.tag,
            self.comm,
        ));
    }

    /// Send single value
    pub fn sendValue(self: Communicator, comptime T: type, value: T, params: CommParams) MpiError!void {
        const data = [_]T{value};
        try self.send(T, &data, params);
    }

    /// Receive data (improved API with CommParams)
    pub fn recv(self: Communicator, comptime T: type, buffer: []T, params: CommParams) MpiError!Status {
        var status: c.MPI_Status = undefined;
        try checkMpiResult(c.MPI_Recv(
            buffer.ptr,
            @intCast(buffer.len),
            getMpiDatatype(T),
            params.dest_or_source,
            params.tag,
            self.comm,
            &status,
        ));
        return Status{ .status = status };
    }

    /// Receive single value
    pub fn recvValue(self: Communicator, comptime T: type, params: CommParams) MpiError!struct { value: T, status: Status } {
        const value: T = undefined;
        var data = [_]T{value};
        const status = try self.recv(T, &data, params);
        return .{ .value = data[0], .status = status };
    }

    /// Non-blocking send
    pub fn isend(self: Communicator, comptime T: type, data: []const T, params: CommParams) MpiError!Request {
        var request: c.MPI_Request = undefined;
        try checkMpiResult(c.MPI_Isend(
            data.ptr,
            @intCast(data.len),
            getMpiDatatype(T),
            params.dest_or_source,
            params.tag,
            self.comm,
            &request,
        ));
        return Request{ .request = request };
    }

    /// Non-blocking send single value
    pub fn isendValue(self: Communicator, comptime T: type, value: T, params: CommParams) MpiError!Request {
        // Note: This requires the value to remain valid until the request completes
        const data = [_]T{value};
        return self.isend(T, &data, params);
    }

    /// Non-blocking receive
    pub fn irecv(self: Communicator, comptime T: type, buffer: []T, params: CommParams) MpiError!Request {
        var request: c.MPI_Request = undefined;
        try checkMpiResult(c.MPI_Irecv(
            buffer.ptr,
            @intCast(buffer.len),
            getMpiDatatype(T),
            params.dest_or_source,
            params.tag,
            self.comm,
            &request,
        ));
        return Request{ .request = request };
    }

    /// Broadcast from root
    pub fn bcast(self: Communicator, comptime T: type, data: []T, root: i32) MpiError!void {
        try checkMpiResult(c.MPI_Bcast(
            data.ptr,
            @intCast(data.len),
            getMpiDatatype(T),
            root,
            self.comm,
        ));
    }

    /// Broadcast single value
    pub fn bcastValue(self: Communicator, comptime T: type, value: *T, root: i32) MpiError!void {
        var data = [_]T{value.*};
        try self.bcast(T, &data, root);
        value.* = data[0];
    }

    /// Reduce operation
    pub fn reduce(self: Communicator, comptime T: type, send_data: []const T, recv_data: ?[]T, operation: Operation, root: i32) MpiError!void {
        const recv_ptr = if (recv_data) |buf| buf.ptr else null;
        try checkMpiResult(c.MPI_Reduce(
            send_data.ptr,
            recv_ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            operation.toC(),
            root,
            self.comm,
        ));
    }

    /// All-reduce operation
    pub fn allReduce(self: Communicator, comptime T: type, send_data: []const T, recv_data: []T, operation: Operation) MpiError!void {
        try checkMpiResult(c.MPI_Allreduce(
            send_data.ptr,
            recv_data.ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            operation.toC(),
            self.comm,
        ));
    }

    /// Gather data to root
    pub fn gather(self: Communicator, comptime T: type, send_data: []const T, recv_data: ?[]T, root: i32) MpiError!void {
        const recv_ptr = if (recv_data) |buf| buf.ptr else null;
        try checkMpiResult(c.MPI_Gather(
            send_data.ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            recv_ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            root,
            self.comm,
        ));
    }

    /// All-gather data
    pub fn allGather(self: Communicator, comptime T: type, send_data: []const T, recv_data: []T) MpiError!void {
        try checkMpiResult(c.MPI_Allgather(
            send_data.ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            recv_data.ptr,
            @intCast(send_data.len),
            getMpiDatatype(T),
            self.comm,
        ));
    }

    /// Scatter data from root
    pub fn scatter(self: Communicator, comptime T: type, send_data: ?[]const T, recv_data: []T, root: i32) MpiError!void {
        const send_ptr = if (send_data) |buf| buf.ptr else null;
        try checkMpiResult(c.MPI_Scatter(
            send_ptr,
            @intCast(recv_data.len),
            getMpiDatatype(T),
            recv_data.ptr,
            @intCast(recv_data.len),
            getMpiDatatype(T),
            root,
            self.comm,
        ));
    }

    /// Get processor name
    pub fn getProcessorName(self: Communicator, allocator: std.mem.Allocator) (MpiError || std.mem.Allocator.Error)![]u8 {
        _ = self; // Not used but kept for API consistency
        var name_buf: [c.MPI_MAX_PROCESSOR_NAME]u8 = undefined;
        var result_len: c_int = undefined;
        try checkMpiResult(c.MPI_Get_processor_name(&name_buf, &result_len));

        return try allocator.dupe(u8, name_buf[0..@intCast(result_len)]);
    }

    /// Distribute work evenly among processes
    pub fn distributeWork(self: Communicator, total_work: i32) MpiError!WorkDistribution {
        const rank = try self.getRank();
        const size = try self.getSize();
        return WorkDistribution.init(total_work, rank, size);
    }
};

/// MPI Environment management
pub const Environment = struct {
    is_initialized: bool = false,
    thread_support: ?ThreadSupport = null,

    /// Initialize MPI environment
    pub fn init(self: *Environment) MpiError!void {
        try checkMpiResult(c.MPI_Init(null, null));
        self.is_initialized = true;
        self.thread_support = .single;
    }

    /// Initialize with thread support
    pub fn initWithThreads(self: *Environment, required: ThreadSupport) MpiError!ThreadSupport {
        var provided: c_int = undefined;
        try checkMpiResult(c.MPI_Init_thread(null, null, @intFromEnum(required), &provided));
        self.is_initialized = true;
        const provided_support: ThreadSupport = @enumFromInt(provided);
        self.thread_support = provided_support;
        return provided_support;
    }

    /// Finalize MPI environment
    pub fn deinit(self: *Environment) MpiError!void {
        if (self.is_initialized) {
            try checkMpiResult(c.MPI_Finalize());
            self.is_initialized = false;
            self.thread_support = null;
        }
    }

    /// Get MPI version information
    pub fn getVersion() MpiError!struct { version: i32, subversion: i32 } {
        var version: c_int = undefined;
        var subversion: c_int = undefined;
        try checkMpiResult(c.MPI_Get_version(&version, &subversion));
        return .{ .version = version, .subversion = subversion };
    }

    /// Check if MPI is initialized
    pub fn isInitialized() MpiError!bool {
        var flag: c_int = undefined;
        try checkMpiResult(c.MPI_Initialized(&flag));
        return flag != 0;
    }

    /// Check if MPI is finalized
    pub fn isFinalized() MpiError!bool {
        var flag: c_int = undefined;
        try checkMpiResult(c.MPI_Finalized(&flag));
        return flag != 0;
    }
};

/// High-level convenience functions
pub const convenience = struct {
    /// Parallel sum reduction (idiomatic helper)
    pub fn parallelSum(comptime T: type, local_value: T, comm: Communicator) MpiError!T {
        const local_data = [_]T{local_value};
        var global_data = [_]T{undefined};

        try comm.allReduce(T, &local_data, &global_data, .sum);
        return global_data[0];
    }

    /// Parallel maximum
    pub fn parallelMax(comptime T: type, local_value: T, comm: Communicator) MpiError!T {
        const local_data = [_]T{local_value};
        var global_data = [_]T{undefined};

        try comm.allReduce(T, &local_data, &global_data, .max);
        return global_data[0];
    }

    /// Parallel minimum
    pub fn parallelMin(comptime T: type, local_value: T, comm: Communicator) MpiError!T {
        const local_data = [_]T{local_value};
        var global_data = [_]T{undefined};

        try comm.allReduce(T, &local_data, &global_data, .min);
        return global_data[0];
    }

    /// Simple point-to-point exchange between two processes
    pub fn exchange(comptime T: type, comm: Communicator, send_data: []const T, recv_buffer: []T, partner_rank: i32, tag: i32) MpiError!Status {
        const rank = try comm.getRank();

        if (rank < partner_rank) {
            // Send first, then receive
            try comm.send(T, send_data, CommParams.dest(partner_rank).withTag(tag));
            return comm.recv(T, recv_buffer, CommParams.source(partner_rank).withTag(tag));
        } else {
            // Receive first, then send
            const status = try comm.recv(T, recv_buffer, CommParams.source(partner_rank).withTag(tag));
            try comm.send(T, send_data, CommParams.dest(partner_rank).withTag(tag));
            return status;
        }
    }

    /// Timing utilities
    pub const timing = struct {
        /// Time a function execution
        pub fn timeFunction(func: anytype, args: anytype) struct { result: @TypeOf(@call(.auto, func, args)), time: f64 } {
            const timer = Timer.start();
            const result = @call(.auto, func, args);
            const elapsed_time = timer.elapsed();
            return .{ .result = result, .time = elapsed_time };
        }

        /// Synchronized timing across all processes
        pub fn syncTime(comm: Communicator, func: anytype, args: anytype) MpiError!struct { result: @TypeOf(@call(.auto, func, args)), max_time: f64, min_time: f64, avg_time: f64 } {
            try comm.barrier();
            const timer = Timer.start();
            const result = @call(.auto, func, args);
            try comm.barrier();
            const elapsed_time = timer.elapsed();

            const max_time = try convenience.parallelMax(f64, elapsed_time, comm);
            const min_time = try convenience.parallelMin(f64, elapsed_time, comm);
            const sum_time = try convenience.parallelSum(f64, elapsed_time, comm);
            const size = try comm.getSize();
            const avg_time = sum_time / @as(f64, @floatFromInt(size));

            return .{ .result = result, .max_time = max_time, .min_time = min_time, .avg_time = avg_time };
        }
    };
};

/// Platform-specific optimizations and compatibility
pub const platform = struct {
    /// Detect MPI implementation at compile time if possible
    pub fn detectMpiImplementation() []const u8 {
        // Try to detect based on compile-time defines
        if (@hasDecl(c, "OPEN_MPI")) {
            return "OpenMPI";
        } else if (@hasDecl(c, "MPICH_VERSION")) {
            return "MPICH";
        } else if (@hasDecl(c, "I_MPI_VERSION")) {
            return "Intel MPI";
        } else if (builtin.os.tag == .windows) {
            return "Microsoft MPI";
        } else {
            return "Unknown";
        }
    }

    /// Get implementation-specific optimizations
    pub fn getOptimizations() struct { supports_cuda: bool, supports_rdma: bool, max_message_size: usize } {
        const impl = detectMpiImplementation();
        if (std.mem.eql(u8, impl, "OpenMPI")) {
            return .{
                .supports_cuda = true,
                .supports_rdma = true,
                .max_message_size = 2 * 1024 * 1024 * 1024 - 1, // ~2GB
            };
        } else if (std.mem.eql(u8, impl, "MPICH")) {
            return .{
                .supports_cuda = false,
                .supports_rdma = true,
                .max_message_size = 1024 * 1024 * 1024, // 1GB
            };
        } else {
            return .{
                .supports_cuda = false,
                .supports_rdma = false,
                .max_message_size = 64 * 1024 * 1024, // 64MB conservative
            };
        }
    }
};

/// Request management for multiple non-blocking operations
pub const RequestManager = struct {
    requests: std.ArrayList(Request),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) RequestManager {
        return RequestManager{
            .requests = std.ArrayList(Request){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RequestManager) void {
        self.requests.deinit(self.allocator);
    }

    pub fn add(self: *RequestManager, request: Request) !void {
        try self.requests.append(self.allocator, request);
    }

    pub fn waitAll(self: *RequestManager) !std.ArrayList(Status) {
        var statuses = std.ArrayList(Status){};
        try statuses.ensureTotalCapacity(self.allocator, self.requests.items.len);

        for (self.requests.items) |*request| {
            const status = try request.wait();
            statuses.appendAssumeCapacity(status);
        }

        self.requests.clearRetainingCapacity();
        return statuses;
    }

    pub fn testAll(self: *RequestManager) !?std.ArrayList(Status) {
        var statuses = std.ArrayList(Status){};
        try statuses.ensureTotalCapacity(self.allocator, self.requests.items.len);

        var all_complete = true;
        for (self.requests.items) |*request| {
            if (try request.testRequest()) |status| {
                statuses.appendAssumeCapacity(status);
            } else {
                all_complete = false;
                break;
            }
        }

        if (all_complete) {
            self.requests.clearRetainingCapacity();
            return statuses;
        } else {
            statuses.deinit(self.allocator);
            return null;
        }
    }
};

/// Testing utilities
pub const testing = struct {
    /// Simple MPI test harness
    pub fn expectMpiSuccess(result: anytype) !void {
        if (@TypeOf(result) == MpiError!void) {
            try result;
        } else {
            _ = try result;
        }
    }

    /// Test if we're in a multi-process environment
    pub fn isMultiProcess(comm: Communicator) !bool {
        const size = try comm.getSize();
        return size > 1;
    }
};

// Compile-time tests
test "MPI datatype mapping" {
    _ = getMpiDatatype(i32);
    _ = getMpiDatatype(f64);
    _ = getMpiDatatype(bool);
}

test "Timer functionality" {
    const timer = Timer.start();
    std.Thread.sleep(1000); // 1 microsecond
    const elapsed = timer.elapsed();
    try std.testing.expect(elapsed >= 0.0);
}

test "Work distribution" {
    const work = WorkDistribution.init(100, 0, 4);
    try std.testing.expect(work.count > 0);
    try std.testing.expect(work.start >= 0);
    try std.testing.expect(work.end > work.start);
}

test "Communication parameters" {
    const params = CommParams.dest(5).withTag(42);
    try std.testing.expect(params.dest_or_source == 5);
    try std.testing.expect(params.tag == 42);
}
