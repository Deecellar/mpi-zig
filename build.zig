// build.zig - Build configuration for Zig 0.15+ MPI Wrapper
// Supports OpenMPI, Microsoft MPI, and MPICH with platform detection

const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Standard target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // MPI configuration options
    const mpi_path = b.option([]const u8, "mpi-path", "Path to MPI installation") orelse detectMpiPath(b);
    const mpi_implementation = b.option([]const u8, "mpi-impl", "MPI implementation (openmpi, msmpi, mpich, intel)") orelse detectMpiImplementation(b, target.result);
    const enable_cuda = b.option(bool, "enable-cuda", "Enable CUDA-aware MPI support") orelse false;
    const enable_profiling = b.option(bool, "enable-profiling", "Enable MPI profiling interface") orelse false;

    // Create the main MPI wrapper module
    const mpi_module = b.createModule(.{
        .root_source_file = b.path("src/mpi_wrapper.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create MPI library
    const mpi_lib = b.addLibrary(.{
        .name = "zig-mpi",
        .root_module = mpi_module,
        .linkage = .static,
    });

    // Configure MPI linking and includes
    configureMpi(b, mpi_lib.root_module, mpi_path, mpi_implementation, enable_cuda, enable_profiling);
    b.installArtifact(mpi_lib);

    // Example programs with improved organization
    const examples = [_]struct { name: []const u8, file: []const u8, description: []const u8 }{
        .{ .name = "hello", .file = "examples/01_hello_world.zig", .description = "Basic MPI hello world" },
        .{ .name = "point-to-point", .file = "examples/02_point_to_point.zig", .description = "Send/receive communication" },
        .{ .name = "non-blocking", .file = "examples/03_non_blocking.zig", .description = "Asynchronous communication" },
        .{ .name = "collective", .file = "examples/04_collective_ops.zig", .description = "Broadcast, reduce, gather operations" },
        .{ .name = "monte-carlo", .file = "examples/05_monte_carlo_pi.zig", .description = "Parallel Monte Carlo Pi calculation" },
        .{ .name = "exchange", .file = "examples/06_process_exchange.zig", .description = "Process pair exchange pattern" },
        .{ .name = "benchmark", .file = "examples/07_benchmark.zig", .description = "Performance benchmarking" },
        .{ .name = "error-handling", .file = "examples/08_error_handling.zig", .description = "Error handling demonstration" },
        .{ .name = "advanced", .file = "examples/09_advanced_patterns.zig", .description = "Advanced MPI patterns" },
    };

    for (examples) |example| {
        const exe_module = b.createModule(.{
            .root_source_file = b.path(example.file),
            .target = target,
            .optimize = optimize,
        });
        const exe = b.addExecutable(.{
            .name = b.fmt("mpi-example-{s}", .{example.name}),
            .root_module = exe_module,
        });

        // Link MPI and add module
        configureMpi(b, exe.root_module, mpi_path, mpi_implementation, enable_cuda, enable_profiling);
        exe.root_module.addImport("mpi", mpi_module);
        b.installArtifact(exe);

        // Create run step for single-process execution
        const run_cmd = b.addRunArtifact(exe);
        if (b.args) |args| run_cmd.addArgs(args);

        const run_step = b.step(b.fmt("run-{s}", .{example.name}), b.fmt("Run {s} example (single process)", .{example.description}));
        run_step.dependOn(&run_cmd.step);

        // Create MPI run steps for different process counts
        for ([_]u8{ 2, 4, 8 }) |np| {
            const mpi_run_cmd = b.addSystemCommand(&[_][]const u8{ "mpirun", "-np", b.fmt("{}", .{np}) });
            mpi_run_cmd.addArtifactArg(exe);
            if (b.args) |args| mpi_run_cmd.addArgs(args);

            const mpi_run_step = b.step(b.fmt("mpi{}-{s}", .{ np, example.name }), b.fmt("Run {s} with mpirun -np {}", .{ example.description, np }));
            mpi_run_step.dependOn(b.getInstallStep());
            mpi_run_step.dependOn(&mpi_run_cmd.step);
        }
    }

    // Unit tests
    const unit_tests = b.addTest(.{
        .name = "mpi-tests",
        .root_module = mpi_module,
    });
    configureMpi(b, unit_tests.root_module, mpi_path, mpi_implementation, enable_cuda, enable_profiling);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    // Documentation generation
    const docs_step = b.step("docs", "Generate documentation");
    const docs_obj = b.addObject(.{
        .name = "mpi-docs",
        .root_module = mpi_module,
    });
    configureMpi(b, docs_obj.root_module, mpi_path, mpi_implementation, enable_cuda, enable_profiling);

    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);

    // Utility commands
    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.dependOn(&b.addRemoveDirTree(.{ .cwd_relative = b.install_prefix }).step);
}

/// Configure MPI linking based on implementation and platform
fn configureMpi(b: *std.Build, module: *std.Build.Module, mpi_path: []const u8, mpi_impl: []const u8, enable_cuda: bool, enable_profiling: bool) void {
    // Add MPI include path (handle Windows case sensitivity)
    const include_path_unix = b.fmt("{s}/include", .{mpi_path});
    const include_path_windows = b.fmt("{s}/Include", .{mpi_path});

    // Try both variants and use the one that exists
    if (pathExists(b, include_path_windows)) {
        module.addIncludePath(.{ .cwd_relative = include_path_windows });
    } else {
        module.addIncludePath(.{ .cwd_relative = include_path_unix });
    }

    // Add MPI library path (handle Windows architecture-specific folders)
    const host_target = builtin.target;
    if (host_target.os.tag == .windows) {
        // Microsoft MPI uses x86/x64 subdirectories for libraries
        const arch_suffix = if (host_target.cpu.arch == .x86_64) "x64" else "x86";

        const lib_paths_to_try = [_][]const u8{
            b.fmt("{s}/Lib/{s}", .{ mpi_path, arch_suffix }), // Standard MSMPI layout
            b.fmt("{s}/lib/{s}", .{ mpi_path, arch_suffix }), // Alternative case
            b.fmt("{s}/Lib", .{mpi_path}), // Fallback without arch
            b.fmt("{s}/lib", .{mpi_path}), // Unix-style fallback
        };

        var lib_path_found = false;
        for (lib_paths_to_try) |lib_path| {
            if (pathExists(b, lib_path)) {
                module.addLibraryPath(.{ .cwd_relative = lib_path });
                lib_path_found = true;
                break;
            }
        }

        if (!lib_path_found) {
            // Use the most likely path as fallback
            const fallback_path = b.fmt("{s}/Lib/{s}", .{ mpi_path, arch_suffix });
            module.addLibraryPath(.{ .cwd_relative = fallback_path });
        }
    } else {
        // Unix systems - standard lib directories
        const lib_path_unix = b.fmt("{s}/lib", .{mpi_path});
        module.addLibraryPath(.{ .cwd_relative = lib_path_unix });
    }

    // Link system libc
    module.link_libc = true; // Implementation-specific configuration
    if (std.mem.eql(u8, mpi_impl, "openmpi")) {
        configureOpenMpi(module);
    } else if (std.mem.eql(u8, mpi_impl, "msmpi")) {
        configureMicrosoftMpi(module);
    } else if (std.mem.eql(u8, mpi_impl, "mpich")) {
        configureMpich(module);
    } else if (std.mem.eql(u8, mpi_impl, "intel")) {
        configureIntelMpi(module);
    } else {
        // Generic MPI configuration
        module.linkSystemLibrary("mpi", .{});
    }

    // Optional features
    if (enable_cuda) {
        module.addCMacro("ENABLE_CUDA_AWARE_MPI", "1");
        // Link CUDA runtime if available
        module.linkSystemLibrary("cudart", .{});
    }

    if (enable_profiling) {
        module.addCMacro("ENABLE_MPI_PROFILING", "1");
    }

    // Suppress C++ bindings warnings
    module.addCMacro("OMPI_SKIP_MPICXX", "1");
    module.addCMacro("MPICH_SKIP_MPICXX", "1");
}

/// Configure OpenMPI specifically
fn configureOpenMpi(module: *std.Build.Module) void {
    module.linkSystemLibrary("mpi", .{});

    // OpenMPI may need additional libraries

    // Add OpenMPI specific definitions
    module.addCMacro("OMPI_SKIP_MPICXX", "1");
}

/// Configure Microsoft MPI specifically
fn configureMicrosoftMpi(module: *std.Build.Module) void {
    // Microsoft MPI uses msmpi
    module.linkSystemLibrary("msmpi", .{});

    // Additional Windows-specific libraries
    module.linkSystemLibrary("kernel32", .{});
    module.linkSystemLibrary("user32", .{});
    module.linkSystemLibrary("ws2_32", .{});

    // Microsoft MPI definitions
    module.addCMacro("MSMPI_SKIP_MPICXX", "1");
}

/// Configure MPICH specifically
fn configureMpich(module: *std.Build.Module) void {
    module.linkSystemLibrary("mpi", .{});

    // MPICH may need additional libraries
    module.linkSystemLibrary("mpich", .{});
    module.linkSystemLibrary("mpl", .{});

    // MPICH specific definitions
    module.addCMacro("MPICH_SKIP_MPICXX", "1");
}

/// Configure Intel MPI specifically
fn configureIntelMpi(module: *std.Build.Module) void {
    module.linkSystemLibrary("mpi", .{});
    module.linkSystemLibrary("mpifort", .{});

    // Intel MPI definitions
    module.addCMacro("I_MPI_SKIP_MPICXX", "1");
}

/// Detect MPI installation path
fn detectMpiPath(b: *std.Build) []const u8 {
    const host_target = builtin.target;

    if (host_target.os.tag == .windows) {
        // Common Microsoft MPI paths
        // Build comprehensive list of potential Microsoft MPI installation paths
        var msmpi_paths = std.ArrayList([]const u8){};
        defer msmpi_paths.deinit(b.allocator);

        // Standard Microsoft MPI installation paths
        const standard_paths = [_][]const u8{
            "C:\\Program Files\\Microsoft MPI",
            "C:\\Program Files (x86)\\Microsoft MPI",
            "C:\\Program Files\\Microsoft SDKs\\MPI",
            "C:\\Program Files (x86)\\Microsoft SDKs\\MPI",
        };

        // Environment variable paths
        const env_vars = [_][]const u8{ "MSMPI_ROOT", "MPI_ROOT", "ProgramFiles", "ProgramFiles(x86)" };
        for (env_vars) |env_var| {
            if (std.process.getEnvVarOwned(b.allocator, env_var)) |base_path| {
                defer b.allocator.free(base_path);

                if (std.mem.eql(u8, env_var, "ProgramFiles") or std.mem.eql(u8, env_var, "ProgramFiles(x86)")) {
                    // Construct MPI paths from Program Files
                    const mpi_suffixes = [_][]const u8{
                        "Microsoft MPI",
                        "Microsoft SDKs\\MPI",
                    };
                    for (mpi_suffixes) |suffix| {
                        const full_path = b.fmt("{s}\\{s}", .{ base_path, suffix });
                        msmpi_paths.append(b.allocator, b.dupe(full_path)) catch continue;
                    }
                } else {
                    msmpi_paths.append(b.allocator, b.dupe(base_path)) catch continue;
                }
            } else |_| {}
        }

        // Add all standard paths
        for (standard_paths) |path| {
            msmpi_paths.append(b.allocator, path) catch continue;
        }

        // Registry-based detection (Windows-specific)
        const registry_paths = detectMpiFromRegistry(b);
        for (registry_paths) |path| {
            msmpi_paths.append(b.allocator, path) catch continue;
        }

        // Check common version-specific paths
        const version_paths = [_][]const u8{
            "C:\\Program Files\\Microsoft MPI\\V10.1.1",
            "C:\\Program Files\\Microsoft MPI\\V10.0",
            "C:\\Program Files (x86)\\Microsoft MPI\\V10.1.1",
            "C:\\Program Files (x86)\\Microsoft MPI\\V10.0",
        };
        for (version_paths) |path| {
            msmpi_paths.append(b.allocator, path) catch continue;
        }

        // Convert to slice for iteration
        const paths_slice = msmpi_paths.toOwnedSlice(b.allocator) catch &[_][]const u8{};
        defer b.allocator.free(paths_slice);

        // Find first valid path with proper validation
        for (paths_slice) |path| {
            if (validateMpiInstallation(b, path)) return b.dupe(path);
        }

        for (msmpi_paths.items) |path| {
            if (pathExists(b, path)) return path;
        }

        // Check environment variable
        if (std.process.getEnvVarOwned(b.allocator, "MSMPI_ROOT")) |path| {
            defer b.allocator.free(path);
            return b.dupe(path);
        } else |_| {}

        return "C:\\Program Files\\Microsoft MPI"; // Default fallback
    } else {
        // Unix-like systems
        const unix_paths = [_][]const u8{
            "/usr/local/mpi",
            "/opt/mpi",
            "/usr/lib/x86_64-linux-gnu/openmpi",
            "/usr/lib/x86_64-linux-gnu/mpich",
            "/opt/intel/mpi",
            "/usr",
        };

        for (unix_paths) |path| {
            const header_path = b.fmt("{s}/include/mpi.h", .{path});
            if (pathExists(b, header_path)) return path;
        }

        // Check common environment variables
        if (std.process.getEnvVarOwned(b.allocator, "MPI_ROOT")) |path| {
            defer b.allocator.free(path);
            return b.dupe(path);
        } else |_| {}

        if (std.process.getEnvVarOwned(b.allocator, "OMPI_ROOT")) |path| {
            defer b.allocator.free(path);
            return b.dupe(path);
        } else |_| {}

        return "/usr"; // Default fallback for Unix
    }
}

/// Detect MPI implementation
fn detectMpiImplementation(b: *std.Build, target: std.Target) []const u8 {
    if (target.os.tag == .windows) {
        return "msmpi";
    } else {
        // Try to detect based on common installation patterns
        // This is a heuristic - in practice, users should specify explicitly
        if (pathExists(b, "/usr/include/openmpi")) {
            return "openmpi";
        } else if (pathExists(b, "/usr/include/mpich")) {
            return "mpich";
        } else if (pathExists(b, "/opt/intel/mpi")) {
            return "intel";
        } else {
            return "openmpi"; // Most common default on Unix
        }
    }
}

/// Check if a path exists
fn pathExists(b: *std.Build, path: []const u8) bool {
    _ = b;
    std.fs.accessAbsolute(path, .{}) catch return false;
    return true;
}

/// Validate MPI installation by checking for required files
fn validateMpiInstallation(b: *std.Build, path: []const u8) bool {
    // Check for essential MPI files
    const include_path = b.fmt("{s}/include/mpi.h", .{path});
    const include_path_alt = b.fmt("{s}/Include/mpi.h", .{path}); // Windows variant

    // Check if include directory exists
    if (!pathExists(b, include_path) and !pathExists(b, include_path_alt)) {
        return false;
    }

    // Check for library directory with Windows architecture support
    const host_target = builtin.target;
    if (host_target.os.tag == .windows) {
        const arch_suffix = if (host_target.cpu.arch == .x86_64) "x64" else "x86";

        const lib_paths_to_check = [_][]const u8{
            b.fmt("{s}/Lib/{s}", .{ path, arch_suffix }), // Standard MSMPI with arch
            b.fmt("{s}/lib/{s}", .{ path, arch_suffix }), // Alternative case with arch
            b.fmt("{s}/Lib", .{path}), // Standard without arch
            b.fmt("{s}/lib", .{path}), // Alternative case
        };

        // Check if at least one library path exists
        for (lib_paths_to_check) |lib_path| {
            if (pathExists(b, lib_path)) {
                return true;
            }
        }
        return false;
    } else {
        // Unix systems - standard lib directory
        const lib_path = b.fmt("{s}/lib", .{path});
        if (!pathExists(b, lib_path)) {
            return false;
        }
    }

    return true;
}

/// Detect MPI installation from Windows Registry
fn detectMpiFromRegistry(b: *std.Build) [][]const u8 {
    // On Windows, we can try to read registry entries for Microsoft MPI
    // This is a simplified version - in production you'd use proper Windows API calls

    var registry_paths = std.ArrayList([]const u8){};

    // Common registry-based paths where Microsoft MPI might be installed
    // These are typical locations that installers might use
    const potential_registry_paths = [_][]const u8{
        "C:\\Program Files\\Microsoft MPI",
        "C:\\Program Files (x86)\\Microsoft MPI",
        "C:\\Program Files\\Microsoft SDKs\\MPI",
        "C:\\Program Files (x86)\\Microsoft SDKs\\MPI",
    };

    // For now, we'll return the common paths as a fallback
    // In a full implementation, you would use Windows Registry API:
    // - HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\MPI
    // - HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\MPI
    // - Check for InstallRoot or similar keys

    for (potential_registry_paths) |path| {
        if (pathExists(b, path)) {
            registry_paths.append(b.allocator, b.dupe(path)) catch continue;
        }
    }

    return registry_paths.toOwnedSlice(b.allocator) catch &[_][]const u8{};
}

/// Cross-compilation support for various platforms
pub fn addCrossCompileTargets(b: *std.Build) void {
    const cross_targets = [_]std.Target.Query{
        .{ .cpu_arch = .x86_64, .os_tag = .linux },
        .{ .cpu_arch = .aarch64, .os_tag = .linux },
        .{ .cpu_arch = .x86_64, .os_tag = .windows },
        .{ .cpu_arch = .aarch64, .os_tag = .macos },
        .{ .cpu_arch = .x86_64, .os_tag = .freebsd },
    };

    for (cross_targets) |query| {
        const target = b.resolveTargetQuery(query);
        const target_name = b.fmt("{s}-{s}", .{ @tagName(query.cpu_arch.?), @tagName(query.os_tag.?) });

        const cross_module = b.createModule(.{
            .root_source_file = b.path("examples/01_hello_world.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        const exe = b.addExecutable(.{
            .name = b.fmt("mpi-example-{s}", .{target_name}),
            .root_module = cross_module,
        });

        // Note: Cross-compilation would require cross-compiled MPI libraries
        // This is a skeleton for future implementation
        exe.linkLibC();

        const install_step = b.step(b.fmt("cross-{s}", .{target_name}), b.fmt("Cross-compile for {s}", .{target_name}));
        install_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}

/// Development utilities
pub fn addDevUtilities(b: *std.Build) void {
    // Format all source code
    const fmt_step = b.step("fmt", "Format all source code");
    const fmt_cmd = b.addFmt(.{
        .paths = &.{
            "src",
            "examples",
            "tests",
            "benchmarks",
            "build.zig",
        },
    });
    fmt_step.dependOn(&fmt_cmd.step);

    // Check for common issues
    const check_step = b.step("check", "Check code for issues");
    const check_cmd = b.addSystemCommand(&[_][]const u8{ "zig", "ast-check", "src/mpi_wrapper.zig" });
    check_step.dependOn(&check_cmd.step);
}
