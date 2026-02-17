#!/usr/bin/env python3
"""
Container Throughput Profiler.

Finds optimal container configuration via bisection search.

Two optimization axes:
1. Workers per container: How many parallel questions within one container
2. Number of containers: Bisection search to find max before resource exhaustion

Usage:
    uv run python scripts/profile_throughput.py
    uv run python scripts/profile_throughput.py --quick    # Skip macro benchmark
    uv run python scripts/profile_throughput.py --save     # Save optimal config
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.container_pool import MultiTenantContainer
from src.utils.docker import cleanup_csv_sandbox_containers


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    n_containers: int
    n_workers: int
    startup_time: float
    execution_time: float
    reset_time: float
    memory_mb: float
    success: bool
    error: Optional[str] = None
    
    @property
    def total_time(self) -> float:
        return self.startup_time + self.execution_time + self.reset_time
    
    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.execution_time <= 0:
            return 0
        return self.n_workers / self.execution_time


@dataclass
class ProfilerResults:
    """Complete profiler results."""
    system_info: dict
    worker_scaling: list[BenchmarkResult] = field(default_factory=list)
    container_scaling: list[BenchmarkResult] = field(default_factory=list)
    optimal_workers: int = 6
    optimal_containers: int = 1
    
    def to_dict(self) -> dict:
        return {
            "system_info": self.system_info,
            "optimal_workers": self.optimal_workers,
            "optimal_containers": self.optimal_containers,
            "worker_scaling": [
                {"n_workers": r.n_workers, "throughput": r.throughput, "memory_mb": r.memory_mb}
                for r in self.worker_scaling
            ],
            "container_scaling": [
                {"n_containers": r.n_containers, "success": r.success, "total_time": r.total_time}
                for r in self.container_scaling
            ],
        }


def get_system_info() -> dict:
    """Get system resource information."""
    import platform
    
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    # CPU cores
    info["cpu_cores"] = os.cpu_count() or 1
    
    # Memory (cross-platform)
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            info["memory_bytes"] = int(result.stdout.strip())
            info["memory_gb"] = info["memory_bytes"] / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["memory_bytes"] = kb * 1024
                        info["memory_gb"] = kb / (1024**2)
                        break
        else:
            info["memory_gb"] = 8  # Fallback
    except Exception:
        info["memory_gb"] = 8  # Fallback
    
    # Docker info
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],
            capture_output=True, text=True
        )
        docker_mem = int(result.stdout.strip())
        info["docker_memory_gb"] = docker_mem / (1024**3)
    except Exception:
        info["docker_memory_gb"] = info.get("memory_gb", 8)
    
    return info


def get_container_memory_mb(container_id: str) -> float:
    """Get memory usage of a container in MB."""
    try:
        result = subprocess.run(
            ["docker", "stats", container_id, "--no-stream", "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10
        )
        # Parse "123MiB / 8GiB" format
        mem_str = result.stdout.strip().split("/")[0].strip()
        if "GiB" in mem_str:
            return float(mem_str.replace("GiB", "")) * 1024
        elif "MiB" in mem_str:
            return float(mem_str.replace("MiB", ""))
        elif "KiB" in mem_str:
            return float(mem_str.replace("KiB", "")) / 1024
        else:
            return 0
    except Exception:
        return 0


async def benchmark_worker_scaling(
    csv_path: str,
    worker_counts: list[int],
) -> list[BenchmarkResult]:
    """Benchmark different worker counts within a single container."""
    results = []
    
    for n_workers in worker_counts:
        print(f"\n  Testing {n_workers} workers...")
        
        try:
            # Startup
            start = time.time()
            container = MultiTenantContainer(csv_path, n_workers=n_workers)
            await container.start()
            startup_time = time.time() - start
            
            # Get memory
            memory_mb = get_container_memory_mb(container.container_id)
            
            # Execute code on all workers in parallel
            test_code = "result = df.shape[0] * 2; result"
            start = time.time()
            await asyncio.gather(*[
                container.run_on_worker(i, test_code)
                for i in range(n_workers)
            ])
            execution_time = time.time() - start
            
            # Reset all workers
            start = time.time()
            await container.reset_all_workers()
            reset_time = time.time() - start
            
            # Cleanup
            await container.stop()
            
            result = BenchmarkResult(
                n_containers=1,
                n_workers=n_workers,
                startup_time=startup_time,
                execution_time=execution_time,
                reset_time=reset_time,
                memory_mb=memory_mb,
                success=True,
            )
            results.append(result)
            
            print(f"    ✓ {n_workers} workers: {execution_time:.2f}s exec, {memory_mb:.0f}MB, "
                  f"throughput={result.throughput:.1f} ops/s")
            
        except Exception as e:
            print(f"    ✗ {n_workers} workers failed: {e}")
            results.append(BenchmarkResult(
                n_containers=1,
                n_workers=n_workers,
                startup_time=0,
                execution_time=0,
                reset_time=0,
                memory_mb=0,
                success=False,
                error=str(e),
            ))
    
    return results


async def test_n_containers(csv_path: str, n_containers: int, n_workers: int = 6) -> BenchmarkResult:
    """Test if we can run N containers simultaneously."""
    containers = []
    
    try:
        # Start all containers
        start = time.time()
        for i in range(n_containers):
            container = MultiTenantContainer(csv_path, n_workers=n_workers)
            await container.start()
            containers.append(container)
        startup_time = time.time() - start
        
        # Get total memory
        total_memory = sum(
            get_container_memory_mb(c.container_id) for c in containers
        )
        
        # Run code on first worker of each container
        test_code = "result = df.shape[0] * 2; result"
        start = time.time()
        await asyncio.gather(*[
            c.run_on_worker(0, test_code) for c in containers
        ])
        execution_time = time.time() - start
        
        # Reset
        start = time.time()
        await asyncio.gather(*[c.reset_all_workers() for c in containers])
        reset_time = time.time() - start
        
        return BenchmarkResult(
            n_containers=n_containers,
            n_workers=n_workers,
            startup_time=startup_time,
            execution_time=execution_time,
            reset_time=reset_time,
            memory_mb=total_memory,
            success=True,
        )
        
    except Exception as e:
        return BenchmarkResult(
            n_containers=n_containers,
            n_workers=n_workers,
            startup_time=0,
            execution_time=0,
            reset_time=0,
            memory_mb=0,
            success=False,
            error=str(e),
        )
    finally:
        # Cleanup
        for c in containers:
            try:
                await c.stop()
            except Exception:
                pass


async def bisection_container_search(
    csv_path: str,
    n_workers: int = 6,
    min_containers: int = 1,
    max_containers: int = 32,
) -> tuple[int, list[BenchmarkResult]]:
    """
    Binary search to find max containers that work reliably.
    
    Returns (optimal_count, all_results).
    """
    results = []
    
    print(f"\n  Bisection search: min={min_containers}, max={max_containers}")
    
    # First, test min to ensure baseline works
    print(f"\n  Testing baseline ({min_containers} containers)...")
    result = await test_n_containers(csv_path, min_containers, n_workers)
    results.append(result)
    
    if not result.success:
        print("    ✗ Baseline failed! Check Docker resources.")
        return 0, results
    
    print(f"    ✓ Baseline OK: {result.startup_time:.1f}s startup, {result.memory_mb:.0f}MB")
    
    # Binary search
    low, high = min_containers, max_containers
    best_working = min_containers
    
    while low <= high:
        mid = (low + high) // 2
        
        if mid == best_working:
            # Already tested this value
            low = mid + 1
            continue
        
        print(f"\n  Testing {mid} containers...")
        cleanup_csv_sandbox_containers()  # Clean slate
        await asyncio.sleep(1)
        
        result = await test_n_containers(csv_path, mid, n_workers)
        results.append(result)
        
        if result.success:
            print(f"    ✓ {mid} containers OK: {result.startup_time:.1f}s startup, "
                  f"{result.memory_mb:.0f}MB total")
            best_working = mid
            low = mid + 1
        else:
            print(f"    ✗ {mid} containers failed: {result.error}")
            high = mid - 1
        
        cleanup_csv_sandbox_containers()
    
    return best_working, results


def find_optimal_workers(results: list[BenchmarkResult]) -> int:
    """Find optimal worker count based on throughput curve."""
    successful = [r for r in results if r.success]
    if not successful:
        return 6  # Default
    
    # Find point of diminishing returns (throughput gain < 20%)
    prev_throughput = 0
    for r in sorted(successful, key=lambda x: x.n_workers):
        if prev_throughput > 0:
            improvement = (r.throughput - prev_throughput) / prev_throughput
            if improvement < 0.2:  # Less than 20% improvement
                return r.n_workers
        prev_throughput = r.throughput
    
    # If still scaling well, return highest tested
    return max(r.n_workers for r in successful)


async def run_profiler(
    csv_path: str,
    quick: bool = False,
    max_containers_to_test: int = 32,
) -> ProfilerResults:
    """Run the full profiler."""
    
    print("=" * 60)
    print("Container Throughput Profiler")
    print("=" * 60)
    
    # System info
    print("\n📊 System Information:")
    system_info = get_system_info()
    print(f"  CPU cores: {system_info['cpu_cores']}")
    print(f"  System RAM: {system_info.get('memory_gb', 'unknown'):.1f} GB")
    print(f"  Docker RAM: {system_info.get('docker_memory_gb', 'unknown'):.1f} GB")
    
    results = ProfilerResults(system_info=system_info)
    
    # Clean up any existing containers
    print("\n🧹 Cleaning up existing containers...")
    cleanup_csv_sandbox_containers()
    
    # Worker scaling benchmark
    print("\n📈 Phase 1: Worker Scaling (within single container)")
    worker_counts = [2, 4, 6, 8, 12, 16] if not quick else [4, 8]
    results.worker_scaling = await benchmark_worker_scaling(csv_path, worker_counts)
    results.optimal_workers = find_optimal_workers(results.worker_scaling)
    
    # Container scaling benchmark (bisection)
    print("\n📈 Phase 2: Container Scaling (bisection search)")
    cleanup_csv_sandbox_containers()
    
    # Estimate max based on memory
    memory_per_container = 500  # MB estimate
    docker_mem_mb = system_info.get("docker_memory_gb", 8) * 1024
    estimated_max = max(1, int(docker_mem_mb / memory_per_container))
    max_to_test = min(max_containers_to_test, estimated_max * 2)
    
    results.optimal_containers, results.container_scaling = await bisection_container_search(
        csv_path,
        n_workers=results.optimal_workers,
        min_containers=1,
        max_containers=max_to_test,
    )
    
    # Final cleanup
    cleanup_csv_sandbox_containers()
    
    return results


def print_summary(results: ProfilerResults):
    """Print summary and recommendations."""
    print("\n" + "=" * 60)
    print("📋 PROFILER RESULTS")
    print("=" * 60)
    
    print("\n🎯 Optimal Configuration:")
    print(f"  n_workers per container: {results.optimal_workers}")
    print(f"  max_concurrent_containers: {results.optimal_containers}")
    
    print("\n📊 Worker Scaling:")
    for r in sorted(results.worker_scaling, key=lambda x: x.n_workers):
        if r.success:
            print(f"  {r.n_workers:2d} workers: {r.throughput:.1f} ops/s, {r.memory_mb:.0f}MB")
    
    print("\n📊 Container Scaling:")
    for r in sorted(results.container_scaling, key=lambda x: x.n_containers):
        status = "✓" if r.success else "✗"
        print(f"  {r.n_containers:2d} containers: {status}")
    
    print("\n💡 Recommendation:")
    print("  Update config.py with:")
    print(f"    n_consistency: {results.optimal_workers - 1}")
    print(f"    max_concurrent_containers: {results.optimal_containers}")


def save_config(results: ProfilerResults):
    """Save optimal config to a file."""
    config_file = Path("configs/optimal_throughput.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "n_workers": results.optimal_workers,
        "max_concurrent_containers": results.optimal_containers,
        "system_info": results.system_info,
        "profiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Saved optimal config to: {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile container throughput")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV file to use for benchmarking (uses first from data/kaggle/ if not specified)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer test points",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save optimal config to configs/optimal_throughput.json",
    )
    parser.add_argument(
        "--max-containers",
        type=int,
        default=32,
        help="Maximum containers to test in bisection (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Find a CSV to test with
    csv_path = args.csv
    if not csv_path:
        kaggle_dir = Path("data/kaggle")
        if kaggle_dir.exists():
            csvs = list(kaggle_dir.glob("*/data.csv"))
            if csvs:
                csv_path = str(csvs[0])
    
    if not csv_path:
        # Create a small test CSV
        csv_path = "data/fixtures/test_profiler.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_path).write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        print(f"Created test CSV: {csv_path}")
    
    print(f"\n📁 Using CSV: {csv_path}")
    
    # Run profiler
    try:
        results = asyncio.run(run_profiler(
            csv_path,
            quick=args.quick,
            max_containers_to_test=args.max_containers,
        ))
        
        print_summary(results)
        
        if args.save:
            save_config(results)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted")
        cleanup_csv_sandbox_containers()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        cleanup_csv_sandbox_containers()
        raise


if __name__ == "__main__":
    main()
