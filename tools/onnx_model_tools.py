"""
ONNX Model Optimizer CLI Tool

This tool provides utilities for optimizing ONNX models and benchmarking performance.
It can be used to:
1. Optimize models for GPU acceleration
2. Benchmark CPU vs GPU performance
3. Analyze model structure and memory usage
4. Convert models to different precision (FP16, INT8)
"""

import argparse
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
import time
import numpy as np
import cv2
import onnxruntime as ort

# Add parent directory to sys.path
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_model_info(model_path):
    """Get basic information about an ONNX model"""
    try:
        import onnx
        model = onnx.load(model_path)
        
        # Get basic model info
        info = {
            "model_path": model_path,
            "ir_version": str(model.ir_version),
            "producer": model.producer_name,
            "model_version": str(model.model_version),
            "domain": model.domain,
            "description": model.doc_string,
            "size_mb": os.path.getsize(model_path) / (1024 * 1024),
            "graph_name": model.graph.name,
        }
        
        # Get input/output info
        inputs = []
        for i in model.graph.input:
            shape = []
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param:
                    shape.append(d.dim_param)
                else:
                    shape.append(d.dim_value)
            inputs.append({"name": i.name, "shape": shape})
        
        outputs = []
        for o in model.graph.output:
            shape = []
            for d in o.type.tensor_type.shape.dim:
                if d.dim_param:
                    shape.append(d.dim_param)
                else:
                    shape.append(d.dim_value)
            outputs.append({"name": o.name, "shape": shape})
        
        # Count operators
        ops = {}
        for node in model.graph.node:
            op_type = node.op_type
            if op_type in ops:
                ops[op_type] += 1
            else:
                ops[op_type] = 1
        
        info["inputs"] = inputs
        info["outputs"] = outputs
        info["operators"] = ops
        info["node_count"] = len(model.graph.node)
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e)}

def benchmark_model(model_path, iterations=10, providers=None):
    """Benchmark model performance with random input data"""
    try:
        # Set default providers if none specified
        if providers is None:
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        active_providers = session.get_providers()
        
        # Get input details
        inputs = session.get_inputs()
        input_feed = {}
        
        for input_meta in inputs:
            # Create random input with proper shape
            shape = input_meta.shape
            
            # Handle dynamic dimensions
            concrete_shape = []
            for dim in shape:
                if isinstance(dim, int):
                    concrete_shape.append(dim)
                else:
                    # Replace dynamic dimension with reasonable size
                    concrete_shape.append(1)  # Use batch size of 1
            
            # Create random tensor with appropriate type
            if input_meta.type == 'tensor(float)':
                tensor = np.random.rand(*concrete_shape).astype(np.float32)
            elif input_meta.type == 'tensor(int64)':
                tensor = np.random.randint(0, 10, size=concrete_shape).astype(np.int64)
            else:
                # Default to float32
                tensor = np.random.rand(*concrete_shape).astype(np.float32)
            
            input_feed[input_meta.name] = tensor
        
        # Warmup
        _ = session.run(None, input_feed)
        
        # Run benchmark
        latencies = []
        for i in range(iterations):
            start_time = time.time()
            _ = session.run(None, input_feed)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        return {
            "model_path": model_path,
            "active_providers": active_providers,
            "iterations": iterations,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latencies_ms": latencies
        }
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return {"error": str(e)}

def optimize_model(model_path, output_path=None, fp16=False, int8=False):
    """Optimize an ONNX model for inference performance"""
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Load the model
        model = onnx.load(model_path)
        
        # Generate output path if not specified
        if output_path is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            name, ext = os.path.splitext(model_name)
            output_path = os.path.join(model_dir, f"{name}_optimized{ext}")
        
        # Apply graph optimizations
        model_type = "bert"  # Generic default
        opt_model = optimizer.optimize_model(
            model_path,
            model_type=model_type,
            num_heads=12,
            hidden_size=768,
            optimization_options=None
        )
        
        # Save optimized model
        opt_model.save_model_to_file(output_path)
        
        # Apply quantization if requested
        if fp16:
            from onnxmltools.utils.float16_converter import convert_float_to_float16
            
            # Load the optimized model
            optimized_model = onnx.load(output_path)
            
            # Convert to FP16
            fp16_model = convert_float_to_float16(optimized_model)
            
            # Generate FP16 output path
            fp16_path = output_path.replace(".onnx", "_fp16.onnx")
            onnx.save(fp16_model, fp16_path)
            
            output_path = fp16_path
        
        if int8:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Generate INT8 output path
            int8_path = output_path.replace(".onnx", "_int8.onnx")
            
            # Quantize the model
            quantize_dynamic(
                model_input=output_path,
                model_output=int8_path,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QInt8
            )
            
            output_path = int8_path
        
        return {
            "original_model": model_path,
            "optimized_model": output_path,
            "fp16_applied": fp16,
            "int8_applied": int8
        }
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return {"error": str(e)}

def compare_models(model_paths, test_image=None, iterations=10):
    """Compare performance of multiple models"""
    results = {}
    
    for model_path in model_paths:
        # Get model info
        model_info = get_model_info(model_path)
        
        # Benchmark with CPU
        cpu_benchmark = benchmark_model(
            model_path, 
            iterations=iterations,
            providers=['CPUExecutionProvider']
        )
        
        # Benchmark with GPU if available
        gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
        gpu_benchmark = None
        
        if gpu_available:
            gpu_benchmark = benchmark_model(
                model_path, 
                iterations=iterations,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        
        # Calculate speedup if GPU benchmark available
        speedup = None
        if gpu_benchmark and 'error' not in gpu_benchmark and 'error' not in cpu_benchmark:
            speedup = cpu_benchmark['avg_latency_ms'] / gpu_benchmark['avg_latency_ms']
        
        # Store results
        model_name = os.path.basename(model_path)
        results[model_name] = {
            "model_info": model_info,
            "cpu_benchmark": cpu_benchmark,
            "gpu_benchmark": gpu_benchmark,
            "speedup": speedup
        }
    
    return results

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="ONNX Model Optimizer and Benchmark Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about an ONNX model")
    info_parser.add_argument("model_path", type=str, help="Path to the ONNX model")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("model_path", type=str, help="Path to the ONNX model")
    benchmark_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    benchmark_parser.add_argument("--cpu-only", action="store_true", help="Use CPU provider only")
    benchmark_parser.add_argument("--gpu-only", action="store_true", help="Use GPU provider only")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize model for inference")
    optimize_parser.add_argument("model_path", type=str, help="Path to the ONNX model")
    optimize_parser.add_argument("--output", type=str, help="Output path for optimized model")
    optimize_parser.add_argument("--fp16", action="store_true", help="Convert to FP16 precision")
    optimize_parser.add_argument("--int8", action="store_true", help="Quantize to INT8 precision")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("model_paths", type=str, nargs="+", help="Paths to ONNX models to compare")
    compare_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    
    # System info command
    sysinfo_parser = subparsers.add_parser("sysinfo", help="Get system and ONNX Runtime information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "info":
        result = get_model_info(args.model_path)
        print(json.dumps(result, indent=2))
    
    elif args.command == "benchmark":
        providers = None
        if args.cpu_only:
            providers = ['CPUExecutionProvider']
        elif args.gpu_only:
            providers = ['CUDAExecutionProvider']
        
        result = benchmark_model(args.model_path, args.iterations, providers)
        print(json.dumps(result, indent=2))
    
    elif args.command == "optimize":
        result = optimize_model(args.model_path, args.output, args.fp16, args.int8)
        print(json.dumps(result, indent=2))
    
    elif args.command == "compare":
        result = compare_models(args.model_paths, iterations=args.iterations)
        print(json.dumps(result, indent=2))
    
    elif args.command == "sysinfo":
        # Get system info
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "onnxruntime_version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "cuda_available": 'CUDAExecutionProvider' in ort.get_available_providers(),
        }
        
        # Get GPU info if available
        if system_info["cuda_available"]:
            try:
                import torch
                if torch.cuda.is_available():
                    system_info["cuda_version"] = torch.version.cuda
                    system_info["gpu_count"] = torch.cuda.device_count()
                    system_info["gpu_name"] = torch.cuda.get_device_name(0)
                    system_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                system_info["gpu_info"] = "PyTorch not available for detailed GPU info"
        
        print(json.dumps(system_info, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
