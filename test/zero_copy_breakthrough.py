#!/usr/bin/env python3
"""
零拷貝突破實現 - 基於您的測試環境
解決OpenCL數據傳輸瓶頸的實際方案
"""

import time
import numpy as np
import pyopencl as cl
import ctypes
from ctypes import c_void_p, c_size_t, c_uint, c_ulong
import logging
from typing import Dict, List, Tuple, Any
import threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroCopyBreathrough:
    """零拷貝突破實現"""
    
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.memory_pool = {}
        self.pool_size = 64 * 1024 * 1024  # 64MB pool
        
    def initialize_opencl(self):
        """初始化OpenCL環境"""
        logger.info("🔧 初始化零拷貝突破環境...")
        
        platforms = cl.get_platforms()
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    self.device = devices[0]
                    self.context = cl.Context([self.device])
                    self.queue = cl.CommandQueue(self.context)
                    break
            except:
                continue
        
        if not self.device:
            raise RuntimeError("沒有找到可用的GPU設備")
        
        logger.info(f"✅ 環境初始化完成")
        logger.info(f"   設備: {self.device.name}")
        logger.info(f"   準備突破數據傳輸瓶頸...")
        
        # 初始化記憶體池
        self._initialize_memory_pool()
    
    def _initialize_memory_pool(self):
        """初始化記憶體池 - 預分配大塊記憶體"""
        logger.info("🏊‍♂️ 初始化記憶體池...")
        
        # 分配不同大小的記憶體池
        pool_sizes = [
            (1024, 100),      # 1K * 100
            (10240, 50),      # 10K * 50  
            (102400, 20),     # 100K * 20
            (1024000, 10)     # 1M * 10
        ]
        
        for size, count in pool_sizes:
            self.memory_pool[size] = []
            for i in range(count):
                # 分配對齊的系統記憶體
                host_mem = np.zeros(size, dtype=np.float32)
                
                # 使用USE_HOST_PTR創建OpenCL buffer，實現零拷貝
                cl_buffer = cl.Buffer(
                    self.context, 
                    cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                    hostbuf=host_mem
                )
                
                self.memory_pool[size].append({
                    'host_ptr': host_mem,
                    'cl_buffer': cl_buffer,
                    'in_use': False
                })
        
        logger.info(f"✅ 記憶體池初始化完成，預分配 {sum(count for _, count in pool_sizes)} 個buffer")
    
    def get_pool_buffer(self, size: int):
        """從記憶體池獲取buffer"""
        if size not in self.memory_pool:
            # 找最接近的大小
            available_sizes = [s for s in self.memory_pool.keys() if s >= size]
            if not available_sizes:
                size = max(self.memory_pool.keys())
            else:
                size = min(available_sizes)
        
        # 找空閒的buffer
        for buffer in self.memory_pool[size]:
            if not buffer['in_use']:
                buffer['in_use'] = True
                return buffer
        
        # 如果沒有空閒的，創建新的
        host_mem = np.zeros(size, dtype=np.float32)
        cl_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
            hostbuf=host_mem
        )
        buffer = {
            'host_ptr': host_mem,
            'cl_buffer': cl_buffer,
            'in_use': True
        }
        self.memory_pool[size].append(buffer)
        return buffer
    
    def return_pool_buffer(self, buffer):
        """歸還buffer到記憶體池"""
        buffer['in_use'] = False
    
    def create_optimized_kernel(self) -> cl.Program:
        """創建優化的kernel"""
        kernel_source = """
        // 針對APU優化的kernel
        __kernel void zero_copy_vector_add(
            __global float* a, 
            __global float* b, 
            __global float* result, 
            int n
        ) {
            int idx = get_global_id(0);
            int stride = get_global_size(0);
            
            // 展開循環，減少分支
            for (int i = idx; i < n; i += stride) {
                result[i] = a[i] + b[i];
            }
        }
        
        __kernel void zero_copy_complex_compute(
            __global float* input,
            __global float* output,
            int n
        ) {
            int idx = get_global_id(0);
            int stride = get_global_size(0);
            
            for (int i = idx; i < n; i += stride) {
                float x = input[i];
                // 複雜計算來測試真實場景
                output[i] = sin(x) * cos(x) + sqrt(abs(x)) * 0.5f;
            }
        }
        """
        
        return cl.Program(self.context, kernel_source).build()
    
    def test_zero_copy_performance(self, data_size: int, iterations: int = 5) -> Dict[str, float]:
        """測試零拷貝性能"""
        logger.info(f"🚀 測試零拷貝性能 (大小: {data_size})")
        
        program = self.create_optimized_kernel()
        kernel = program.zero_copy_vector_add
        
        times = {
            'buffer_acquisition': [],
            'data_preparation': [],
            'kernel_execution': [],
            'result_access': [],
            'buffer_cleanup': [],
            'total': []
        }
        
        for i in range(iterations):
            start_total = time.perf_counter()
            
            # 1. 從記憶體池獲取buffer
            start = time.perf_counter()
            buf_a = self.get_pool_buffer(data_size)
            buf_b = self.get_pool_buffer(data_size)
            buf_result = self.get_pool_buffer(data_size)
            times['buffer_acquisition'].append(time.perf_counter() - start)
            
            # 2. 直接在host memory準備數據 (零拷貝！)
            start = time.perf_counter()
            buf_a['host_ptr'][:data_size] = np.random.rand(data_size).astype(np.float32)
            buf_b['host_ptr'][:data_size] = np.random.rand(data_size).astype(np.float32)
            times['data_preparation'].append(time.perf_counter() - start)
            
            # 3. 執行kernel (GPU直接訪問host memory)
            start = time.perf_counter()
            kernel.set_arg(0, buf_a['cl_buffer'])
            kernel.set_arg(1, buf_b['cl_buffer'])
            kernel.set_arg(2, buf_result['cl_buffer'])
            kernel.set_arg(3, np.int32(data_size))
            
            cl.enqueue_nd_range_kernel(self.queue, kernel, (min(data_size, 1024),), None)
            self.queue.finish()
            times['kernel_execution'].append(time.perf_counter() - start)
            
            # 4. 直接訪問結果 (零拷貝！)
            start = time.perf_counter()
            result = buf_result['host_ptr'][:data_size].copy()  # 只是為了模擬使用
            times['result_access'].append(time.perf_counter() - start)
            
            # 5. 歸還buffer
            start = time.perf_counter()
            self.return_pool_buffer(buf_a)
            self.return_pool_buffer(buf_b)
            self.return_pool_buffer(buf_result)
            times['buffer_cleanup'].append(time.perf_counter() - start)
            
            times['total'].append(time.perf_counter() - start_total)
        
        # 計算平均值
        avg_times = {key: np.mean(time_list) * 1000 for key, time_list in times.items()}
        
        logger.info(f"   Buffer獲取: {avg_times['buffer_acquisition']:.3f} ms")
        logger.info(f"   數據準備: {avg_times['data_preparation']:.3f} ms")
        logger.info(f"   Kernel執行: {avg_times['kernel_execution']:.3f} ms")
        logger.info(f"   結果訪問: {avg_times['result_access']:.3f} ms")
        logger.info(f"   Buffer歸還: {avg_times['buffer_cleanup']:.3f} ms")
        logger.info(f"   總時間: {avg_times['total']:.3f} ms")
        
        return avg_times
    
    def test_async_pipeline(self, data_size: int, chunks: int = 4) -> Dict[str, float]:
        """測試異步流水線處理"""
        logger.info(f"🔄 測試異步流水線 (大小: {data_size}, 分塊: {chunks})")
        
        program = self.create_optimized_kernel()
        kernel = program.zero_copy_complex_compute
        
        chunk_size = data_size // chunks
        
        # 創建多個CommandQueue實現並行
        queues = [cl.CommandQueue(self.context) for _ in range(min(chunks, 3))]
        
        start_total = time.perf_counter()
        
        # 準備所有數據塊
        input_data = np.random.rand(data_size).astype(np.float32)
        results = []
        
        def process_chunk(chunk_id, start_idx, end_idx, queue_idx):
            """處理單個數據塊"""
            queue = queues[queue_idx % len(queues)]
            
            # 獲取buffer
            input_buf = self.get_pool_buffer(chunk_size)
            output_buf = self.get_pool_buffer(chunk_size)
            
            # 準備數據
            chunk_data = input_data[start_idx:end_idx]
            input_buf['host_ptr'][:len(chunk_data)] = chunk_data
            
            # 執行kernel
            kernel.set_arg(0, input_buf['cl_buffer'])
            kernel.set_arg(1, output_buf['cl_buffer'])
            kernel.set_arg(2, np.int32(len(chunk_data)))
            
            cl.enqueue_nd_range_kernel(queue, kernel, (min(len(chunk_data), 256),), None)
            queue.finish()
            
            # 獲取結果
            result = output_buf['host_ptr'][:len(chunk_data)].copy()
            
            # 歸還buffer
            self.return_pool_buffer(input_buf)
            self.return_pool_buffer(output_buf)
            
            return chunk_id, result
        
        # 並行處理所有塊
        with ThreadPoolExecutor(max_workers=chunks) as executor:
            futures = []
            for i in range(chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, data_size)
                future = executor.submit(process_chunk, i, start_idx, end_idx, i)
                futures.append(future)
            
            # 收集結果
            for future in futures:
                chunk_id, result = future.result()
                results.append((chunk_id, result))
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        logger.info(f"   流水線總時間: {total_time:.3f} ms")
        logger.info(f"   處理 {chunks} 個塊並行完成")
        
        return {'pipeline_total_ms': total_time, 'chunks_processed': chunks}
    
    def run_breakthrough_comparison(self):
        """運行突破性對比測試"""
        logger.info("🔥 開始零拷貝突破對比測試")
        
        test_sizes = [1024, 10240, 102400, 1024000]
        
        results = {
            'device_info': {
                'name': self.device.name,
                'vendor': self.device.vendor,
                'version': self.device.version
            },
            'zero_copy_tests': {},
            'pipeline_tests': {}
        }
        
        # 零拷貝測試
        logger.info("\n🚀 零拷貝性能測試:")
        for size in test_sizes:
            logger.info(f"\n--- 測試大小: {size} 元素 ({size*4/1024:.1f} KB) ---")
            results['zero_copy_tests'][size] = self.test_zero_copy_performance(size)
        
        # 流水線測試
        logger.info("\n🔄 異步流水線測試:")
        for size in [102400, 1024000]:  # 較大數據測試流水線
            logger.info(f"\n--- 流水線大小: {size} 元素 ---")
            results['pipeline_tests'][size] = self.test_async_pipeline(size)
        
        return results
    
    def analyze_breakthrough(self, results: Dict[str, Any], baseline_results: Dict[str, Any] = None):
        """分析突破效果"""
        logger.info("\n" + "="*80)
        logger.info("🎯 零拷貝突破效果分析")
        logger.info("="*80)
        
        device_info = results['device_info']
        logger.info(f"🖥️ 測試設備: {device_info['name']} ({device_info['vendor']})")
        
        # 分析零拷貝性能
        logger.info(f"\n🚀 零拷貝性能分析:")
        zero_copy_tests = results['zero_copy_tests']
        
        for size, times in zero_copy_tests.items():
            compute_time = times['kernel_execution']
            data_prep_time = times['data_preparation'] + times['result_access']
            management_time = times['buffer_acquisition'] + times['buffer_cleanup']
            
            logger.info(f"\n   數據大小 {size} ({size*4/1024:.1f} KB):")
            logger.info(f"     計算時間: {compute_time:.3f} ms ({compute_time/times['total']*100:.1f}%)")
            logger.info(f"     數據處理: {data_prep_time:.3f} ms ({data_prep_time/times['total']*100:.1f}%)")
            logger.info(f"     管理開銷: {management_time:.3f} ms ({management_time/times['total']*100:.1f}%)")
            logger.info(f"     總時間: {times['total']:.3f} ms")
            
            # 如果有基準測試結果，進行對比
            if baseline_results and size in baseline_results.get('opencl_buffer_tests', {}):
                baseline = baseline_results['opencl_buffer_tests'][size]
                speedup = baseline['total'] / times['total']
                transfer_eliminated = baseline['data_upload'] + baseline['data_download']
                
                logger.info(f"     🔥 性能提升: {speedup:.2f}倍")
                logger.info(f"     📈 節省傳輸時間: {transfer_eliminated:.3f} ms")
        
        # 分析流水線效果
        if results['pipeline_tests']:
            logger.info(f"\n🔄 異步流水線分析:")
            for size, pipeline_result in results['pipeline_tests'].items():
                chunks = pipeline_result['chunks_processed']
                total_time = pipeline_result['pipeline_total_ms']
                
                # 估算串行處理時間
                if size in zero_copy_tests:
                    estimated_serial = zero_copy_tests[size]['total'] * chunks
                    parallel_efficiency = estimated_serial / total_time
                    
                    logger.info(f"\n   數據大小 {size}，分 {chunks} 塊:")
                    logger.info(f"     並行總時間: {total_time:.3f} ms")
                    logger.info(f"     估算串行時間: {estimated_serial:.3f} ms")
                    logger.info(f"     並行效率: {parallel_efficiency:.2f}倍")
        
        # 突破效果總結
        logger.info(f"\n🎯 突破效果總結:")
        
        # 計算平均數據處理占比
        avg_data_ratio = np.mean([
            (times['data_preparation'] + times['result_access']) / times['total']
            for times in zero_copy_tests.values()
        ])
        
        avg_compute_ratio = np.mean([
            times['kernel_execution'] / times['total']
            for times in zero_copy_tests.values()
        ])
        
        logger.info(f"📊 平均計算時間占比: {avg_compute_ratio*100:.1f}%")
        logger.info(f"📊 平均數據處理占比: {avg_data_ratio*100:.1f}%")
        
        if avg_compute_ratio > 0.6:
            logger.info("✅ 成功突破！計算成為主要部分，數據傳輸瓶頸已解決")
        elif avg_compute_ratio > 0.4:
            logger.info("🔥 顯著改善！計算占比大幅提升")
        else:
            logger.info("⚠️ 仍有優化空間，繼續調整策略")
        
        logger.info("💡 零拷貝 + 記憶體池 + 異步流水線 = 突破數據傳輸瓶頸")
        
        return results

def main():
    """主測試函數"""
    breakthrough = ZeroCopyBreathrough()
    
    try:
        # 初始化
        breakthrough.initialize_opencl()
        
        # 運行突破測試
        results = breakthrough.run_breakthrough_comparison()
        
        # 分析結果
        breakthrough.analyze_breakthrough(results)
        
        logger.info("\n🎉 零拷貝突破測試完成！")
        
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()