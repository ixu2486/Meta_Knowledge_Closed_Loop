(env311) PS F:\test> & F:/test/env311/Scripts/python.exe
 f:/test/net/zero_copy_breakthrough.py
INFO:__main__:🔧 初始化高級零拷貝環境...
INFO:__main__:✅ RetryIX SVM Core 已載入: OpenCL.dll
INFO:__main__:   SVM粗粒度: ✅
INFO:__main__:   SVM細粒度: ✅
INFO:__main__:   SVM原子操作: ❌
INFO:__main__:   統一內存: ❌
INFO:__main__:   內存映射: ✅
INFO:__main__:✅ 環境初始化完成
INFO:__main__:   設備: gfx1010:xnack-
INFO:__main__:   支持特性: ['svm_coarse', 'svm_fine', 'svm_atomics', 'unified_memory', 'map_buffer']
INFO:__main__:🚀 初始化高級內存管理...
INFO:__main__:   初始化HOST_PTR池...
INFO:__main__:   初始化SVM池...
INFO:__main__:   RetryIX SVM池初始化成功！分配 3 個SVM buffer
INFO:__main__:   初始化映射內存池...
INFO:__main__:🎯 開始全面零拷貝技術對比測試
INFO:__main__:
📊 測試策略: ['traditional', 'use_host_ptr', 'ultra_fast_host_ptr', 'map_buffer', 'svm_coarse']
INFO:__main__:📊 測試大小: [1024, 10240, 102400, 1024000]
INFO:__main__:🚀 RetryIX SVM 測試已啟用
INFO:__main__:
🔬 測試策略: traditional
INFO:__main__:   測試大小: 1024 元素 (4.0 KB)
F:\test\env311\Lib\site-packages\pyopencl\cache.py:496: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
  _create_built_program_from_source_cached(
INFO:__main__:     總時間: 1014.4 μs (1.01 ms)
INFO:__main__:     其中內核: 237.7 μs (23.4%)
INFO:__main__:   測試大小: 10240 元素 (40.0 KB)
INFO:__main__:     總時間: 432.8 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 164.5 μs (38.0%)
INFO:__main__:     數據準備: 86.4 μs ⚡ 超高效!
INFO:__main__:   測試大小: 102400 元素 (400.0 KB)       
INFO:__main__:     總時間: 952.1 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 247.6 μs (26.0%)
INFO:__main__:   測試大小: 1024000 元素 (4000.0 KB)     
INFO:__main__:     總時間: 10470.1 μs (10.47 ms)
INFO:__main__:     其中內核: 969.9 μs (9.3%)
INFO:__main__:
🔬 測試策略: use_host_ptr
INFO:__main__:   測試大小: 1024 元素 (4.0 KB)
INFO:__main__:     總時間: 93.8 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 83.9 μs (89.4%)
INFO:__main__:     數據準備: 2.5 μs ⚡ 超高效!
INFO:__main__:   測試大小: 10240 元素 (40.0 KB)
INFO:__main__:     總時間: 107.5 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 94.7 μs (88.1%)
INFO:__main__:     數據準備: 4.6 μs ⚡ 超高效!
INFO:__main__:   測試大小: 102400 元素 (400.0 KB)       
INFO:__main__:     總時間: 254.4 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 224.0 μs (88.1%)
INFO:__main__:     數據準備: 16.7 μs ⚡ 超高效!
INFO:__main__:   測試大小: 1024000 元素 (4000.0 KB)     
INFO:__main__:     總時間: 1652.2 μs (1.65 ms)
INFO:__main__:     其中內核: 1138.6 μs (68.9%)
INFO:__main__:
🔬 測試策略: ultra_fast_host_ptr
INFO:__main__:   測試大小: 1024 元素 (4.0 KB)
INFO:__main__:     總時間: 119.8 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 110.9 μs (92.6%)
INFO:__main__:     數據準備: 2.3 μs ⚡ 超高效!
INFO:__main__:   測試大小: 10240 元素 (40.0 KB)
INFO:__main__:     總時間: 149.6 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 134.6 μs (90.0%)
INFO:__main__:     數據準備: 8.0 μs ⚡ 超高效!
INFO:__main__:   測試大小: 102400 元素 (400.0 KB)       
INFO:__main__:     總時間: 222.2 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 203.7 μs (91.7%)
INFO:__main__:     數據準備: 15.5 μs ⚡ 超高效!
INFO:__main__:   測試大小: 1024000 元素 (4000.0 KB)     
INFO:__main__:     總時間: 1476.1 μs (1.48 ms)
INFO:__main__:     其中內核: 1103.0 μs (74.7%)
INFO:__main__:
🔬 測試策略: map_buffer
INFO:__main__:   測試大小: 1024 元素 (4.0 KB)
INFO:__main__:     總時間: 268.1 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 142.8 μs (53.3%)
INFO:__main__:     數據準備: 44.9 μs ⚡ 超高效!
INFO:__main__:   測試大小: 10240 元素 (40.0 KB)
INFO:__main__:     總時間: 253.5 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 139.0 μs (54.8%)
INFO:__main__:     數據準備: 39.0 μs ⚡ 超高效!
INFO:__main__:   測試大小: 102400 元素 (400.0 KB)       
INFO:__main__:     總時間: 544.2 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 292.0 μs (53.6%)
INFO:__main__:     數據準備: 77.0 μs ⚡ 超高效!
INFO:__main__:   測試大小: 1024000 元素 (4000.0 KB)     
INFO:__main__:     總時間: 3668.5 μs (3.67 ms)
INFO:__main__:     其中內核: 1873.8 μs (51.1%)
INFO:__main__:
🔬 測試策略: svm_coarse
INFO:__main__:   測試大小: 1024 元素 (4.0 KB)
INFO:__main__:     總時間: 140.0 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 93.3 μs (66.6%)
INFO:__main__:     數據準備: 21.0 μs ⚡ 超高效!
INFO:__main__:   測試大小: 10240 元素 (40.0 KB)
INFO:__main__:     總時間: 170.6 μs ⚡ 亞毫秒級!        
INFO:__main__:     其中內核: 117.1 μs (68.7%)
INFO:__main__:     數據準備: 25.5 μs ⚡ 超高效!
INFO:__main__:   測試大小: 102400 元素 (400.0 KB)       
INFO:__main__:     總時間: 389.7 μs ⚡ 亞毫秒級!
INFO:__main__:     其中內核: 271.0 μs (69.6%)
INFO:__main__:     數據準備: 68.3 μs ⚡ 超高效!
INFO:__main__:   測試大小: 1024000 元素 (4000.0 KB)     
INFO:__main__:     總時間: 1596.1 μs (1.60 ms)
INFO:__main__:     其中內核: 1207.5 μs (75.7%)
INFO:__main__:
================================================================================
INFO:__main__:🎯 高級零拷貝技術效果分析
INFO:__main__:================================================================================
INFO:__main__:
📊 性能對比表 (時間單位: 微秒)
INFO:__main__:策略\大小                     1024(4KB)    10240(40KB)  102400(400KB)1024000(4000KB)
INFO:__main__:--------------------------------------------------------------------------------
INFO:__main__:traditional             1014(1.00x)     433(1.00x)     952(1.00x)   10470(1.00x)
INFO:__main__:use_host_ptr             94(10.82x)     107(4.03x)     254(3.74x)    1652(6.34x)
INFO:__main__:ultra_fast_host_ptr      120(8.47x)     150(2.89x)     222(4.28x)    1476(7.09x)
INFO:__main__:map_buffer               268(3.78x)     253(1.71x)     544(1.75x)    3668(2.85x)
INFO:__main__:svm_coarse               140(7.24x)     171(2.54x)     390(2.44x)    1596(6.56x)
INFO:__main__:
🏆 最佳策略分析:
INFO:__main__:
   數據大小 1024 (4.0 KB):
INFO:__main__:     最佳策略: use_host_ptr
INFO:__main__:     總時間: 93.8 μs
INFO:__main__:     計算占比: 89.4%
INFO:__main__:     性能提升: 10.82倍
INFO:__main__:
   數據大小 10240 (40.0 KB):
INFO:__main__:     最佳策略: use_host_ptr
INFO:__main__:     總時間: 107.5 μs
INFO:__main__:     計算占比: 88.1%
INFO:__main__:     性能提升: 4.03倍
INFO:__main__:
   數據大小 102400 (400.0 KB):
INFO:__main__:     最佳策略: ultra_fast_host_ptr        
INFO:__main__:     總時間: 222.2 μs
INFO:__main__:     計算占比: 91.7%
INFO:__main__:     性能提升: 4.28倍
INFO:__main__:
   數據大小 1024000 (4000.0 KB):
INFO:__main__:     最佳策略: ultra_fast_host_ptr        
INFO:__main__:     總時間: 1476.1 μs
INFO:__main__:     計算占比: 74.7%
INFO:__main__:     性能提升: 7.09倍
INFO:__main__:
🚀 亞毫秒級技術突破總結:
INFO:__main__:📈 亞毫秒級測試占比: 14/20 (70.0%)        
INFO:__main__:⚡ 最快記錄: use_host_ptr @ 1024元素 = 93.8 μs
INFO:__main__:✅ 最高計算占比策略: ultra_fast_host_ptr (87.2%)
INFO:__main__:🎉 亞毫秒級突破成功！數據傳輸延遲基本消除 
INFO:__main__:
💡 性能推薦方案:
INFO:__main__:   小數據(< 10KB): use_host_ptr (94μs ⚡, 計算占比89.4%)
INFO:__main__:   中數據(10KB-100KB): use_host_ptr (107μs ⚡, 計算占比88.1%)
INFO:__main__:   大數據(100KB-1MB): ultra_fast_host_ptr (222μs ⚡, 計算占比91.7%)
INFO:__main__:   超大數據(> 1MB): ultra_fast_host_ptr (1.48ms, 計算占比74.7%)
INFO:__main__:
🎉 高級零拷貝技術測試完成！
