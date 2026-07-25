import os

def get_folder_size(path):
    """计算文件夹总大小（字节）"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                pass
    return total

def format_size(size_bytes):
    """格式化字节为可读大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def find_large_folders(root_path, min_size_gb=10):
    """找出大于指定GB的所有文件夹"""
    min_bytes = min_size_gb * 1024 * 1024 * 1024
    results = []
    
    def scan(path, depth=0):
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    size = get_folder_size(item_path)
                    if size >= min_bytes:
                        results.append((item_path, size, depth))
                        # 继续深入扫描
                        scan(item_path, depth + 1)
        except PermissionError:
            pass
    
    print(f"正在扫描: {root_path}")
    print(f"筛选条件: 大于 {min_size_gb} GB 的文件夹")
    print("=" * 70)
    
    scan(root_path)
    
    # 按大小降序
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'大小':<15} {'路径'}")
    print("-" * 70)
    
    for path, size, depth in results:
        indent = "  " * depth
        print(f"{format_size(size):<15} {indent}{path}")
    
    print("-" * 70)
    print(f"共找到 {len(results)} 个大于 {min_size_gb}GB 的文件夹")

# 扫描 G:\game，只显示大于10GB的
find_large_folders("G:\\game", min_size_gb=10)