#!/usr/bin/env python3
"""
代码打包脚本 - 将多个目录的代码文件合并为单个文本文件
直接指定文件夹地址，无需命令行参数
"""

import os
import datetime
from pathlib import Path

def pack_code_directories():
    """直接打包指定目录的代码文件"""
    
    # ========== 在这里修改配置 ==========
    
    # 要打包的目录列表（直接在这里修改）
    directories_to_pack = [
        'csrc',
        'include'
        # 可以添加更多目录，例如：
        # 'src',
        # 'lib',
        # 'utils'
    ]
    
    # 输出文件名
    output_filename = "code_context.txt"
    
    # 要包含的文件扩展名
    target_extensions = {
        '.cc', '.cpp', '.c', '.h', '.hpp', '.cu', '.cuh',  # C/C++/CUDA
        '.py',  # Python
        '.java', '.js', '.ts', '.go', '.rs', '.php', '.rb',  # 其他语言
        '.md', '.txt'  # 文档
    }
    
    # 要忽略的目录和文件模式
    ignore_patterns = [
        '__pycache__', '*.pyc', 'node_modules', '.git', 
        '.svn', '.DS_Store', '*.so', '*.dll', '*.exe',
        '*.o', '*.a', '*.class', '*.jar', '*.war',
        '*.log', '*.tmp', '*.temp', 'build/', 'dist/',
        '*.egg-info', '.env', 'venv/', 'env/', '.venv'
    ]
    
    # ========== 配置结束 ==========
    
    print("开始代码打包...")
    print(f"打包目录: {directories_to_pack}")
    print(f"输出文件: {output_filename}")
    print(f"目标扩展名: {', '.join(sorted(target_extensions))}")
    
    # 检查目录是否存在
    valid_dirs = []
    for directory in directories_to_pack:
        if os.path.exists(directory):
            valid_dirs.append(Path(directory).resolve())
            print(f"✓ 找到目录: {directory}")
        else:
            print(f"✗ 目录不存在: {directory}")
    
    if not valid_dirs:
        print("错误：没有找到任何有效目录！")
        return False
    
    total_files = 0
    total_size = 0
    file_stats = {}
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            # 写入文件头信息
            outfile.write(f"{'#'*80}\n")
            outfile.write("# 代码上下文打包文件\n")
            outfile.write(f"# 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"# 打包目录: {[str(d) for d in valid_dirs]}\n")
            outfile.write(f"# 文件类型: {', '.join(sorted(target_extensions))}\n")
            outfile.write(f"{'#'*80}\n\n")
            
            for directory in valid_dirs:
                outfile.write(f"\n{'#'*60}\n")
                outfile.write(f"# 目录: {directory}\n")
                outfile.write(f"{'#'*60}\n\n")
                
                # 递归遍历目录中的所有文件
                for filepath in directory.rglob('*'):
                    if filepath.is_file():
                        # 检查是否应该忽略该文件
                        if should_ignore_file(filepath, ignore_patterns):
                            continue
                        
                        # 检查文件扩展名
                        if filepath.suffix.lower() in target_extensions:
                            process_single_file(filepath, directory, outfile)
                            total_files += 1
                            
                            # 获取文件大小用于统计
                            try:
                                file_size = filepath.stat().st_size
                                total_size += file_size
                                ext = filepath.suffix.lower()
                                file_stats[ext] = file_stats.get(ext, 0) + 1
                            except:
                                pass
            
            # 写入统计信息
            outfile.write(f"\n{'#'*80}\n")
            outfile.write("# 打包统计信息\n")
            outfile.write(f"# 总文件数: {total_files}\n")
            outfile.write(f"# 总大小: {total_size} 字节\n")
            outfile.write("# 文件类型分布:\n")
            for ext, count in sorted(file_stats.items()):
                outfile.write(f"#   {ext}: {count} 个文件\n")
            outfile.write(f"{'#'*80}\n")
        
        print(f"\n✅ 打包完成！")
        print(f"📊 统计信息:")
        print(f"   文件数量: {total_files}")
        print(f"   总大小: {total_size} 字节")
        print(f"   输出文件: {output_filename}")
        
        if file_stats:
            print(f"   文件类型分布:")
            for ext, count in sorted(file_stats.items()):
                print(f"     {ext}: {count} 个文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 打包过程中出现错误: {str(e)}")
        return False

def should_ignore_file(filepath, ignore_patterns):
    """检查文件是否应该被忽略"""
    import fnmatch
    
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(filepath.name, pattern):
            return True
        if pattern.endswith('/') and pattern[:-1] in str(filepath):
            return True
    return False

def process_single_file(filepath, base_directory, outfile):
    """处理单个文件：读取内容并写入输出文件"""
    try:
        # 获取相对路径
        relative_path = filepath.relative_to(base_directory)
        
        # 获取文件信息
        try:
            stat = filepath.stat()
            file_size = stat.st_size
            modified_time = datetime.datetime.fromtimestamp(stat.st_mtime)
        except:
            file_size = 0
            modified_time = None
        
        # 写入文件头
        outfile.write(f"\n{'='*80}\n")
        outfile.write(f"文件: {relative_path}\n")
        outfile.write(f"扩展名: {filepath.suffix}\n")
        outfile.write(f"大小: {file_size} 字节\n")
        if modified_time:
            outfile.write(f"修改时间: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"{'='*80}\n\n")
        
        # 读取并写入文件内容
        content = read_file_with_encoding(filepath)
        outfile.write(content)
        outfile.write('\n')  # 文件间空行
        
        print(f"✓ 已处理: {relative_path}")
        
    except Exception as e:
        print(f"⚠️  处理文件失败 {filepath}: {e}")
        outfile.write(f"<处理文件时出错: {str(e)}>\n")

def read_file_with_encoding(filepath):
    """尝试用不同编码读取文件"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'gbk', 'gb2312', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # 如果所有编码都失败，尝试二进制读取
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        return f"<二进制文件，大小: {len(content)} 字节>"
    except Exception as e:
        return f"<读取文件错误: {str(e)}>"

# 高级版本：支持更多定制选项
def advanced_pack_code(directories, output_file="code_context.txt", 
                      extensions=None, max_file_size=1024*1024):
    """高级打包函数，支持更多选项"""
    if extensions is None:
        extensions = {'.cc', '.cpp', '.c', '.h', '.hpp', '.cu', '.py'}
    
    print(f"高级打包模式启动...")
    print(f"目录: {directories}")
    print(f"输出: {output_file}")
    print(f"扩展名: {extensions}")
    print(f"最大文件大小: {max_file_size} 字节")
    
    file_count = 0
    skipped_files = []
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"代码打包文件 - 生成时间: {datetime.datetime.now()}\n\n")
        
        for directory in directories:
            if not os.path.exists(directory):
                print(f"跳过不存在的目录: {directory}")
                continue
                
            for root, dirs, files in os.walk(directory):
                # 过滤忽略的目录
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
                
                for file in files:
                    filepath = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in extensions:
                        try:
                            # 检查文件大小
                            file_size = os.path.getsize(filepath)
                            if file_size > max_file_size:
                                skipped_files.append(f"{filepath} (大小: {file_size} 字节)")
                                continue
                            
                            # 写入文件内容
                            relative_path = os.path.relpath(filepath, directory)
                            outfile.write(f"\n{'='*60}\n")
                            outfile.write(f"文件: {relative_path}\n")
                            outfile.write(f"{'='*60}\n\n")
                            
                            content = read_file_with_encoding(Path(filepath))
                            outfile.write(content)
                            outfile.write('\n\n')
                            
                            file_count += 1
                            print(f"已添加: {relative_path}")
                            
                        except Exception as e:
                            print(f"错误处理文件 {filepath}: {e}")
    
    print(f"\n打包完成！共处理 {file_count} 个文件")
    if skipped_files:
        print(f"跳过了 {len(skipped_files)} 个过大文件:")
        for f in skipped_files[:5]:  # 只显示前5个
            print(f"  {f}")
        if len(skipped_files) > 5:
            print(f"  ... 还有 {len(skipped_files)-5} 个文件被跳过")
    
    return file_count

if __name__ == "__main__":
    # 使用方法1：直接调用主函数（推荐）
    pack_code_directories()
    
    # 使用方法2：高级定制版本
    # custom_dirs = ['csrc', 'include', 'src']
    # custom_exts = {'.cc', '.h', '.hpp', '.cu', '.py'}
    # advanced_pack_code(custom_dirs, "custom_context.txt", custom_exts)
