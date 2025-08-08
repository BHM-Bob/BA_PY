// By DeepSeek R1, prompt by BHM-Bob_G

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <locale.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <iconv.h>
#endif

// 字符串列表结构
typedef struct {
    wchar_t** items;
    int count;
} WStringList;

// 匹配配置结构
typedef struct {
    WStringList extensions;
    const wchar_t* name_substr;
    const wchar_t* path_substr;
    int case_sensitive;
    int include_dirs;
    int path_substr_len;
    int name_substr_len;
} MatchConfig;

// 释放宽字符串列表
void free_wstring_list(WStringList* list) {
    if (list->items) {
        for (int i = 0; i < list->count; i++) {
            free(list->items[i]);
        }
        free(list->items);
    }
    list->count = 0;
    list->items = NULL;
}

// 添加宽字符串到列表
void add_to_wstring_list(WStringList* list, const wchar_t* str) {
    if (!str) return;
    
    wchar_t** new_items = (wchar_t**)realloc(list->items, sizeof(wchar_t*) * (list->count + 1));
    if (!new_items) return;
    
    list->items = new_items;
    list->items[list->count] = _wcsdup(str);
    list->count++;
}

// 预分配宽字符串列表
WStringList prealloc_wstring_list(int capacity) {
    WStringList list;
    list.items = (wchar_t**)malloc(sizeof(wchar_t*) * capacity);
    list.count = 0;
    return list;
}

// UTF-8 到宽字符转换
wchar_t* utf8_to_wchar(const char* utf8_str) {
    if (!utf8_str) return NULL;
    
    #ifdef _WIN32
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, NULL, 0);
    if (len == 0) return NULL;
    
    wchar_t* wstr = (wchar_t*)malloc(len * sizeof(wchar_t));
    if (!wstr) return NULL;
    
    MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, wstr, len);
    return wstr;
    #else
    size_t in_len = strlen(utf8_str);
    size_t out_len = mbstowcs(NULL, utf8_str, 0) + 1;
    if (out_len == 0) return NULL;
    
    wchar_t* wstr = (wchar_t*)malloc(out_len * sizeof(wchar_t));
    if (!wstr) return NULL;
    
    setlocale(LC_ALL, "en_US.UTF-8");
    size_t converted = mbstowcs(wstr, utf8_str, out_len);
    if (converted == (size_t)-1) {
        free(wstr);
        return NULL;
    }
    
    return wstr;
    #endif
}

// 宽字符到 UTF-8 转换
char* wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return NULL;
    
    #ifdef _WIN32
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (len == 0) return NULL;
    
    char* str = (char*)malloc(len);
    if (!str) return NULL;
    
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
    return str;
    #else
    size_t in_len = wcslen(wstr);
    size_t out_len = wcstombs(NULL, wstr, 0) + 1;
    if (out_len == 0) return NULL;
    
    char* str = (char*)malloc(out_len);
    if (!str) return NULL;
    
    setlocale(LC_ALL, "en_US.UTF-8");
    size_t converted = wcstombs(str, wstr, out_len);
    if (converted == (size_t)-1) {
        free(str);
        return NULL;
    }
    
    return str;
    #endif
}

// 预编译匹配配置
MatchConfig prepare_match_config(const char* extensions, 
                                 const char* name_substr, 
                                 const char* path_substr, 
                                 int case_sensitive, 
                                 int include_dirs) {
    MatchConfig config;
    memset(&config, 0, sizeof(MatchConfig));
    
    // 处理扩展名
    if (extensions && *extensions) {
        // 转换为宽字符
        wchar_t* wext = utf8_to_wchar(extensions);
        if (!wext) return config;
        
        // 统计扩展名数量
        int count = 1;
        wchar_t* p = wext;
        while (*p) {
            if (*p == L';') count++;
            p++;
        }
        
        config.extensions = prealloc_wstring_list(count);
        
        // 分割扩展名
        wchar_t* token = wcstok(wext, L";");
        while (token) {
            // 处理大小写
            wchar_t* ext = _wcsdup(token);
            if (!case_sensitive) {
                for (wchar_t* c = ext; *c; c++) *c = (wchar_t)towlower(*c);
            }
            
            add_to_wstring_list(&config.extensions, ext);
            free(ext);
            token = wcstok(NULL, L";");
        }
        free(wext);
    }
    
    // 处理子串匹配
    if (name_substr && *name_substr) {
        wchar_t* wname = utf8_to_wchar(name_substr);
        if (wname) {
            if (!case_sensitive) {
                for (wchar_t* c = wname; *c; c++) *c = (wchar_t)towlower(*c);
            }
            config.name_substr = wname;
            config.name_substr_len = (int)wcslen(wname);
        }
    }
    
    if (path_substr && *path_substr) {
        wchar_t* wpath = utf8_to_wchar(path_substr);
        if (wpath) {
            if (!case_sensitive) {
                for (wchar_t* c = wpath; *c; c++) *c = (wchar_t)towlower(*c);
            }
            config.path_substr = wpath;
            config.path_substr_len = (int)wcslen(wpath);
        }
    }
    
    config.case_sensitive = case_sensitive;
    config.include_dirs = include_dirs;
    
    return config;
}

// 释放匹配配置
void free_match_config(MatchConfig* config) {
    free_wstring_list(&config->extensions);
    if (config->name_substr) free((void*)config->name_substr);
    if (config->path_substr) free((void*)config->path_substr);
    memset(config, 0, sizeof(MatchConfig));
}

// 检查扩展名是否匹配
int check_extension_match(const MatchConfig* config, const wchar_t* filename) {
    if (config->extensions.count == 0) return 1;
    
    size_t filename_len = wcslen(filename);
    wchar_t* filename_copy = NULL;
    const wchar_t* filename_to_compare = filename;  // 指向要比较的字符串
    
    // 仅在大小写不敏感时创建副本
    if (!config->case_sensitive) {
        filename_copy = _wcsdup(filename);
        if (!filename_copy) return 0;  // 内存分配失败
        
        // 转换为小写
        for (wchar_t* c = filename_copy; *c; c++) {
            *c = (wchar_t)towlower(*c);
        }
        filename_to_compare = filename_copy;
    }
    
    int match_found = 0;
    for (int i = 0; i < config->extensions.count; i++) {
        const wchar_t* ext = config->extensions.items[i];
        size_t ext_len = wcslen(ext);
        
        if (ext_len == 0) continue;
        
        if (filename_len >= ext_len) {
            const wchar_t* filename_end = filename_to_compare + filename_len - ext_len;
            
            if (wcsncmp(filename_end, ext, ext_len) == 0) {
                match_found = 1;
                break;
            }
        }
    }
    
    // 释放副本内存（如果有）
    if (filename_copy) {
        free(filename_copy);
    }
    
    return match_found;
}

// 检查文件名是否匹配
int check_name_match(const MatchConfig* config, const wchar_t* filename) {
    if (!config->name_substr) return 1;
    
    // 处理大小写
    wchar_t* fname_copy = _wcsdup(filename);
    if (!config->case_sensitive) {
        for (wchar_t* c = fname_copy; *c; c++) *c = (wchar_t)towlower(*c);
    }
    
    // 检查子串
    int match = (wcsstr(fname_copy, config->name_substr) != NULL);
    free(fname_copy);
    return match;
}

// 检查路径是否匹配
int check_path_match(const MatchConfig* config, const wchar_t* full_path) {
    if (!config->path_substr) return 1;
    
    // 处理大小写
    wchar_t* path_copy = _wcsdup(full_path);
    if (!config->case_sensitive) {
        for (wchar_t* c = path_copy; *c; c++) *c = (wchar_t)towlower(*c);
    }
    
    // 检查子串
    int match = (wcsstr(path_copy, config->path_substr) != NULL);
    free(path_copy);
    return match;
}

#ifdef _WIN32

// Windows Unicode 遍历实现
WStringList search_files_win(const wchar_t* root_path, const MatchConfig* config, int recursive) {
    WStringList results = {0};
    int stack_size = 0;
    int stack_capacity = 32;
    wchar_t** dir_stack = (wchar_t**)malloc(sizeof(wchar_t*) * stack_capacity);
    
    // 初始根目录
    dir_stack[0] = _wcsdup(root_path);
    stack_size = 1;
    
    // 路径缓冲区
    size_t path_buf_size = MAX_PATH * 4;
    wchar_t* current_path = (wchar_t*)malloc(path_buf_size * sizeof(wchar_t));
    wchar_t* find_path = (wchar_t*)malloc(path_buf_size * sizeof(wchar_t));
    
    while (stack_size > 0) {
        // 出栈当前目录
        wchar_t* current_dir = dir_stack[--stack_size];
        
        // 构造查找路径
        _snwprintf(find_path, path_buf_size, L"%s\\*", current_dir);
        
        WIN32_FIND_DATAW find_data;
        HANDLE hFind = FindFirstFileW(find_path, &find_data);
        
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                // 忽略当前目录和上级目录
                if (wcscmp(find_data.cFileName, L".") == 0 || 
                    wcscmp(find_data.cFileName, L"..") == 0) {
                    continue;
                }
                
                // 构造完整路径
                _snwprintf(current_path, path_buf_size, L"%s\\%s", current_dir, find_data.cFileName);
                
                // 判断文件类型
                BOOL is_dir = (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
                
                // 处理目录
                if (is_dir && recursive) {
                    // 入栈子目录
                    if (stack_size >= stack_capacity) {
                        stack_capacity *= 2;
                        dir_stack = (wchar_t**)realloc(dir_stack, sizeof(wchar_t*) * stack_capacity);
                    }
                    dir_stack[stack_size++] = _wcsdup(current_path);
                }
                
                // 检查是否需要处理目录项
                if (is_dir && !config->include_dirs) continue;
                
                // 快速文件名匹配检查
                if (config->name_substr) {
                    if (!check_name_match(config, find_data.cFileName)) {
                        continue;
                    }
                }
                
                // 扩展名匹配检查（针对文件）
                if (!is_dir && config->extensions.count > 0) {
                    if (!check_extension_match(config, find_data.cFileName)) {
                        continue;
                    }
                }
                
                // 完整路径匹配检查
                if (config->path_substr) {
                    if (!check_path_match(config, current_path)) {
                        continue;
                    }
                }
                
                // 添加到结果
                add_to_wstring_list(&results, current_path);
                
            } while (FindNextFileW(hFind, &find_data));
            
            FindClose(hFind);
        }
        
        free(current_dir);
    }
    
    free(current_path);
    free(find_path);
    free(dir_stack);
    
    return results;
}

#else

// POSIX Unicode 遍历实现
WStringList search_files_posix(const wchar_t* root_path, const MatchConfig* config, int recursive) {
    WStringList results = {0};
    int stack_size = 0;
    int stack_capacity = 32;
    wchar_t** dir_stack = (wchar_t**)malloc(sizeof(wchar_t*) * stack_capacity);
    
    // 初始根目录
    dir_stack[0] = wcsdup(root_path);
    stack_size = 1;
    
    // 路径缓冲区
    size_t path_buf_size = PATH_MAX * 4;
    wchar_t* current_path = (wchar_t*)malloc(path_buf_size * sizeof(wchar_t));
    
    while (stack_size > 0) {
        // 出栈当前目录
        wchar_t* current_dir = dir_stack[--stack_size];
        
        // 转换为 UTF-8 路径
        char* utf8_dir = wchar_to_utf8(current_dir);
        if (!utf8_dir) {
            free(current_dir);
            continue;
        }
        
        DIR* dir = opendir(utf8_dir);
        free(utf8_dir);
        
        if (!dir) {
            free(current_dir);
            continue;
        }
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            // 忽略当前目录和上级目录
            if (strcmp(entry->d_name, ".") == 0 || 
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            
            // 构造完整路径
            swprintf(current_path, path_buf_size, L"%ls/%hs", current_dir, entry->d_name);
            
            // 获取文件属性
            struct stat stat_buf;
            char* utf8_path = wchar_to_utf8(current_path);
            if (!utf8_path || stat(utf8_path, &stat_buf) != 0) {
                if (utf8_path) free(utf8_path);
                continue;
            }
            free(utf8_path);
            
            // 判断文件类型
            int is_dir = S_ISDIR(stat_buf.st_mode);
            
            // 处理目录
            if (is_dir && recursive) {
                // 入栈子目录
                if (stack_size >= stack_capacity) {
                    stack_capacity *= 2;
                    dir_stack = (wchar_t**)realloc(dir_stack, sizeof(wchar_t*) * stack_capacity);
                }
                dir_stack[stack_size++] = wcsdup(current_path);
            }
            
            // 检查是否需要处理目录项
            if (is_dir && !config->include_dirs) continue;
            
            // 转换文件名为宽字符
            wchar_t* wfilename = utf8_to_wchar(entry->d_name);
            if (!wfilename) continue;
            
            // 快速文件名匹配检查
            if (config->name_substr) {
                if (!check_name_match(config, wfilename)) {
                    free(wfilename);
                    continue;
                }
            }
            
            // 扩展名匹配检查（针对文件）
            if (!is_dir && config->extensions.count > 0) {
                if (!check_extension_match(config, wfilename)) {
                    free(wfilename);
                    continue;
                }
            }
            
            free(wfilename);
            
            // 完整路径匹配检查
            if (config->path_substr) {
                if (!check_path_match(config, current_path)) {
                    continue;
                }
            }
            
            // 添加到结果
            add_to_wstring_list(&results, current_path);
        }
        
        closedir(dir);
        free(current_dir);
    }
    
    free(current_path);
    free(dir_stack);
    
    return results;
}

#endif

// 主搜索函数
WStringList search_files(const wchar_t* root_path, 
                         const char* extensions,
                         const char* name_substr,
                         const char* path_substr,
                         int case_sensitive,
                         int recursive,
                         int include_dirs) {
    // 预编译匹配配置
    MatchConfig config = prepare_match_config(extensions, name_substr, path_substr, 
                                            case_sensitive, include_dirs);
    
    // 平台特定搜索
#ifdef _WIN32
    WStringList results = search_files_win(root_path, &config, recursive);
#else
    WStringList results = search_files_posix(root_path, &config, recursive);
#endif
    
    // 清理配置
    free_match_config(&config);
    
    return results;
}

// 将宽字符串列表合并为 UTF-8 字符串
char* merge_results_to_utf8(const WStringList* results) {
    // 计算总长度
    size_t total_len = 1; // 空结束符
    for (int i = 0; i < results->count; i++) {
        char* utf8 = wchar_to_utf8(results->items[i]);
        if (utf8) {
            total_len += strlen(utf8) + 1; // +1 换行符
            free(utf8);
        }
    }
    
    // 分配内存
    char* output = (char*)malloc(total_len);
    if (!output) return NULL;
    
    // 拼接字符串
    char* current = output;
    for (int i = 0; i < results->count; i++) {
        char* utf8 = wchar_to_utf8(results->items[i]);
        if (utf8) {
            size_t len = strlen(utf8);
            memcpy(current, utf8, len);
            current += len;
            *current++ = '\n';
            free(utf8);
        }
    }
    *current = '\0';
    
    return output;
}

// 导出接口
extern "C" {
    char* search_files_c(const char* root_path_utf8,
                        const char* extensions,
                        const char* name_substr,
                        const char* path_substr,
                        int case_sensitive,
                        int recursive,
                        int include_dirs) {
        // 转换根路径为宽字符
        wchar_t* wroot_path = utf8_to_wchar(root_path_utf8);
        if (!wroot_path) return NULL;
        
        // 执行搜索
        WStringList results = search_files(wroot_path, extensions, name_substr, path_substr,
                                        case_sensitive, recursive, include_dirs);
        
        // 清理根路径
        free(wroot_path);
        
        // 合并结果为 UTF-8
        char* output = merge_results_to_utf8(&results);
        
        // 清理中间结果
        free_wstring_list(&results);
        
        return output;
    }

    void free_search_result(char* result) {
        free(result);
    }
}


#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <fstream>


int main() {
    //获取运行时间
    auto start = std::chrono::high_resolution_clock::now();

    const char* root_path = "E:\\Env";
    const char* extensions = "exe";
    const char* name_substr = "";
    const char* path_substr = "";
    int case_sensitive = 1;
    int recursive_mode = 1;
    int include_dirs = 0;
    char* ret = search_files_c(root_path, extensions, name_substr, path_substr, case_sensitive, recursive_mode, include_dirs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time cost: " << duration.count()/1000 << " ms" << std::endl;

    // ret 写入文件
    if (ret) {
        std::ofstream outfile("../../../data_tmp/file_sf_result.txt", std::ios::binary);
        if (outfile) {
            outfile << ret;
            outfile.close();
            std::cout << "Result written to result.txt" << std::endl;
        } else {
            std::cerr << "Failed to open result.txt for writing." << std::endl;
        }
        free_search_result(ret);
    }
}