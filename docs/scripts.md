# mbapy.scripts

## 安装
确保你已经安装了 Python 3，并且已经安装了 argparse 模块。

## 用法
```bash
python -m mbapy.scripts.help [选项]
```

## 选项
- `-l`, `--list`: 打印可用脚本列表和简要描述。
- `-i`, `--info`: 打印可用脚本的详细描述。

## 示例
##### 打印脚本列表
```bash
python -m mbapy.scripts.help -l
```

##### 打印脚本详细信息
```bash
python -m mbapy.scripts.help -i
```

## 可用脚本列表
### cnipa
从 CNIPA 获取专利信息。
##### 安装
确保已安装以下 Python 模块：
- easyocr
- numpy
- pyautogui

##### 使用方法
```bash
python cnipa_script.py -q "query" -o "output_directory" -m "model_path" -l
```

##### 参数说明
- `-q`, `--query`: 检索词
- `-o`, `--out`: 输出文件目录
- `-m`, `--model_path`: EasyOCR 模型目录（可选）
- `-l`, `--log`: 启用日志记录（可选）

##### 示例
```bash
python cnipa_script.py -q "peptide" -o "E:\\peptide_patents" -m "E:\\easyocr_models" -l
```

##### 注意事项
- 该脚本会从 CNIPA 下载专利信息，并在每个条目成功后保存一次。
- 请确保已安装 Chrome 浏览器，并且已配置好 Chrome WebDriver以及undetected_chromedriver。
- 请确保已准备好验证码识别所需的模型文件（如果使用了自定义模型）。
- 脚本执行过程中保持浏览器窗口最大化并且为置顶状态。

### sciHub
从 SciHub 下载论文及其引用（可选）。
##### 安装
确保已安装以下 Python 模块：
- tqdm  

##### 使用方法
```bash
python scihub_script.py -i "ris_file_path" -o "output_directory" -r -l
```

##### 参数说明
- `-i`, `--ris`: RIS 文件路径
- `-o`, `--out`: 输出文件目录
- `-r`, `--ref`: 启用引用模式以下载引用（可选）
- `-l`, `--log`: 启用日志记录（可选）

##### 示例
```bash
python scihub_script.py -i "E:\\peptide.ris" -o "E:\\peptide_papers" -r -l
```

##### 注意事项
- 该脚本会从 SciHub 下载论文及其引用（如果启用了引用模式）。
- 在下载过程中，依次按下 "e" + "Enter" 键可以停止并保存会话以便下次启动时恢复进度。

### sciHub_selenium
使用 Selenium 从 SciHub 下载论文及其引用（可选）。
##### 安装
确保已安装以下 Python 模块：
- requests
- tqdm
- wget

##### 使用方法
```bash
python scihub_selenium_script.py -i "ris_file_path" -o "output_directory" -r -g -u -l
```

##### 参数说明
- `-i`, `--ris`: RIS 文件路径
- `-o`, `--out`: 输出文件目录
- `-r`, `--ref`: 启用引用模式以下载引用（可选）
- `-g`, `--gui`: 启用浏览器 GUI（可选）
- `-u`, `--undetected`: 启用使用 undetected_chromedriver（可选）
- `-l`, `--log`: 启用日志记录（可选）

##### 示例
```bash
python scihub_selenium_script.py -i "E:\\peptide.ris" -o "E:\\peptide_papers" -r -g -u -l
```

##### 注意事项
- 该脚本会从 SciHub 下载论文及其引用（如果启用了引用模式）。
- 请确保已安装 Chrome 浏览器，并且已配置好 Chrome WebDriver。
- 在下载过程中，依次按下 "e" + "Enter" 键可以停止并保存会话以便下次启动时恢复进度。


### extract_paper
提取论文内容到 JSON 文件。
##### 安装
确保已安装以下 Python 模块：
- tqdm

##### 使用方法
```bash
python extract_paper_script.py -i "input_directory" -o "output_file_name" -b "backend" -l
```

##### 参数说明
- `-i`, `--input`: 输入论文（PDF）文件目录
- `-o`, `--output`: 输出文件名，默认为 `_mbapy_extract_paper.json`
- `-b`, `--backend`: 指定后端解析器，默认为 `pdfminer`
- `-l`, `--log`: 启用日志记录（可选）

##### 示例
```bash
python extract_paper_script.py -i "E:\\peptide_papers" -o "peptide_extracted.json" -b "pdfplumber" -l
```

##### 注意事项
- 该脚本用于提取论文内容，并将结果保存为 JSON 文件。
- 如果论文包含书签，将会提取书签中的英文部分作为论文的章节信息。
- 如果论文无法解析或提取书签信息，将会将整篇论文内容保存为字符串。
- 请确保已安装相应的 PDF 解析器（如 pdfminer 或 pdfplumber）。

