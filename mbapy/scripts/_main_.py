import importlib
import os
import sys

os.environ['MBAPY_FAST_LOAD'] = 'True'
os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'

scripts_info = {
    'cnipa':
        {
            'brief': 'get patents info from CNIPA.',
            'detailed': 
                """
                If it accessible, get info and store it into a json session file.
                Example: python -m mbapy.scripts.cnipa -q "peptide" -o "E:\\peptide patents" -log
                """,
        },
    'scihub':
        {
            'brief': 'download papers and it\'s refs(optional) from scihub.',
            'detailed': 
                """
                If it can be downloaded, download and store it into session.
                Example: python -m mbapy.scripts.scihub -i "E:\\peptide.ris" -o "E:\\peptide papers" -log
                """,
        },
    'scihub_selenium':
        {
            'brief': 'download papers and it\'s refs(optional) from scihub using selenium.',
            'detailed': 
                """
                If it can be downloaded, download and store it into session.
                Example: python -m mbapy.scripts.scihub_selenium -i "E:\\peptide.ris" -o "E:\\peptide papers" -log -gui
                """,
        },
    'extract_paper':
        {
            'brief': 'extract papers content to a json file.',
            'detailed': 
                """
                If it can be parsed to sections, output each sections separatly, else all text content.
                Example: python -m mbapy.scripts.extract_paper -i "E:\\peptide papers" -log
                """,
        },
    'peptide':
        {
            'brief': 'tools for peptide.',
            'detailed':
                """
                sub command 1: subval:
                    : calcu SPPS substitution value for a release test of resin.
                sub command 2: mw:
                    : calcu MW and Exact Mass for a peptide.
                sub command 1: mmw:
                    : calcu MW of each peptide mutations syn by SPPS.
                """
        }
}


def print_scripts_list():
    for idx, script in enumerate(scripts_info):
        print(f'scripts {idx:3d}: {script}')
        print(scripts_info[script]['brief'])
        print('-'*100)

def print_scripts_info():
    for idx, script in enumerate(scripts_info):
        print(f'scripts {idx:3d}: {script}')
        print(scripts_info[script]['brief'])
        print(scripts_info[script]['detailed'])
        print('-'*100)
        
def exec_scripts():
    import mbapy

    # append '-h' if only input 'mbapy-cli sub-script'
    sys.argv.extend(['-h'] if len(sys.argv) == 2 else [])
    # NOTE: DO NOT use exec
    # check and exec scripts
    script = importlib.import_module(f'.{sys.argv[1]}', 'mbapy.scripts')
    script.main(sys.argv[2:])
    
def main():    
    if len(sys.argv) == 1:
        import mbapy
        print('mbapy python package command-line tools')
        print('mbapy version: ', mbapy.__version__, ', build: ', mbapy.__build__)
        print('mbapy author: ', mbapy.__author__, ', email: ', mbapy.__author_email__)
        print('mbapy url: ', mbapy.__url__, ', license: ', mbapy.__license__)
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-l', '--list']:
            print_scripts_list()
        elif sys.argv[1] in ['-i', '--info']:
            print_scripts_info()
        elif sys.argv[1] in ['-h', '--help']:
            help = """
            usage-1: mbapy-cli [-h] [-l | -i]
            options:
            -h, --help  show this help message and exit
            -l, --list  print scripts list
            -i, --info  print scripts info
            
            usage-2: mbapy-cli [sub-scripts-name] [args] [-h]
            options:
            sub-scripts-name  name of scripts in mbapy.scripts
            args  args for sub-scripts
            -h, --help  show this help message and exit
            """
            print(help)
        elif sys.argv[1] in scripts_info:
            # exec scripts only with arg '-h'
            exec_scripts()
    else:
        if sys.argv[1] in scripts_info:
            # exec scripts
            exec_scripts()
        else:
            print(f'unkown scripts: {sys.argv[1]} and args: ', end='')
            [print(f' {arg}', end='') for arg in sys.argv[1:]]
            print('')
            

if __name__ == '__main__':
    main()