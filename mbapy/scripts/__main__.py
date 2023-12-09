
import argparse

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

if __name__ == '__main__':
    args_paser = argparse.ArgumentParser()
    group = args_paser.add_mutually_exclusive_group()
    group.add_argument("-l", "--list", action='store_true', help="print scripts list")
    group.add_argument("-i", "--info", action="store_true", help="print scripts info")
    args = args_paser.parse_args()
    
    print('\n *** mbapy.scripts.help *** ')
    
    if args.list:
        print_scripts_list()
    elif args.info:
        print_scripts_info()