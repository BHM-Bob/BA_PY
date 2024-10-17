
if __name__ == '__main__':
    # dev mode
    from mbapy_lite.bio import peptide
else:
    # release mode
    from . import peptide