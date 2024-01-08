
if __name__ == '__main__':
    # dev mode
    from mbapy.bio import peptide
else:
    # release mode
    from . import peptide