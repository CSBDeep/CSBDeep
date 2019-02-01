#!/usr/bin/env python

import sys
from zipfile import ZipFile
from pathlib import Path

suffix_out = '.contents'

def create_contents(f):
    file_zip = Path(f)
    # remove suffix_out if present
    if len(file_zip.suffixes) > 0 and file_zip.suffixes[-1] == suffix_out:
        file_zip = Path(file_zip.stem)
    file_out = file_zip.with_suffix(''.join(file_zip.suffixes+[suffix_out]))
    assert file_zip.exists() and file_zip.is_file()

    with ZipFile(file_zip) as zfile:
        with file_out.open('w') as ofile:
            for entry in zfile.infolist():
                print("%10d\t%s" % (entry.file_size, entry.filename), file=ofile)

    print(f"written: {file_out.resolve()}")


if __name__ == '__main__':
    '__file__' in locals() or '__file__' in globals() or sys.exit(0)
    if len(sys.argv) == 1:
        print(f"usage: {sys.argv[0]} [zip files...]")
        sys.exit(0)

    for f in sys.argv[1:]:
        f = Path(f)
        if f.exists() and f.is_file() and '.zip' in f.suffixes:
            create_contents(f)
