"""Provides a parser for the output files of bcftools stats.
"""

from collections.abc import Iterable
from os import PathLike
from typing import Union

import pandas as pd


TableRow = list[Union[int, float, str]]


def parse_description(line: str, identifier: str) -> str:
    """Parses description of a table.

    Parameters
    ----------
    line : str
        Line to parse.
    identifier : str
        Table identifier to remove from description.

    Returns
    -------
    description : str
        Description of table.

    Examples
    --------
    >>> parse_description("# Definition of sets:\\n", "ID")
    'Definition of sets'

    >>> parse_description("# TSTV, transitions/transversions:\\n", "TSTV")
    'transitions/transversions'

    >>> parse_description("# HWE\\n", "HWE")
    'HWE'
    """
    return line.strip().strip("#: ").removeprefix(f"{identifier}, ")


def parse_header(line: str) -> list[str]:
    """Parses table header.

    Parameters
    ----------
    line : str
        Tab-separated line to parse.

    Returns
    -------
    columns : list[str]
        List of column names. Anything contained within square brakets will be removed.

    Examples
    --------
    >>> parse_header("# SN\\t[2]id\\t[3]key\\t[4]value\\n")
    ['SN', 'id', 'key', 'value']
    """
    return [s.split("]")[-1] for s in line.strip("# \n").split("\t")]


def parse_tline(line: str) -> TableRow:
    """Parses a line of a table.

    Parameters
    ----------
    line : str
        Line to parse.

    Returns
    -------
    values : list[Union[int, float, str]]
        Parsed values.

    Examples
    --------
    >>> parse_tline("SN\\t0\\tnumber of samples:\\t78\\n")
    [0, 'number of samples', 78]

    >>> parse_tline("PSI\\t0\\tCW101M\\t0\\t0\\t0\\t0.00\\t427550\\t763806\\t359721\\t654868\\n")
    [0, 'CW101M', 0, 0, 0, 0.0, 427550, 763806, 359721, 654868]
    """
    row = line.strip().split("\t")[1:]
    values = []
    v_: Union[float, str]
    for v in row:
        if v.isdecimal():
            values.append(int(v))
        else:
            try:
                v_ = float(v)
            except ValueError:
                v_ = v.strip(":")
            values.append(v_)

    return values


class VCHK:
    def __init__(self, lines: Iterable):
        self._lines = iter(lines)

        self._description_line: str = ""
        self._header_line: str = ""

        self._dfs: dict[str, tuple[str, pd.DataFrame]] = {}
        self._in_table = False
        self._table: list[TableRow] = []

    def _save_df(self):
        header = parse_header(self._header_line)
        description = parse_description(self._description_line, header[0])
        self._dfs[header[0]] = description, pd.DataFrame(
            self._table, columns=header[1:]
        )

        self._table = []
        self._header_line = ""
        self._description_line = ""
        self._in_table = False

    def _hashline(self, line: str):
        if self._in_table:
            self._save_df()

        if line.strip().endswith(":") or f"{line} _".split()[1].endswith(","):
            self._description_line = line
        else:
            self._header_line = line

    def _tline(self, line: str):
        row = parse_tline(line)
        if row:
            self._in_table = True
            self._table.append(row)

    def _next_line(self):
        try:
            line = next(self._lines)
        except StopIteration:
            self._save_df()
            raise

        if line.startswith("#"):
            self._hashline(line)
        else:
            self._tline(line)

    def parse(self):
        """Parses vchk file.

        Returns
        -------
        dfs : dict[str, pd.DataFrame]
            DataFrames containing data from vchk file.
        """
        while True:
            try:
                self._next_line()
            except StopIteration:
                break

        return self._dfs


def load_vchk(file: PathLike) -> dict[str, pd.DataFrame]:
    """Parses a vchk file.

    Parameters
    ----------
    file : PathLike
        Path to vchk file

    Returns
    -------
    dfs : dict[str, pd.DataFrame]
        DataFrames containing data from vchk file.
    """
    with open(file, "r") as f:
        dfs = VCHK(f).parse()

    return dfs
