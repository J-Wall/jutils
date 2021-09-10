def normalise_name(samplename: str, suf: str = "") -> str:
    normalised = []

    samplename = samplename.removesuffix(suf)
    zero_idx = samplename.find("0")

    if zero_idx == -1:
        return samplename

    else:
        return "".join((samplename[:zero_idx], samplename[zero_idx:].lstrip("0")))
