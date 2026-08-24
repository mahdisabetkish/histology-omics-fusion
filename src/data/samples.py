"""Section-level metadata for the spatialLIBD DLPFC Visium experiment.

Twelve sections were collected from three neurotypical donors. Each donor
contributed two pairs of directly adjacent sections, and the two pairs were
taken 300 um apart in the tissue block (Maynard et al., Nat Neurosci 2021).

Because sections from the same donor are spatially and genetically dependent,
every split used in this project is defined at the donor level.
"""

SECTIONS = {
    "151507": {"donor": "Br5292", "position": 0},
    "151508": {"donor": "Br5292", "position": 0},
    "151509": {"donor": "Br5292", "position": 300},
    "151510": {"donor": "Br5292", "position": 300},
    "151669": {"donor": "Br5595", "position": 0},
    "151670": {"donor": "Br5595", "position": 0},
    "151671": {"donor": "Br5595", "position": 300},
    "151672": {"donor": "Br5595", "position": 300},
    "151673": {"donor": "Br8100", "position": 0},
    "151674": {"donor": "Br8100", "position": 0},
    "151675": {"donor": "Br8100", "position": 300},
    "151676": {"donor": "Br8100", "position": 300},
}

DONORS = ["Br5292", "Br5595", "Br8100"]

# Br8100 is never seen during fitting or checkpoint selection, so the headline
# test numbers describe transfer to a new brain rather than to a new section of
# a brain the model already knows.
TEST_DONOR = "Br8100"
TRAIN_DONORS = ["Br5292", "Br5595"]

# Validation is section-level rather than donor-level: with only three donors,
# spending one on early stopping costs a third of the training data. One
# section is withheld from each training donor instead. These sections share a
# donor with the training data, so validation scores run slightly optimistic
# and are used only to pick the stopping epoch, never to report performance.
VAL_SECTIONS = ["151510", "151672"]

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
LAYER_TO_INDEX = {name: i for i, name in enumerate(LAYERS)}

# Colours follow the palette used in the original spatialLIBD figures so the
# spatial maps in this repo can be compared against the published ones.
LAYER_COLORS = {
    "L1": "#F0027F",
    "L2": "#377EB8",
    "L3": "#4DAF4A",
    "L4": "#984EA3",
    "L5": "#FFD700",
    "L6": "#FF7F00",
    "WM": "#1A1A1A",
}


def sections_for(donors):
    """Return section ids belonging to any of `donors`, in a stable order."""
    keep = set(donors)
    return [s for s in sorted(SECTIONS) if SECTIONS[s]["donor"] in keep]


def donor_of(section_id):
    return SECTIONS[section_id]["donor"]


def split_sections():
    """Map each split name to the section ids it draws from."""
    val = list(VAL_SECTIONS)
    train = [s for s in sections_for(TRAIN_DONORS) if s not in val]
    test = sections_for([TEST_DONOR])
    return {"train": train, "val": val, "test": test}


def split_of(section_id):
    for name, ids in split_sections().items():
        if section_id in ids:
            return name
    raise KeyError(section_id)
