import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

import gemmi

from alphafold3tools.structure.mmcif_utils import (
    _process_altloc_group_partial,
    _process_altloc_groups_whole,
    fix_arginine_residues,
    fix_structure,
    mse_to_met,
    resolve_mmcif_altlocs,
)


def create_test_arginine_structure():
    """Create a test structure with an arginine residue with swapped NH1/NH2."""
    # Create a simple mmCIF structure with an arginine residue
    # NH1 and NH2 are intentionally swapped to test the fix
    cif_content = textwrap.dedent("""\
                data_TEST
                #
                loop_
                _atom_site.group_PDB
                _atom_site.id
                _atom_site.type_symbol
                _atom_site.label_atom_id
                _atom_site.label_alt_id
                _atom_site.label_comp_id
                _atom_site.label_asym_id
                _atom_site.label_entity_id
                _atom_site.label_seq_id
                _atom_site.pdbx_PDB_ins_code
                _atom_site.Cartn_x
                _atom_site.Cartn_y
                _atom_site.Cartn_z
                _atom_site.occupancy
                _atom_site.B_iso_or_equiv
                _atom_site.auth_seq_id
                _atom_site.auth_comp_id
                _atom_site.auth_asym_id
                _atom_site.auth_atom_id
                _atom_site.pdbx_PDB_model_num
                ATOM   1    N  N   . ARG A 1 1   ? 10.0 10.0 10.0 1.0 20.0 1   ARG A N   1
                ATOM   2    C  CA  . ARG A 1 1   ? 11.0 10.0 10.0 1.0 20.0 1   ARG A CA  1
                ATOM   3    C  C   . ARG A 1 1   ? 12.0 10.0 10.0 1.0 20.0 1   ARG A C   1
                ATOM   4    O  O   . ARG A 1 1   ? 13.0 10.0 10.0 1.0 20.0 1   ARG A O   1
                ATOM   5    C  CB  . ARG A 1 1   ? 11.0 11.0 10.0 1.0 20.0 1   ARG A CB  1
                ATOM   6    C  CG  . ARG A 1 1   ? 11.0 12.0 10.0 1.0 20.0 1   ARG A CG  1
                ATOM   7    C  CD  . ARG A 1 1   ? 11.0 13.0 10.0 1.0 20.0 1   ARG A CD  1
                ATOM   8    N  NE  . ARG A 1 1   ? 11.0 14.0 10.0 1.0 20.0 1   ARG A NE  1
                ATOM   9    C  CZ  . ARG A 1 1   ? 11.0 15.0 10.0 1.0 20.0 1   ARG A CZ  1
                ATOM   10   N  NH1 . ARG A 1 1   ? 11.0 17.0 10.0 1.0 20.0 1   ARG A NH1 1
                ATOM   11   N  NH2 . ARG A 1 1   ? 11.0 16.0 10.0 1.0 20.0 1   ARG A NH2 1
                #
                """)
    return cif_content


def test_fix_arginine_residues_basic():
    """Test that fix_arginine_residues correctly swaps NH1 and NH2 when needed."""
    # Create test structure
    cif_content = create_test_arginine_structure()

    # Write to temporary file and read
    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)

        struc = gemmi.read_structure(str(cif_path))

        # Get the arginine residue
        model = struc[0]
        chain = model[0]
        residue = chain[0]

        # Check initial state (NH1 should be farther from CD than NH2)
        cd = residue.find_atom("CD", "*")
        nh1_before = residue.find_atom("NH1", "*")
        nh2_before = residue.find_atom("NH2", "*")

        assert cd is not None
        assert nh1_before is not None
        assert nh2_before is not None

        # Calculate distances before fix
        def dist_squared(pos1, pos2):
            dx = pos1.x - pos2.x
            dy = pos1.y - pos2.y
            dz = pos1.z - pos2.z
            return dx * dx + dy * dy + dz * dz

        dist_cd_nh1_before = dist_squared(cd.pos, nh1_before.pos)
        dist_cd_nh2_before = dist_squared(cd.pos, nh2_before.pos)

        # NH1 should be farther from CD than NH2 (incorrect state)
        assert dist_cd_nh1_before > dist_cd_nh2_before

        # Apply fix
        fix_arginine_residues(residue)

        # Check after fix
        nh1_after = residue.find_atom("NH1", "*")
        nh2_after = residue.find_atom("NH2", "*")

        dist_cd_nh1_after = dist_squared(cd.pos, nh1_after.pos)
        dist_cd_nh2_after = dist_squared(cd.pos, nh2_after.pos)

        # After fix, NH1 should be closer to CD than NH2
        assert dist_cd_nh1_after < dist_cd_nh2_after


def test_fix_structure():
    """Test fix_structure function with arginine fixing enabled."""
    cif_content = create_test_arginine_structure()

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)

        struc = gemmi.read_structure(str(cif_path))

        # Apply fix_structure with arginine fixing
        fix_structure(struc, fix_arginine=True)

        # Get the arginine residue
        residue = struc[0][0][0]

        cd = residue.find_atom("CD", "*")
        nh1 = residue.find_atom("NH1", "*")
        nh2 = residue.find_atom("NH2", "*")

        # Calculate distances
        def dist_squared(pos1, pos2):
            dx = pos1.x - pos2.x
            dy = pos1.y - pos2.y
            dz = pos1.z - pos2.z
            return dx * dx + dy * dy + dz * dz

        dist_cd_nh1 = dist_squared(cd.pos, nh1.pos)
        dist_cd_nh2 = dist_squared(cd.pos, nh2.pos)

        # NH1 should be closer to CD after fix
        assert dist_cd_nh1 < dist_cd_nh2


def test_fix_arginine_no_fix_needed():
    """Test that fix_arginine_residues does nothing when NH1/NH2 are correct."""
    # Create a structure where NH1 is already closer to CD
    cif_content = textwrap.dedent("""data_TEST
                    #
                    loop_
                    _atom_site.group_PDB
                    _atom_site.id
                    _atom_site.type_symbol
                    _atom_site.label_atom_id
                    _atom_site.label_alt_id
                    _atom_site.label_comp_id
                    _atom_site.label_asym_id
                    _atom_site.label_entity_id
                    _atom_site.label_seq_id
                    _atom_site.pdbx_PDB_ins_code
                    _atom_site.Cartn_x
                    _atom_site.Cartn_y
                    _atom_site.Cartn_z
                    _atom_site.occupancy
                    _atom_site.B_iso_or_equiv
                    _atom_site.auth_seq_id
                    _atom_site.auth_comp_id
                    _atom_site.auth_asym_id
                    _atom_site.auth_atom_id
                    _atom_site.pdbx_PDB_model_num
                    ATOM   1    N  N   . ARG A 1 1   ? 10.0 10.0 10.0 1.0 20.0 1   ARG A N   1
                    ATOM   2    C  CA  . ARG A 1 1   ? 11.0 10.0 10.0 1.0 20.0 1   ARG A CA  1
                    ATOM   3    C  C   . ARG A 1 1   ? 12.0 10.0 10.0 1.0 20.0 1   ARG A C   1
                    ATOM   4    O  O   . ARG A 1 1   ? 13.0 10.0 10.0 1.0 20.0 1   ARG A O   1
                    ATOM   5    C  CB  . ARG A 1 1   ? 11.0 11.0 10.0 1.0 20.0 1   ARG A CB  1
                    ATOM   6    C  CG  . ARG A 1 1   ? 11.0 12.0 10.0 1.0 20.0 1   ARG A CG  1
                    ATOM   7    C  CD  . ARG A 1 1   ? 11.0 13.0 10.0 1.0 20.0 1   ARG A CD  1
                    ATOM   8    N  NE  . ARG A 1 1   ? 11.0 14.0 10.0 1.0 20.0 1   ARG A NE  1
                    ATOM   9    C  CZ  . ARG A 1 1   ? 11.0 15.0 10.0 1.0 20.0 1   ARG A CZ  1
                    ATOM   10   N  NH1 . ARG A 1 1   ? 11.0 16.0 10.0 1.0 20.0 1   ARG A NH1 1
                    ATOM   11   N  NH2 . ARG A 1 1   ? 11.0 17.0 10.0 1.0 20.0 1   ARG A NH2 1
                    #
                    """)

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)

        struc = gemmi.read_structure(str(cif_path))
        residue = struc[0][0][0]

        # Store original positions
        nh1_before = residue.find_atom("NH1", "*")
        nh2_before = residue.find_atom("NH2", "*")
        nh1_pos_before = (nh1_before.pos.x, nh1_before.pos.y, nh1_before.pos.z)
        nh2_pos_before = (nh2_before.pos.x, nh2_before.pos.y, nh2_before.pos.z)

        # Apply fix
        fix_arginine_residues(residue)

        # Check that positions didn't change (names didn't swap)
        nh1_after = residue.find_atom("NH1", "*")
        nh2_after = residue.find_atom("NH2", "*")
        nh1_pos_after = (nh1_after.pos.x, nh1_after.pos.y, nh1_after.pos.z)
        nh2_pos_after = (nh2_after.pos.x, nh2_after.pos.y, nh2_after.pos.z)

        # Positions should remain the same (no swap occurred)
        assert nh1_pos_before == nh1_pos_after
        assert nh2_pos_before == nh2_pos_after


def test_fix_arginine_non_arginine_residue():
    """Test that fix_arginine_residues does nothing for non-arginine residues."""
    cif_content = textwrap.dedent("""data_TEST
                    #
                    loop_
                    _atom_site.group_PDB
                    _atom_site.id
                    _atom_site.type_symbol
                    _atom_site.label_atom_id
                    _atom_site.label_alt_id
                    _atom_site.label_comp_id
                    _atom_site.label_asym_id
                    _atom_site.label_entity_id
                    _atom_site.label_seq_id
                    _atom_site.pdbx_PDB_ins_code
                    _atom_site.Cartn_x
                    _atom_site.Cartn_y
                    _atom_site.Cartn_z
                    _atom_site.occupancy
                    _atom_site.B_iso_or_equiv
                    _atom_site.auth_seq_id
                    _atom_site.auth_comp_id
                    _atom_site.auth_asym_id
                    _atom_site.auth_atom_id
                    _atom_site.pdbx_PDB_model_num
                    ATOM   1    N  N   . ALA A 1 1   ? 10.0 10.0 10.0 1.0 20.0 1   ALA A N   1
                    ATOM   2    C  CA  . ALA A 1 1   ? 11.0 10.0 10.0 1.0 20.0 1   ALA A CA  1
                    ATOM   3    C  C   . ALA A 1 1   ? 12.0 10.0 10.0 1.0 20.0 1   ALA A C   1
                    ATOM   4    O  O   . ALA A 1 1   ? 13.0 10.0 10.0 1.0 20.0 1   ALA A O   1
                    ATOM   5    C  CB  . ALA A 1 1   ? 11.0 11.0 10.0 1.0 20.0 1   ALA A CB  1
                    #
                    """)

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)

        struc = gemmi.read_structure(str(cif_path))
        residue = struc[0][0][0]

        # Should not raise any errors
        fix_arginine_residues(residue)

        # Residue should still be ALA
        assert residue.name == "ALA"


def create_test_resolve_mmcif_altlocs():
    """Create a test structure with double altlocs for testing resolve_mmcif_altlocs.
    There are two residues at position 42 (GLU) and 58 (SEP/SER) with altlocs A and B.
    In this case, SER58(B) with occupancy 0.60 should be preferred over SEP58(A) with occupancy 0.40.
    """
    cif_content = textwrap.dedent("""\
            data_5ZA2
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            ATOM   303  N N   . ARG A 1 41  ? -10.762 -10.321 -24.400 1.00 21.14 ? 48  ARG A N   1
            ATOM   304  C CA  . ARG A 1 41  ? -11.277 -10.465 -23.044 1.00 21.73 ? 48  ARG A CA  1
            ATOM   305  C C   . ARG A 1 41  ? -12.729 -10.021 -22.924 1.00 23.67 ? 48  ARG A C   1
            ATOM   306  O O   . ARG A 1 41  ? -13.455 -10.546 -22.072 1.00 27.59 ? 48  ARG A O   1
            ATOM   307  C CB  . ARG A 1 41  ? -10.412 -9.682  -22.064 1.00 24.16 ? 48  ARG A CB  1
            ATOM   308  C CG  . ARG A 1 41  ? -9.150  -10.422 -21.663 1.00 31.26 ? 48  ARG A CG  1
            ATOM   309  C CD  . ARG A 1 41  ? -8.512  -9.712  -20.494 1.00 33.01 ? 48  ARG A CD  1
            ATOM   310  N NE  . ARG A 1 41  ? -8.835  -8.289  -20.521 1.00 41.54 ? 48  ARG A NE  1
            ATOM   311  C CZ  . ARG A 1 41  ? -8.250  -7.379  -19.755 1.00 41.82 ? 48  ARG A CZ  1
            ATOM   312  N NH1 . ARG A 1 41  ? -7.305  -7.750  -18.899 1.00 36.04 ? 48  ARG A NH1 1
            ATOM   313  N NH2 . ARG A 1 41  ? -8.603  -6.103  -19.854 1.00 36.61 ? 48  ARG A NH2 1
            ATOM   314  N N   . GLU A 1 42  ? -13.185 -9.074  -23.754 1.00 23.51 ? 49  GLU A N   1
            ATOM   315  C CA  A GLU A 1 42  ? -14.547 -8.571  -23.581 0.51 23.21 ? 49  GLU A CA  1
            ATOM   316  C CA  B GLU A 1 42  ? -14.546 -8.581  -23.571 0.49 23.22 ? 49  GLU A CA  1
            ATOM   317  C C   . GLU A 1 42  ? -15.576 -9.512  -24.197 1.00 20.35 ? 49  GLU A C   1
            ATOM   318  O O   . GLU A 1 42  ? -16.669 -9.686  -23.647 1.00 23.01 ? 49  GLU A O   1
            ATOM   319  C CB  A GLU A 1 42  ? -14.680 -7.152  -24.154 0.51 23.25 ? 49  GLU A CB  1
            ATOM   320  C CB  B GLU A 1 42  ? -14.688 -7.167  -24.134 0.49 23.25 ? 49  GLU A CB  1
            ATOM   321  C CG  A GLU A 1 42  ? -14.542 -6.975  -25.665 0.51 23.82 ? 49  GLU A CG  1
            ATOM   322  C CG  B GLU A 1 42  ? -13.776 -6.133  -23.469 0.49 22.16 ? 49  GLU A CG  1
            ATOM   323  C CD  A GLU A 1 42  ? -14.833 -5.539  -26.111 0.51 24.65 ? 49  GLU A CD  1
            ATOM   324  C CD  B GLU A 1 42  ? -14.085 -5.861  -21.990 0.49 25.01 ? 49  GLU A CD  1
            ATOM   325  O OE1 A GLU A 1 42  ? -15.114 -4.697  -25.235 0.51 29.69 ? 49  GLU A OE1 1
            ATOM   326  O OE1 B GLU A 1 42  ? -15.060 -6.414  -21.417 0.49 22.79 ? 49  GLU A OE1 1
            ATOM   327  O OE2 A GLU A 1 42  ? -14.783 -5.240  -27.329 0.51 18.21 ? 49  GLU A OE2 1
            ATOM   328  O OE2 B GLU A 1 42  ? -13.335 -5.064  -21.393 0.49 23.67 ? 49  GLU A OE2 1
            ATOM   329  N N   . SER A 1 43  ? -15.252 -10.131 -25.327 1.00 21.22 ? 50  SER A N   1
            ATOM   330  C CA  . SER A 1 43  ? -16.169 -11.092 -25.920 1.00 21.09 ? 50  SER A CA  1
            ATOM   331  C C   . SER A 1 43  ? -15.986 -12.496 -25.360 1.00 23.05 ? 50  SER A C   1
            ATOM   332  O O   . SER A 1 43  ? -16.933 -13.299 -25.389 1.00 22.06 ? 50  SER A O   1
            ATOM   333  C CB  . SER A 1 43  ? -15.983 -11.112 -27.437 1.00 19.72 ? 50  SER A CB  1
            ATOM   334  O OG  . SER A 1 43  ? -14.711 -11.659 -27.753 1.00 21.06 ? 50  SER A OG  1
            ATOM   3145 N N   . GLY A 1 57  ? 26.651  4.429   3.669   1.00 18.24 ? 63  GLY A N   1
            ATOM   3146 C CA  . GLY A 1 57  ? 26.090  5.576   2.975   1.00 18.15 ? 63  GLY A CA  1
            ATOM   3147 C C   . GLY A 1 57  ? 25.791  5.259   1.523   1.00 17.43 ? 63  GLY A C   1
            ATOM   3148 O O   . GLY A 1 57  ? 25.232  4.208   1.196   1.00 18.91 ? 63  GLY A O   1
            HETATM 3149 N N   A SEP A 1 58  ? 26.196  6.180   0.656   0.40 19.35 ? 64  SEP A N   1
            HETATM 3150 C CA  A SEP A 1 58  ? 25.897  6.120   -0.774  0.40 18.88 ? 64  SEP A CA  1
            HETATM 3151 C CB  A SEP A 1 58  ? 26.384  7.385   -1.481  0.40 19.93 ? 64  SEP A CB  1
            HETATM 3152 O OG  A SEP A 1 58  ? 25.458  8.445   -1.335  0.40 23.31 ? 64  SEP A OG  1
            HETATM 3153 C C   A SEP A 1 58  ? 26.438  4.899   -1.513  0.40 17.48 ? 64  SEP A C   1
            HETATM 3154 O O   A SEP A 1 58  ? 26.004  4.622   -2.636  0.40 17.93 ? 64  SEP A O   1
            HETATM 3155 P P   A SEP A 1 58  ? 26.111  9.615   -0.444  0.40 22.18 ? 64  SEP A P   1
            HETATM 3156 O O1P A SEP A 1 58  ? 27.046  10.520  -1.389  0.40 18.80 ? 64  SEP A O1P 1
            HETATM 3157 O O2P A SEP A 1 58  ? 24.902  10.458  0.195   0.40 22.19 ? 64  SEP A O2P 1
            HETATM 3158 O O3P A SEP A 1 58  ? 27.023  8.969   0.701   0.40 19.31 ? 64  SEP A O3P 1
            ATOM   3159 N N   B SER A 1 58  ? 26.205  6.158   0.638   0.60 19.37 ? 64  SER A N   1
            ATOM   3160 C CA  B SER A 1 58  ? 25.820  6.081   -0.766  0.60 18.85 ? 64  SER A CA  1
            ATOM   3161 C C   B SER A 1 58  ? 26.429  4.900   -1.510  0.60 17.49 ? 64  SER A C   1
            ATOM   3162 O O   B SER A 1 58  ? 26.026  4.646   -2.652  0.60 17.92 ? 64  SER A O   1
            ATOM   3163 C CB  B SER A 1 58  ? 26.200  7.375   -1.473  0.60 19.91 ? 64  SER A CB  1
            ATOM   3164 O OG  B SER A 1 58  ? 25.417  8.448   -0.980  0.60 23.21 ? 64  SER A OG  1
            ATOM   3165 N N   . VAL A 1 59  ? 27.372  4.168   -0.910  1.00 17.62 ? 65  VAL A N   1
            ATOM   3166 C CA  . VAL A 1 59  ? 27.783  2.897   -1.496  1.00 17.62 ? 65  VAL A CA  1
            ATOM   3167 C C   . VAL A 1 59  ? 26.577  1.973   -1.603  1.00 16.68 ? 65  VAL A C   1
            ATOM   3168 O O   . VAL A 1 59  ? 26.536  1.092   -2.472  1.00 17.48 ? 65  VAL A O   1
            ATOM   3169 C CB  . VAL A 1 59  ? 28.932  2.257   -0.687  1.00 17.68 ? 65  VAL A CB  1
            ATOM   3170 C CG1 . VAL A 1 59  ? 29.307  0.860   -1.253  1.00 18.45 ? 65  VAL A CG1 1
            ATOM   3171 C CG2 . VAL A 1 59  ? 30.140  3.183   -0.684  1.00 17.83 ? 65  VAL A CG2 1
            """)
    return cif_content


def create_test_resolve_mmcif_altlocs2():
    """Create a test structure with double altlocs for testing resolve_mmcif_altlocs.
    There are two residues at position 42 (GLU) and 58 (SEP/SER) with altlocs A and B.
    In this case, SEP58 has higher occupancy than SER58, then SEP58 should be kept.
    """
    cif_content = textwrap.dedent("""\
            data_5ZA2
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            ATOM   303  N N   . ARG A 1 41  ? -10.762 -10.321 -24.400 1.00 21.14 ? 48  ARG A N   1
            ATOM   304  C CA  . ARG A 1 41  ? -11.277 -10.465 -23.044 1.00 21.73 ? 48  ARG A CA  1
            ATOM   305  C C   . ARG A 1 41  ? -12.729 -10.021 -22.924 1.00 23.67 ? 48  ARG A C   1
            ATOM   306  O O   . ARG A 1 41  ? -13.455 -10.546 -22.072 1.00 27.59 ? 48  ARG A O   1
            ATOM   307  C CB  . ARG A 1 41  ? -10.412 -9.682  -22.064 1.00 24.16 ? 48  ARG A CB  1
            ATOM   308  C CG  . ARG A 1 41  ? -9.150  -10.422 -21.663 1.00 31.26 ? 48  ARG A CG  1
            ATOM   309  C CD  . ARG A 1 41  ? -8.512  -9.712  -20.494 1.00 33.01 ? 48  ARG A CD  1
            ATOM   310  N NE  . ARG A 1 41  ? -8.835  -8.289  -20.521 1.00 41.54 ? 48  ARG A NE  1
            ATOM   311  C CZ  . ARG A 1 41  ? -8.250  -7.379  -19.755 1.00 41.82 ? 48  ARG A CZ  1
            ATOM   312  N NH1 . ARG A 1 41  ? -7.305  -7.750  -18.899 1.00 36.04 ? 48  ARG A NH1 1
            ATOM   313  N NH2 . ARG A 1 41  ? -8.603  -6.103  -19.854 1.00 36.61 ? 48  ARG A NH2 1
            ATOM   314  N N   . GLU A 1 42  ? -13.185 -9.074  -23.754 1.00 23.51 ? 49  GLU A N   1
            ATOM   315  C CA  A GLU A 1 42  ? -14.547 -8.571  -23.581 0.51 23.21 ? 49  GLU A CA  1
            ATOM   316  C CA  B GLU A 1 42  ? -14.546 -8.581  -23.571 0.49 23.22 ? 49  GLU A CA  1
            ATOM   317  C C   . GLU A 1 42  ? -15.576 -9.512  -24.197 1.00 20.35 ? 49  GLU A C   1
            ATOM   318  O O   . GLU A 1 42  ? -16.669 -9.686  -23.647 1.00 23.01 ? 49  GLU A O   1
            ATOM   319  C CB  A GLU A 1 42  ? -14.680 -7.152  -24.154 0.51 23.25 ? 49  GLU A CB  1
            ATOM   320  C CB  B GLU A 1 42  ? -14.688 -7.167  -24.134 0.49 23.25 ? 49  GLU A CB  1
            ATOM   321  C CG  A GLU A 1 42  ? -14.542 -6.975  -25.665 0.51 23.82 ? 49  GLU A CG  1
            ATOM   322  C CG  B GLU A 1 42  ? -13.776 -6.133  -23.469 0.49 22.16 ? 49  GLU A CG  1
            ATOM   323  C CD  A GLU A 1 42  ? -14.833 -5.539  -26.111 0.51 24.65 ? 49  GLU A CD  1
            ATOM   324  C CD  B GLU A 1 42  ? -14.085 -5.861  -21.990 0.49 25.01 ? 49  GLU A CD  1
            ATOM   325  O OE1 A GLU A 1 42  ? -15.114 -4.697  -25.235 0.51 29.69 ? 49  GLU A OE1 1
            ATOM   326  O OE1 B GLU A 1 42  ? -15.060 -6.414  -21.417 0.49 22.79 ? 49  GLU A OE1 1
            ATOM   327  O OE2 A GLU A 1 42  ? -14.783 -5.240  -27.329 0.51 18.21 ? 49  GLU A OE2 1
            ATOM   328  O OE2 B GLU A 1 42  ? -13.335 -5.064  -21.393 0.49 23.67 ? 49  GLU A OE2 1
            ATOM   329  N N   . SER A 1 43  ? -15.252 -10.131 -25.327 1.00 21.22 ? 50  SER A N   1
            ATOM   330  C CA  . SER A 1 43  ? -16.169 -11.092 -25.920 1.00 21.09 ? 50  SER A CA  1
            ATOM   331  C C   . SER A 1 43  ? -15.986 -12.496 -25.360 1.00 23.05 ? 50  SER A C   1
            ATOM   332  O O   . SER A 1 43  ? -16.933 -13.299 -25.389 1.00 22.06 ? 50  SER A O   1
            ATOM   333  C CB  . SER A 1 43  ? -15.983 -11.112 -27.437 1.00 19.72 ? 50  SER A CB  1
            ATOM   334  O OG  . SER A 1 43  ? -14.711 -11.659 -27.753 1.00 21.06 ? 50  SER A OG  1
            ATOM   3145 N N   . GLY A 1 57  ? 26.651  4.429   3.669   1.00 18.24 ? 63  GLY A N   1
            ATOM   3146 C CA  . GLY A 1 57  ? 26.090  5.576   2.975   1.00 18.15 ? 63  GLY A CA  1
            ATOM   3147 C C   . GLY A 1 57  ? 25.791  5.259   1.523   1.00 17.43 ? 63  GLY A C   1
            ATOM   3148 O O   . GLY A 1 57  ? 25.232  4.208   1.196   1.00 18.91 ? 63  GLY A O   1
            HETATM 3149 N N   A SEP A 1 58  ? 26.196  6.180   0.656   0.60 19.35 ? 64  SEP A N   1
            HETATM 3150 C CA  A SEP A 1 58  ? 25.897  6.120   -0.774  0.60 18.88 ? 64  SEP A CA  1
            HETATM 3151 C CB  A SEP A 1 58  ? 26.384  7.385   -1.481  0.60 19.93 ? 64  SEP A CB  1
            HETATM 3152 O OG  A SEP A 1 58  ? 25.458  8.445   -1.335  0.60 23.31 ? 64  SEP A OG  1
            HETATM 3153 C C   A SEP A 1 58  ? 26.438  4.899   -1.513  0.60 17.48 ? 64  SEP A C   1
            HETATM 3154 O O   A SEP A 1 58  ? 26.004  4.622   -2.636  0.60 17.93 ? 64  SEP A O   1
            HETATM 3155 P P   A SEP A 1 58  ? 26.111  9.615   -0.444  0.60 22.18 ? 64  SEP A P   1
            HETATM 3156 O O1P A SEP A 1 58  ? 27.046  10.520  -1.389  0.60 18.80 ? 64  SEP A O1P 1
            HETATM 3157 O O2P A SEP A 1 58  ? 24.902  10.458  0.195   0.60 22.19 ? 64  SEP A O2P 1
            HETATM 3158 O O3P A SEP A 1 58  ? 27.023  8.969   0.701   0.60 19.31 ? 64  SEP A O3P 1
            ATOM   3159 N N   B SER A 1 58  ? 26.205  6.158   0.638   0.40 19.37 ? 64  SER A N   1
            ATOM   3160 C CA  B SER A 1 58  ? 25.820  6.081   -0.766  0.40 18.85 ? 64  SER A CA  1
            ATOM   3161 C C   B SER A 1 58  ? 26.429  4.900   -1.510  0.40 17.49 ? 64  SER A C   1
            ATOM   3162 O O   B SER A 1 58  ? 26.026  4.646   -2.652  0.40 17.92 ? 64  SER A O   1
            ATOM   3163 C CB  B SER A 1 58  ? 26.200  7.375   -1.473  0.40 19.91 ? 64  SER A CB  1
            ATOM   3164 O OG  B SER A 1 58  ? 25.417  8.448   -0.980  0.40 23.21 ? 64  SER A OG  1
            ATOM   3165 N N   . VAL A 1 59  ? 27.372  4.168   -0.910  1.00 17.62 ? 65  VAL A N   1
            ATOM   3166 C CA  . VAL A 1 59  ? 27.783  2.897   -1.496  1.00 17.62 ? 65  VAL A CA  1
            ATOM   3167 C C   . VAL A 1 59  ? 26.577  1.973   -1.603  1.00 16.68 ? 65  VAL A C   1
            ATOM   3168 O O   . VAL A 1 59  ? 26.536  1.092   -2.472  1.00 17.48 ? 65  VAL A O   1
            ATOM   3169 C CB  . VAL A 1 59  ? 28.932  2.257   -0.687  1.00 17.68 ? 65  VAL A CB  1
            ATOM   3170 C CG1 . VAL A 1 59  ? 29.307  0.860   -1.253  1.00 18.45 ? 65  VAL A CG1 1
            ATOM   3171 C CG2 . VAL A 1 59  ? 30.140  3.183   -0.684  1.00 17.83 ? 65  VAL A CG2 1
            """)
    return cif_content


def test_process_altloc_groups_whole():
    """Test the _process_altloc_groups_whole function."""
    cif_content = create_test_resolve_mmcif_altlocs()

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)

        struc = gemmi.read_structure(str(cif_path))
        model = struc[0]
        chain = model[0]

        for group in chain.whole().residue_groups():
            if len(group) > 1:
                _process_altloc_groups_whole(group)

        out_content = struc.make_mmcif_block().as_string()
        assert (
            "ATOM 37 N N . SER A 1 58 ? 26.205 6.158 0.638 0.6 19.37 ? 64 A 1"
            in out_content
        )


def test_process_altloc_groups_whole2():
    """Test the _process_altloc_groups_whole function."""
    cif_content2 = create_test_resolve_mmcif_altlocs2()

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content2)
        struc = gemmi.read_structure(str(cif_path))
        model = struc[0]
        chain = model[0]

        for group in chain.whole().residue_groups():
            if len(group) > 1:
                _process_altloc_groups_whole(group)

        out_content = struc.make_mmcif_block().as_string()
        assert (
            "HETATM 37 N N . SEP A 1 58 ? 26.196 6.18 0.656 0.6 19.35 ? 64 A 1"
            in out_content
        )


def test_resolve_mmcif_altlocs():
    """Test the resolve_mmcif_altlocs function."""
    cif_content = create_test_resolve_mmcif_altlocs()

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)
        struc = gemmi.read_structure(str(cif_path))

        model = struc[0]

        for chain in model:
            resolve_mmcif_altlocs(chain)

        out_content = struc.make_mmcif_block().as_string()
        assert (
            "ATOM 13 C CA . GLU A 1 42 ? -14.547 -8.571 -23.581 0.51 23.21 ? 49 A 1"
            in out_content
        )
        assert (
            "ATOM 31 N N . SER A 1 58 ? 26.205 6.158 0.638 0.6 19.37 ? 64 A 1"
            in out_content
        )


def test_resolve_mmcif_altlocs2():
    """Test the resolve_mmcif_altlocs function."""
    cif_content2 = create_test_resolve_mmcif_altlocs2()

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content2)
        struc = gemmi.read_structure(str(cif_path))

        model = struc[0]

        for chain in model:
            resolve_mmcif_altlocs(chain)

        out_content = struc.make_mmcif_block().as_string()
        assert (
            "ATOM 13 C CA . GLU A 1 42 ? -14.547 -8.571 -23.581 0.51 23.21 ? 49 A 1"
            in out_content
        )
        assert (
            "HETATM 31 N N . SEP A 1 58 ? 26.196 6.18 0.656 0.6 19.35 ? 64 A 1"
            in out_content
        )


def test_deuterium_to_hydrogen_conversion():
    """Test whether deuterium atoms are correctly converted to hydrogen atoms."""
    cif_content = textwrap.dedent("""\
            data_4ZZ4
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            ATOM   106  N N    . PHE A 1 8   ? -5.100  5.198   7.667  1.00 12.50 ? 8   PHE A N    1
            ATOM   107  C CA   . PHE A 1 8   ? -4.133  4.926   8.735  1.00 11.87 ? 8   PHE A CA   1
            ATOM   108  C C    . PHE A 1 8   ? -4.807  4.922   10.112 1.00 12.05 ? 8   PHE A C    1
            ATOM   109  O O    . PHE A 1 8   ? -4.552  4.059   10.931 1.00 12.12 ? 8   PHE A O    1
            ATOM   110  C CB   . PHE A 1 8   ? -3.008  5.952   8.701  1.00 12.72 ? 8   PHE A CB   1
            ATOM   111  C CG   . PHE A 1 8   ? -2.010  5.775   9.807  1.00 10.90 ? 8   PHE A CG   1
            ATOM   112  C CD1  . PHE A 1 8   ? -2.205  6.362   11.055 1.00 12.39 ? 8   PHE A CD1  1
            ATOM   113  C CD2  . PHE A 1 8   ? -0.861  5.025   9.595  1.00 13.55 ? 8   PHE A CD2  1
            ATOM   114  C CE1  . PHE A 1 8   ? -1.307  6.179   12.065 1.00 12.92 ? 8   PHE A CE1  1
            ATOM   115  C CE2  . PHE A 1 8   ? 0.058   4.856   10.617 1.00 13.82 ? 8   PHE A CE2  1
            ATOM   116  C CZ   . PHE A 1 8   ? -0.177  5.413   11.856 1.00 14.44 ? 8   PHE A CZ   1
            ATOM   117  D D    A PHE A 1 8   ? -4.939  5.900   7.198  1.00 15.00 ? 8   PHE A D    1
            ATOM   118  H H    B PHE A 1 8   ? -4.939  5.900   7.198  0.00 15.00 ? 8   PHE A H    1
            ATOM   119  H HA   . PHE A 1 8   ? -3.743  4.050   8.591  1.00 14.24 ? 8   PHE A HA   1
            ATOM   120  H HB2  . PHE A 1 8   ? -2.536  5.873   7.857  1.00 15.27 ? 8   PHE A HB2  1
            ATOM   121  H HB3  . PHE A 1 8   ? -3.390  6.840   8.783  1.00 15.27 ? 8   PHE A HB3  1
            ATOM   122  H HD1  . PHE A 1 8   ? -2.976  6.859   11.211 1.00 14.86 ? 8   PHE A HD1  1
            ATOM   123  H HD2  . PHE A 1 8   ? -0.712  4.629   8.767  1.00 16.26 ? 8   PHE A HD2  1
            ATOM   124  H HE1  . PHE A 1 8   ? -1.456  6.570   12.896 1.00 15.50 ? 8   PHE A HE1  1
            ATOM   125  H HE2  . PHE A 1 8   ? 0.821   4.343   10.475 1.00 16.58 ? 8   PHE A HE2  1
            ATOM   126  H HZ   . PHE A 1 8   ? 0.446   5.310   12.538 1.00 17.33 ? 8   PHE A HZ   1
            ATOM   127  N N    . GLU A 1 9   ? -5.674  5.895   10.377 1.00 11.74 ? 9   GLU A N    1
            ATOM   128  C CA   . GLU A 1 9   ? -6.399  5.924   11.629 1.00 11.11 ? 9   GLU A CA   1
            ATOM   129  C C    . GLU A 1 9   ? -7.231  4.644   11.781 1.00 11.97 ? 9   GLU A C    1
            ATOM   130  O O    . GLU A 1 9   ? -7.241  4.024   12.832 1.00 11.97 ? 9   GLU A O    1
            ATOM   131  C CB   . GLU A 1 9   ? -7.301  7.160   11.737 1.00 13.32 ? 9   GLU A CB   1
            ATOM   132  C CG   A GLU A 1 9   ? -6.539  8.451   11.706 0.49 11.75 ? 9   GLU A CG   1
            ATOM   133  C CG   B GLU A 1 9   ? -7.856  7.382   13.132 0.51 16.75 ? 9   GLU A CG   1
            ATOM   134  C CD   A GLU A 1 9   ? -7.390  9.670   11.954 0.49 18.06 ? 9   GLU A CD   1
            ATOM   135  C CD   B GLU A 1 9   ? -8.856  8.536   13.226 0.51 17.32 ? 9   GLU A CD   1
            ATOM   136  O OE1  A GLU A 1 9   ? -6.980  10.744  11.476 0.49 19.40 ? 9   GLU A OE1  1
            ATOM   137  O OE1  B GLU A 1 9   ? -9.065  9.251   12.227 0.51 26.95 ? 9   GLU A OE1  1
            ATOM   138  O OE2  A GLU A 1 9   ? -8.418  9.556   12.658 0.49 20.45 ? 9   GLU A OE2  1
            ATOM   139  O OE2  B GLU A 1 9   ? -9.423  8.710   14.319 0.51 26.44 ? 9   GLU A OE2  1
            ATOM   140  D D    A GLU A 1 9   ? -5.856  6.547   9.846  0.77 14.08 ? 9   GLU A D    1
            ATOM   141  H H    B GLU A 1 9   ? -5.856  6.547   9.846  0.23 14.08 ? 9   GLU A H    1
            ATOM   142  H HA   . GLU A 1 9   ? -5.762  5.955   12.360 1.00 13.34 ? 9   GLU A HA   1
            ATOM   143  H HB2  A GLU A 1 9   ? -7.923  7.162   10.993 0.49 15.99 ? 9   GLU A HB2  1
            ATOM   144  H HB2  B GLU A 1 9   ? -6.788  7.946   11.492 0.51 15.99 ? 9   GLU A HB2  1
            ATOM   145  H HB3  A GLU A 1 9   ? -7.788  7.121   12.576 0.49 15.99 ? 9   GLU A HB3  1
            ATOM   146  H HB3  B GLU A 1 9   ? -8.052  7.054   11.132 0.51 15.99 ? 9   GLU A HB3  1
            ATOM   147  H HG2  A GLU A 1 9   ? -5.852  8.425   12.390 0.49 14.10 ? 9   GLU A HG2  1
            ATOM   148  H HG2  B GLU A 1 9   ? -8.308  6.574   13.421 0.51 20.11 ? 9   GLU A HG2  1
            ATOM   149  H HG3  A GLU A 1 9   ? -6.129  8.549   10.832 0.49 14.10 ? 9   GLU A HG3  1
            ATOM   150  H HG3  B GLU A 1 9   ? -7.120  7.579   13.733 0.51 20.11 ? 9   GLU A HG3  1
            """)

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)
        struc = gemmi.read_structure(str(cif_path))
        fix_structure(struc)
        out_content = struc.make_mmcif_block().as_string()
        assert (
            "ATOM 26 C CG . GLU A 1 9 ? -7.856 7.382 13.132 0.51 16.75 ? 9 A 1 0"
            in out_content
        )
        assert (
            "ATOM 12 H H . PHE A 1 8 ? -4.939 5.9 7.198 1 15 ? 8 A 1 1" in out_content
        )
        assert (
            "ATOM 12 D D . PHE A 1 8 ? -4.939 5.9 7.198 1 15 ? 8 A 1 1"
            not in out_content
        )


def test_mse_to_met_conversion():
    """Test whether MSE residues are correctly converted to MET residues."""
    cif_content = textwrap.dedent("""\
            data_3NY5
            #
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.auth_seq_id
            _atom_site.auth_comp_id
            _atom_site.auth_asym_id
            _atom_site.auth_atom_id
            _atom_site.pdbx_PDB_model_num
            ATOM   1    N  N   . HIS A 1 10 ? 5.470   60.870 54.783  1.00 53.22  ? 151 HIS A N   1
            ATOM   2    C  CA  . HIS A 1 10 ? 5.803   62.269 54.553  1.00 44.30  ? 151 HIS A CA  1
            ATOM   3    C  C   . HIS A 1 10 ? 6.477   62.930 55.779  1.00 39.46  ? 151 HIS A C   1
            ATOM   4    O  O   . HIS A 1 10 ? 6.487   62.404 56.899  1.00 36.12  ? 151 HIS A O   1
            ATOM   5    C  CB  . HIS A 1 10 ? 4.567   63.056 54.091  1.00 39.07  ? 151 HIS A CB  1
            HETATM 6    N  N   . MSE A 1 11 ? 7.064   64.087 55.543  1.00 38.40  ? 152 MSE A N   1
            HETATM 7    C  CA  . MSE A 1 11 ? 7.794   64.795 56.593  1.00 43.88  ? 152 MSE A CA  1
            HETATM 8    C  C   . MSE A 1 11 ? 6.910   65.022 57.831  1.00 46.96  ? 152 MSE A C   1
            HETATM 9    O  O   . MSE A 1 11 ? 7.385   65.010 58.978  1.00 46.66  ? 152 MSE A O   1
            HETATM 10   C  CB  . MSE A 1 11 ? 8.339   66.133 56.065  1.00 35.30  ? 152 MSE A CB  1
            HETATM 11   C  CG  . MSE A 1 11 ? 9.161   66.895 57.099  1.00 41.45  ? 152 MSE A CG  1
            HETATM 12   SE SE  . MSE A 1 11 ? 10.649  65.796 57.774  1.00 85.51  ? 152 MSE A SE  1
            HETATM 13   C  CE  . MSE A 1 11 ? 12.106  66.518 56.649  1.00 49.90  ? 152 MSE A CE  1
            """)

    with TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        cif_path.write_text(cif_content)
        struc = gemmi.read_structure(str(cif_path))
        fix_structure(struc)
        out_content = struc.make_mmcif_block().as_string()
        assert "N N . MET A 1 11 ? 7.064 64.087 55.543" in out_content
        assert "N N . MSE A 1 11 ? 7.064   64.087 55.543" not in out_content
        assert "C CA . MET A 1 11 ? 7.794 64.795 56.593" in out_content
