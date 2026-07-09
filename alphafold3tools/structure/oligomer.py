"""Guess the homo-oligomeric state of a protein from its mmCIF structure.

Given a PDB structure and an author chain ID, this module inspects the
biological assembly definition (``_pdbx_struct_assembly_gen`` /
``_pdbx_struct_oper_list`` / ``_struct_asym`` / ``_entity_poly``) and counts how
many copies of the *same polymer entity* as the query chain are present in the
biological assembly that contains that chain. That copy count is the guessed
homo-oligomer number (e.g. 6 for a homohexamer such as PDB ``1BJP``).

Parsing is done with gemmi, which expands the ``oper_expression`` product
notation (e.g. ``(1-60)(61-88)``) into a flat list of operators, so no manual
expression parsing is required.
"""

import gemmi
from loguru import logger

import alphafold3tools.structure_stores as structure_stores


def _load_structure(mmcif_string: str) -> gemmi.Structure:
    """Parses an mmCIF string into a gemmi Structure with entities set up."""
    block = gemmi.cif.read_string(mmcif_string).sole_block()
    structure = gemmi.make_structure_from_block(block)
    # Ensure per-residue subchain assignment and entity subchain lists exist even
    # when the source mmCIF omits some of that bookkeeping.
    structure.setup_entities()
    return structure


def guess_homomer_count(mmcif_string: str, auth_chain_id: str) -> int:
    """Guesses the homo-oligomer count for ``auth_chain_id``.

    Args:
        mmcif_string: Full mmCIF content of the structure.
        auth_chain_id: Author chain ID (``auth_asym_id``) of the query chain.

    Returns:
        The number of copies of the query chain's polymer entity within the
        biological assembly that contains the query chain. Returns 1 when no
        biological assembly information is available or the query chain / its
        polymer entity cannot be located.
    """
    try:
        structure = _load_structure(mmcif_string)
    except Exception as e:  # noqa: BLE001 - defensive: malformed mmCIF -> monomer.
        logger.warning(f"Failed to parse mmCIF for oligomer guessing: {e}")
        return 1

    # Map polymer subchains (label_asym_id) to their entity, and each polymer
    # entity to its set of subchains.
    polymer_subchain_to_entity: dict[str, str] = {}
    entity_to_subchains: dict[str, set[str]] = {}
    for entity in structure.entities:
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        entity_to_subchains[entity.name] = set(entity.subchains)
        for subchain in entity.subchains:
            polymer_subchain_to_entity[subchain] = entity.name

    if not structure:
        return 1
    model = structure[0]

    query_chain = next((c for c in model if c.name == auth_chain_id), None)
    if query_chain is None:
        logger.warning(
            f"Chain {auth_chain_id!r} not found in structure; assuming monomer."
        )
        return 1

    # The query auth chain may contain several subchains (polymer + ligands +
    # water); pick the polymer subchain to identify the query entity.
    query_subchain = next(
        (
            res.subchain
            for res in query_chain
            if res.subchain in polymer_subchain_to_entity
        ),
        None,
    )
    if query_subchain is None:
        logger.warning(
            f"Chain {auth_chain_id!r} has no polymer subchain; assuming monomer."
        )
        return 1

    query_entity = polymer_subchain_to_entity[query_subchain]
    same_entity_subchains = entity_to_subchains[query_entity]

    # Find the biological assembly containing the query subchain and count the
    # number of same-entity polymer copies it generates.
    best_count = 1
    for assembly in structure.assemblies:
        generators = assembly.generators
        if not any(query_subchain in set(g.subchains) for g in generators):
            continue
        count = 0
        for gen in generators:
            num_operators = len(gen.operators)
            num_same_entity = sum(
                1 for s in gen.subchains if s in same_entity_subchains
            )
            count += num_operators * num_same_entity
        best_count = max(best_count, count)

    return best_count


def guess_homomer_count_from_store(
    store: structure_stores.StructureStore,
    pdb_id: str,
    auth_chain_id: str,
) -> int:
    """Reads the mmCIF for ``pdb_id`` from ``store`` and guesses its oligomer count.

    Args:
        store: StructureStore pointing at the mmCIF database.
        pdb_id: PDB ID of the structure (as used by the store; typically lower
            case, e.g. ``"1bjp"``).
        auth_chain_id: Author chain ID of the query chain.

    Returns:
        The guessed homo-oligomer count. Returns 1 when the structure cannot be
        retrieved or parsed.
    """
    try:
        mmcif_string = store.get_mmcif_str(pdb_id)
    except Exception as e:  # noqa: BLE001 - defensive: missing file -> monomer.
        logger.warning(
            f"Could not retrieve mmCIF for {pdb_id!r} while guessing copies: {e}"
        )
        return 1
    return guess_homomer_count(mmcif_string, auth_chain_id)
