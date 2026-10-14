"""Dataset and ground-truth engine (Phase 2).

Submodules (import them directly; this package intentionally imports nothing eagerly so that
``flawbench.config`` can depend on :mod:`flawbench.dataset.params` without import cycles):

* ``params``     flaw identifiers, parameter specs, bounds and profile validation (pure)
* ``schema``     source and sample record schemas, timeline convention
* ``io``         checksums, atomic writes, JSON Lines helpers
* ``registry``   reference-recording registry
* ``injectors``  pure audio transformations returning exact annotations
* ``plan``       generation-plan loading and expansion
* ``generator``  transactional variant and batch generation
* ``validator``  dataset auditing against real files
* ``splits``     source-level split assignment
* ``export``     export of redistributable samples
"""
