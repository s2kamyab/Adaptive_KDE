"""Named estimator configurations.

Keeping every configuration in one place means an experiment script names the
methods it wants and nothing else, and that the settings behind a reported
number can be recovered from the results file.

The comparison set follows the recommended prioritisation: the five
"essential" rows plus kernel thinning are in ``ESSENTIAL``; the conventional
EM control and the ablations sit alongside.  Methods deliberately *not*
implemented are listed in :data:`NOT_IMPLEMENTED` with the reason, so their
absence from a results table is documented rather than merely unexplained.
"""

from __future__ import annotations

from copy import deepcopy

from .dictionary import DictionaryConfig
from .estimators import (
    AKDE, AKDEConfig, AbramsonConfig, AbramsonKDE, EMConfig, EMGaussianMixture,
    Estimator, FixedKDE, SPADES, SPADESConfig, SPADESSimplex, SongConfig,
    SongKernelMoment, ThinnedKDE, ThinningConfig, TuningConfig, WEIGHTINGS,
)

__all__ = ["build", "ESSENTIAL", "ABLATIONS", "ALL", "NOT_IMPLEMENTED", "shared_dictionary"]


def shared_dictionary(n_loc: int = 150, n_scale: int = 10) -> DictionaryConfig:
    """The dictionary given to every dictionary-based method.

    Using one dictionary across AKDE, SPADES and Song is what makes the
    comparison about the fitting criterion rather than about the
    representation, as the recommendations require.
    """
    return DictionaryConfig(n_loc=n_loc, n_scale=n_scale, location_rule="grid",
                            scale_lo_mult=0.05, scale_hi_mult=4.0)


#: Everything that can be requested by name.
_FACTORIES: dict[str, callable] = {}


def _register(name: str, fn) -> None:
    _FACTORIES[name] = fn


# --- the proposed estimator, and its weighting ablations ------------------

for _w in WEIGHTINGS:
    def _mk(w=_w):
        def f(n_loc=150, n_scale=10, **kw):
            # n_loc / n_scale set the LARGEST dictionary offered; the size
            # itself is chosen by cross-validation (see AKDEConfig.dict_sizes).
            cfg = AKDEConfig(dictionary=shared_dictionary(n_loc, n_scale), weighting=w)
            for k, v in kw.items():
                setattr(cfg, k, v)
            return AKDE(cfg, name=f"AKDE[{w}]")
        return f
    _register(f"akde_{_w}", _mk())

_register("akde", _FACTORIES["akde_ridge"])


def _akde_variant(name: str, **overrides):
    def f(n_loc=150, n_scale=10, **kw):
        cfg = AKDEConfig(dictionary=shared_dictionary(n_loc, n_scale))
        for k, v in {**overrides, **kw}.items():
            setattr(cfg, k, v)
        return AKDE(cfg, name=name)
    return f


# ablations of the estimator's own structure
_register("akde_single_stage", _akde_variant("AKDE[no refit]", single_stage=True))
_register("akde_no_sparsity", _akde_variant("AKDE[no sparsity]", no_sparsity=True))
#: The historical configuration, where measurement kernels are a subsample of
#: the dictionary itself. Kept so the original design stays reproducible.
_register("akde_matched_meas", _akde_variant("AKDE[matched k]", meas_mode="matched",
                                             meas_n_loc=40, meas_n_scale=3))


def _akde_one_scale(n_loc=1500, n_scale=1, **kw):
    """Single-bandwidth dictionary: the 'one bandwidth vs multiple' ablation.

    Location count is raised so the dictionary size matches the multiscale
    default, otherwise the ablation confounds resolution with dictionary size.
    """
    cfg = AKDEConfig(dictionary=DictionaryConfig(n_loc=n_loc, n_scale=1,
                                                 scale_lo_mult=1.0, scale_hi_mult=1.0))
    for k, v in kw.items():
        setattr(cfg, k, v)
    return AKDE(cfg, name="AKDE[single scale]")


_register("akde_single_scale", _akde_one_scale)


def _akde_fixed_dict(nl: int, ns: int, label: str):
    """AKDE at a pinned dictionary size, ignoring the run's --n-loc/--n-scale.

    Dictionary size turned out to matter far more than expected: an
    over-complete dictionary packs near-duplicate atoms whose moment
    signatures are almost identical, which drives the condition number of the
    design matrix towards 1e17 and leaves the moment fit under-determined in
    L2.  On MW10 at N=5000, M=600 gave ISE 1.7e-2 against 2.2e-1 for M=1500.
    These entries exist so that effect is measured across the whole suite
    rather than asserted from a handful of draws.
    """
    def f(n_loc=None, n_scale=None, **kw):
        # dict_sizes=None pins the dictionary, so this measures one size
        # rather than letting cross-validation choose.
        cfg = AKDEConfig(dictionary=shared_dictionary(nl, ns), dict_sizes=None)
        for k, v in kw.items():
            setattr(cfg, k, v)
        return AKDE(cfg, name=label)
    return f


_register("akde_M300", _akde_fixed_dict(60, 5, "AKDE[M=300]"))
_register("akde_M600", _akde_fixed_dict(100, 6, "AKDE[M=600]"))
_register("akde_M1500", _akde_fixed_dict(150, 10, "AKDE[M=1500]"))

# --- baselines ------------------------------------------------------------

_register("spades", lambda n_loc=150, n_scale=10, **kw:
          SPADES(SPADESConfig(dictionary=shared_dictionary(n_loc, n_scale))))
_register("spades_simplex", lambda n_loc=150, n_scale=10, **kw:
          SPADESSimplex(SPADESConfig(dictionary=shared_dictionary(n_loc, n_scale))))
_register("song", lambda n_loc=150, n_scale=10, **kw:
          SongKernelMoment(SongConfig(dictionary=shared_dictionary(n_loc, n_scale))))

_register("kde_sj", lambda **kw: FixedKDE("sj"))
_register("kde_lscv", lambda **kw: FixedKDE("lscv"))
_register("kde_botev", lambda **kw: FixedKDE("botev_isj"))
_register("kde_silverman", lambda **kw: FixedKDE("silverman"))
_register("abramson", lambda **kw: AbramsonKDE(AbramsonConfig()))

_register("gmm_em", lambda **kw: EMGaussianMixture(EMConfig()))
_register("gmm_em_bic", lambda **kw: EMGaussianMixture(EMConfig(selection="bic")))

_register("thin_kt", lambda **kw: ThinnedKDE(ThinningConfig(method="kt")))
_register("thin_herding", lambda **kw: ThinnedKDE(ThinningConfig(method="herding")))


def build(name: str, **kwargs) -> Estimator:
    if name not in _FACTORIES:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**kwargs)


#: The recommended first-round comparison: five essential rows + kernel thinning.
ESSENTIAL = [
    "akde",             # proposed (ridge-regularised full covariance weighting)
    "spades",           # essential: direct predecessor, original signed objective
    "spades_simplex",   # density-valid variant, isolates the loss from the constraint
    "song",             # essential: kernel-embedding mixture fitting
    "kde_sj",           # essential: well-tuned conventional KDE
    "kde_lscv",         # essential: cross-validated bandwidth
    "abramson",         # essential: established local bandwidth adaptation
    "kde_botev",        # essential: improved plug-in (Botev et al.)
    "gmm_em",           # useful conventional control
    "thin_kt",          # high priority recent: kernel thinning + KDE
]

#: Core ablations of the proposed estimator.
ABLATIONS = [
    # weighting matrix: identity / diagonal / ridge-regularised full / truncated pseudo-inverse
    "akde_identity",
    "akde_diagonal",
    "akde_ridge",
    "akde_trunc_pinv",
    # one bandwidth versus several, at matched dictionary size
    "akde_single_scale",
    # sparsity on/off, and refit on/off
    "akde_no_sparsity",
    "akde_single_stage",
    # matched versus independently designed measurement family
    "akde_matched_meas",
    # dictionary size relative to the sample
    "akde_M300",
    "akde_M600",
    "akde_M1500",
]

ALL = sorted(_FACTORIES)


#: Comparators considered and deliberately excluded from this run, with reasons.
#: Stating these keeps their absence from the results an explicit scope
#: decision rather than an oversight.
NOT_IMPLEMENTED = {
    "psd_model": (
        "Rudi & Ciliberto (NeurIPS 2021) PSD probability model. Next priority after "
        "the essential set. Excluded here because a faithful implementation needs "
        "their specific regularised fitting scheme and Nystrom construction; a "
        "hand-rolled approximation would not be a fair comparison, and its model "
        "size (a PSD matrix and its rank) is not interchangeable with mixture "
        "support size, so it needs its own cost axis."
    ),
    "low_rank_thinning": (
        "Carrell et al. (ICML 2025). The proceedings software link points to "
        "Thinformer, which targets attention approximation rather than density "
        "estimation, so the exact thinning routine for a KDE experiment is not "
        "settled. Kernel thinning is included instead as the compression baseline."
    ),
    "dgmm": (
        "Zhang et al. (2025 preprint) diagonally-weighted GMM. Code is promised on "
        "publication and not currently available. Its singular components may lack "
        "a full-dimensional Lebesgue density, so it needs a compatible metric and a "
        "separate experiment rather than a row in the ISE table. NOTE: the "
        "'akde_diagonal' ablation here is an ablation of THIS estimator's weighting "
        "matrix and must not be labelled as Zhang et al.'s method."
    ),
    "neural_spline_flow": (
        "Durkan et al. (NeurIPS 2019). Most informative once the evaluation extends "
        "well beyond small-sample univariate data; a univariate version needs a "
        "scalar monotone spline rather than a coupling architecture applied "
        "mechanically. Out of scope for this round."
    ),
}
