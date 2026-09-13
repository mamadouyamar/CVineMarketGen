# -*- coding: utf-8 -*-
"""
C-vine copula financial market generator: family selection (Algorithm 3),
correlation-targeting calibration (Algorithm 4), scenario generation
(Algorithm 5). Relocated from momentmatchingscript.py; load_data_and_setup
takes DataFrames instead of Excel paths.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
import pyvinecopulib as pv

from .moment_match import MomentMatch
from .copulas import CopulaTools


class _EarlyStopSLSQP(Exception):
    pass


class CVineGenerator(MomentMatch, CopulaTools):
    """C-vine generator (article 3, Sections 4.5 and 5)."""
    _EarlyStopSLSQP = _EarlyStopSLSQP
    _FAMILY = {'gaussian': pv.BicopFamily.gaussian, 'clayton': pv.BicopFamily.clayton,
               'gumbel': pv.BicopFamily.gumbel, 'joe': pv.BicopFamily.joe, 'frank': pv.BicopFamily.frank}


    def estimate_CVine_preselected_V2(
            self,
            input_data,
            target_corr,
            use_mixture_on_firsttree=True,
            use_ncs_on_firsttree=True,
            use_mixture_on_deepertrees=False,
            use_ncs_on_deepertrees=False,
            force_try_ncscopula=False):

        ## ==================================================================================
        ## INITIALIZATION
        ## Convert historical returns to pseudo-observations (empirical CDF rank transform)
        ## so that all dependence information is captured by the copula, free of marginals.
        ## ==================================================================================
        data = pd.DataFrame(pv.to_pseudo_obs(input_data),
                            columns=[str(val + 1) for val in range(input_data.shape[1])])

        n_tree = data.shape[1] - 1

        trees_copulas = {}  ## stores the fitted copula object for each edge
        trees_hfunc = {}  ## stores the h-function output (conditional pseudo-obs) at each tree level
        trees_prespecification = {}  ## stores the tail dependence classification for each edge

        d = data.shape[1]

        ## ==================================================================================
        ## Allocate (N-1) x (N-1) parameter matrices.
        ## Position [k-1, k-1+j] stores the copula for edge j in tree k.
        ## Column c of these matrices contains all copulas needed to construct variable c+1,
        ## which is how run_vine_optimization reads them column-by-column.
        ## ==================================================================================
        thetas_final_1p = np.zeros([d - 1, d - 1],
                                   dtype=object) * np.nan  ## primary parameter (or full param vector for NCS/mixture)
        thetas_final_2p = np.zeros([d - 1, d - 1]) * np.nan  ## second parameter (e.g. df for Student-t)
        a1_final = np.zeros([d - 1, d - 1]) * np.nan  ## NCS parameter a1
        a2_final = np.zeros([d - 1, d - 1]) * np.nan  ## NCS parameter a2
        fams_cops = np.zeros([d - 1, d - 1],
                             dtype=object) * np.nan  ## copula family (BicopFamily or list of Bicop for mixture)
        fams_ncscops_status = np.zeros([d - 1, d - 1], dtype=object) * np.nan  ## True if NCS copula was selected
        fams_mixture_status = np.zeros([d - 1, d - 1], dtype=object) * np.nan  ## True if mixture copula was selected
        fams_status = np.zeros([d - 1, d - 1], dtype=object) * np.nan  ## 'ordinary', 'ncscopula', or 'mixture'
        fams_rots = np.zeros([d - 1, d - 1], dtype=int) * np.nan  ## rotation (0, 90, 180, 270)

        ### Copula that have the same tail dependence structure and monotonocity
        ## — tight fallback list for optimization: same (ℓ, u, m)
        testable_copulas = np.zeros([d - 1, d - 1], dtype=object) * np.nan
        ### Copula that have the same tail dependence structure only (i.e., non monotonocity imposed)
        ## — broad fallback list for optimization: same (ℓ, u) only
        testable_copulas_only_tail = np.zeros([d - 1, d - 1], dtype=object) * np.nan
        # --------------------------------------------------------------------
        ## Tree 0: the h-function values are just the pseudo-observations themselves
        trees_hfunc['0'] = data.copy()

        ## Load three lookup tables that catalog copula families by their
        ## tail dependence properties. These are the "rules" for pre-selection.
        copulas_specifications = self.get_copulas_specifications()  ## Clayton, Gumbel, Gaussian with rotations
        copulas_specifications_joe_t = self.get_copulas_specifications_Joe_student()  ## Joe, Student-t (secondary alternatives)
        copulas_mixture_specifications = self.get_copulas_mixture_specifications()  ## mixture copula combinations

        ## ==================================================================================
        ## MAIN LOOP: iterate tree by tree, edge by edge through the C-vine structure.
        ## At each edge, select and fit the bivariate copula that best captures the
        ## dependence between the two h-function pseudo-observations at that position.
        ## ==================================================================================
        for tree in range(1, n_tree + 1):

            trees_copulas[str(tree)] = pd.Series(dtype=object)

            trees_hfunc[str(tree)] = pd.DataFrame()

            ii = tree  ## running index for asset labeling in h-function column names

            trees_prespecification[str(tree)] = pd.DataFrame(index=input_data.columns[tree:],
                                                             columns=['lower_tail_dependence',
                                                                      'upper_tail_dependence',
                                                                      'target_correlation_sign',
                                                                      'monotone_dependence'])

            ## Loop over each edge in the current tree.
            ## The pair under consideration is always (column 0, column i) of the previous tree's h-function output.
            ## Column 0 = central node of the C-vine at this tree level.
            for i in range(1, trees_hfunc[str(tree - 1)].shape[1]):

                ## Initialize all temporary variables for this edge
                t_copulas_est, t_copulas, t_copulas_bic = None, None, np.inf
                t_copulas_ncs_est, t_copulas_ncs, t_copulas_ncs_rotation = None, None, None
                t_copulas_ncs_bic, t_copulas_ncs_family, t_copulas_ncs_params = np.inf, None, None
                t_condcorr_metrics, t_lower_tail_dependence, t_upper_tail_dependence = None, None, None
                t_target_correlation_sign, t_monotone_dependence, t_target_specs = None, None, None
                t_mask, t_matching_index, t_tt_monotone = None, None, None
                pos_corr_tt, neg_corr_tt, posneg_corr_tt = None, None, None
                t_target_mix_specs, t_mask_mix, t_matching_mix_index = None, None, None

                t_copula_mix_bic, t_copulas_mixture_est, t_tt_nonmonotone = np.inf, None, None
                i_keep_it = None

                ## ==============================================================================
                ## STEP 1 — EXCEEDANCE CORRELATION AND CLASSIFICATION
                ## Compute the exceedance correlation curve to characterize the qualitative
                ## dependence structure: lower/upper tail dependence, correlation sign, monotonicity.
                ## This determines which copula families are admissible before any statistical fitting.
                ## ==============================================================================
                if tree == 1:
                    ## Tree 1: use raw observations directly (inputs_are_obs=True).
                    ## The pair is (asset 1, asset i+1) from the original return data.
                    t_condcorr = self.draw_cond_corr(
                        x=input_data[input_data.columns[0]] if isinstance(input_data, pd.DataFrame) \
                            else input_data[:, 0],
                        y=input_data[input_data.columns[i]] if isinstance(input_data, pd.DataFrame) \
                            else input_data[:, i],
                        figsize=(6, 6),
                        thetas_lb=-0.5,
                        thetas_ub=0.5,
                        names=['asset1', 'asset2'],
                        return_values=True,
                        inputs_are_obs=True,
                        show_plot=False)

                    ## Extract summary statistics from the exceedance correlation curve
                    t_condcorr_metrics = self.get_condcorrelation_metrics(t_condcorr)

                    ## Classify lower tail dependence:
                    ## True if the left side of the curve is above the midpoint reference
                    ## AND the right side is entirely below it
                    t_lower_tail_dependence = t_condcorr_metrics['mean_leftside'] > t_condcorr_metrics[
                        'halfway_from_jump'] and t_condcorr_metrics['max_rightside'] < t_condcorr_metrics[
                                                  'halfway_from_jump']
                    ## Classify upper tail dependence: symmetric condition
                    t_upper_tail_dependence = t_condcorr_metrics['mean_rightside'] > t_condcorr_metrics[
                        'halfway_from_jump'] and t_condcorr_metrics['max_leftside'] < t_condcorr_metrics[
                                                  'halfway_from_jump']
                    ## Sign of the LTCMA target correlation for this pair
                    t_target_correlation_sign = np.sign(target_corr.iloc[0, i])
                    ## Monotonicity: +1 if the curve does not change sign, -1 if it does
                    t_monotone_dependence = np.sign(t_condcorr.min() * t_condcorr.max())

                    ## Tiebreaker rules when both ℓ and u are False
                    if t_lower_tail_dependence == False and t_upper_tail_dependence == False:
                        ## Tiebreaker 1: compare ranges — non-overlapping ranges give a clear answer
                        if t_condcorr_metrics['max_leftside'] < t_condcorr_metrics['min_rightside']:
                            t_upper_tail_dependence = True
                        elif t_condcorr_metrics['min_leftside'] > t_condcorr_metrics['max_rightside']:
                            t_lower_tail_dependence = True
                        else:
                            ## Tiebreaker 2: compare slopes near zero
                            ## If the curve is rising from left-of-zero to right-of-zero → upper tail
                            if t_condcorr_metrics['mean_leftside'] < t_condcorr[t_condcorr.index < 0].iloc[-1] and \
                                    t_condcorr_metrics['mean_rightside'] > t_condcorr[t_condcorr.index > 0].iloc[0]:
                                t_upper_tail_dependence = True
                            ## If the curve is falling from left-of-zero to right-of-zero → lower tail
                            elif t_condcorr_metrics['mean_leftside'] > t_condcorr[t_condcorr.index < 0].iloc[-1] and \
                                    t_condcorr_metrics['mean_rightside'] < t_condcorr[t_condcorr.index > 0].iloc[0]:
                                t_lower_tail_dependence = True
                            else:
                                ## Unresolvable: flag as -11, which defaults to Gaussian
                                t_target_correlation_sign = -11

                    ## ==============================================================================
                    ## STEP 2 — CANDIDATE PRE-SELECTION AND FITTING (TREE 1)
                    ## Query the lookup table for copula families matching (ℓ, u, s, m).
                    ## If matches exist → fit the best among them by BIC.
                    ## If no matches (non-monotone pair) → try mixture and/or NCS copulas.
                    ## ==============================================================================
                    t_target_specs = [t_lower_tail_dependence,
                                      t_upper_tail_dependence,
                                      t_target_correlation_sign,
                                      t_monotone_dependence]

                    ## Query the ordinary copula lookup table for exact (ℓ, u, s, m) match
                    t_mask = copulas_specifications.eq(t_target_specs).all(axis=1)
                    t_matching_index = list(copulas_specifications.index[t_mask])

                    if (isinstance(t_matching_index, list) and len(t_matching_index) >= 1):
                        ## We found preselected copulas — fit the best among them by BIC
                        t_copulas_est = self.select_best_preselected_bivariate_copula(
                            trees_hfunc[str(tree - 1)].iloc[:, [0, i]],
                            families_and_rotations=t_matching_index)
                        t_copulas = t_copulas_est['copula']
                        t_copulas_bic = t_copulas_est['best_bic']

                    else:
                        ## No match in the ordinary catalog.
                        ## This happens when m = -1 (non-monotone conditional correlation).
                        ## Try mixture and/or NCS copulas depending on user flags.
                        if t_monotone_dependence == -1 and use_mixture_on_firsttree == True:
                            ## Filter the mixture catalog by (ℓ, u)
                            t_target_mix_specs = [t_lower_tail_dependence,
                                                  t_upper_tail_dependence]
                            t_mask_mix = copulas_mixture_specifications.eq(t_target_mix_specs).all(axis=1)
                            t_matching_mix_index = list(copulas_mixture_specifications.index[t_mask_mix])
                            ## If no mixture matches the tail structure, try all mixtures
                            if len(t_matching_mix_index) == 0:
                                t_matching_mix_index = list(copulas_mixture_specifications.index)
                            t_copulas_mixture_est = self.select_best_preselected_mixture_copula(
                                data=trees_hfunc[str(tree - 1)].iloc[:, [0, i]],
                                list_of_copulas_families=t_matching_mix_index)
                            t_copula_mix_bic = t_copulas_mixture_est['best_bic']

                        if t_monotone_dependence == -1 and use_ncs_on_firsttree == True:
                            ## Fit best NCS copula (unrestricted family search)
                            t_copulas_ncs_est = self.select_best_bivariate_ncscopula(
                                trees_hfunc[str(tree - 1)].iloc[:, [0, i]])
                            t_copulas_ncs = t_copulas_ncs_est['copula']
                            t_copulas_ncs_rotation = t_copulas_ncs_est['best_rotation']
                            t_copulas_ncs_bic = t_copulas_ncs_est['best_bic']
                            t_copulas_ncs_family = t_copulas_ncs_est['best_family']
                            t_copulas_ncs_params = t_copulas_ncs_est['copula']['params']

                        ## If neither mixture nor NCS was tried (flags both off, but still no ordinary match),
                        ## fall back to unrestricted MLE over all ordinary families
                        if not (t_monotone_dependence == -1 and use_mixture_on_firsttree == True) and \
                                not (t_monotone_dependence == -1 and use_ncs_on_firsttree == True):
                            t_copulas_est = self.select_best_bivariate_copula(
                                trees_hfunc[str(tree - 1)].iloc[:, [0, i]])
                            t_copulas = t_copulas_est['copula']
                            t_copulas_bic = t_copulas_est['best_bic']

                    ## Override: if force_try_ncscopula is True, always fit NCS regardless of classification
                    if force_try_ncscopula:
                        t_copulas_ncs_est = self.select_best_bivariate_ncscopula(
                            trees_hfunc[str(tree - 1)].iloc[:, [0, i]])
                        t_copulas_ncs = t_copulas_ncs_est['copula']
                        t_copulas_ncs_rotation = t_copulas_ncs_est['best_rotation']
                        t_copulas_ncs_bic = t_copulas_ncs_est['best_bic']
                        t_copulas_ncs_family = t_copulas_ncs_est['best_family']
                        t_copulas_ncs_params = t_copulas_ncs_est['copula']['params']


                elif tree >= 2:
                    ## ==============================================================================
                    ## STEP 1–2 REPEATED FOR DEEPER TREES (k >= 2)
                    ## Same logic as tree 1, but:
                    ##   - inputs_are_obs=False → draw_cond_corr applies Φ⁻¹ to h-function values
                    ##   - the pair is (column 0, column i) of trees_hfunc[str(tree-1)]
                    ##   - flags use_ncs_on_deepertrees and use_mixture_on_deepertrees apply instead
                    ## ==============================================================================

                    t_condcorr = self.draw_cond_corr(
                        x=trees_hfunc[str(tree - 1)].iloc[:, 0],
                        y=trees_hfunc[str(tree - 1)].iloc[:, i],
                        figsize=(6, 6),
                        thetas_lb=-0.5,
                        thetas_ub=0.5,
                        names=['asset1', 'asset2'],
                        return_values=True,
                        inputs_are_obs=False,
                        show_plot=False)

                    t_condcorr_metrics = self.get_condcorrelation_metrics(t_condcorr)
                    t_lower_tail_dependence = t_condcorr_metrics['mean_leftside'] > t_condcorr_metrics[
                        'halfway_from_jump'] and t_condcorr_metrics['max_rightside'] < t_condcorr_metrics[
                                                  'halfway_from_jump']
                    t_upper_tail_dependence = t_condcorr_metrics['mean_rightside'] > t_condcorr_metrics[
                        'halfway_from_jump'] and t_condcorr_metrics['max_leftside'] < t_condcorr_metrics[
                                                  'halfway_from_jump']
                    t_target_correlation_sign = np.sign(target_corr.iloc[0, i])
                    t_monotone_dependence = np.sign(t_condcorr.min() * t_condcorr.max())
                    if t_lower_tail_dependence == False and t_upper_tail_dependence == False:
                        if t_condcorr_metrics['max_leftside'] < t_condcorr_metrics['min_rightside']:
                            t_upper_tail_dependence = True
                        elif t_condcorr_metrics['min_leftside'] > t_condcorr_metrics['max_rightside']:
                            t_lower_tail_dependence = True
                        else:
                            if t_condcorr_metrics['mean_leftside'] < t_condcorr[t_condcorr.index < 0].iloc[-1] and \
                                    t_condcorr_metrics['mean_rightside'] > t_condcorr[t_condcorr.index > 0].iloc[0]:
                                t_upper_tail_dependence = True
                            elif t_condcorr_metrics['mean_leftside'] > t_condcorr[t_condcorr.index < 0].iloc[-1] and \
                                    t_condcorr_metrics['mean_rightside'] < t_condcorr[t_condcorr.index > 0].iloc[0]:
                                t_lower_tail_dependence = True
                            else:
                                t_target_correlation_sign = -11

                    ## For deeper trees: try NCS if non-monotone and flag is on
                    if t_monotone_dependence == -1 and use_ncs_on_deepertrees:
                        t_copulas_ncs_est = self.select_best_bivariate_ncscopula(
                            trees_hfunc[str(tree - 1)].iloc[:, [0, i]])
                        t_copulas_ncs = t_copulas_ncs_est['copula']
                        t_copulas_ncs_rotation = t_copulas_ncs_est['best_rotation']
                        t_copulas_ncs_bic = t_copulas_ncs_est['best_bic']
                        t_copulas_ncs_family = t_copulas_ncs_est['best_family']
                        t_copulas_ncs_params = t_copulas_ncs_est['copula']['params']
                        ###print(f"t_copulas_ncs_bic : {t_copulas_ncs_bic}")

                    ## For deeper trees: try mixture if non-monotone and flag is on
                    if t_monotone_dependence == -1 and use_mixture_on_deepertrees:
                        t_target_mix_specs = [t_lower_tail_dependence, t_upper_tail_dependence]
                        t_mask_mix = copulas_mixture_specifications.eq(t_target_mix_specs).all(axis=1)
                        t_matching_mix_index = list(copulas_mixture_specifications.index[t_mask_mix])
                        if len(t_matching_mix_index) == 0:
                            t_matching_mix_index = list(copulas_mixture_specifications.index)
                        t_copulas_mixture_est = self.select_best_preselected_mixture_copula(
                            data=trees_hfunc[str(tree - 1)].iloc[:, [0, i]],
                            list_of_copulas_families=t_matching_mix_index)
                        t_copula_mix_bic = t_copulas_mixture_est['best_bic']

                    ## If neither NCS nor mixture was tried, fall back to unrestricted ordinary MLE
                    if not (t_monotone_dependence == -1 and use_ncs_on_deepertrees) and \
                            not (t_monotone_dependence == -1 and use_mixture_on_deepertrees):
                        t_copulas_est = self.select_best_bivariate_copula(trees_hfunc[str(tree - 1)].iloc[:, [0, i]])
                        t_copulas = t_copulas_est['copula']
                        t_copulas_bic = t_copulas_est['best_bic']
                        ###print(f"t_copulas_bic : {t_copulas_bic}")
                    ###print(trees_prespecification[str(tree)].index)

                ## ==============================================================================
                ## Store the four classification characteristics for this edge
                ## ==============================================================================
                trees_prespecification[str(tree)].loc[trees_prespecification[str(tree)].index[i - 1],
                'lower_tail_dependence'] = t_lower_tail_dependence

                trees_prespecification[str(tree)].loc[trees_prespecification[str(tree)].index[i - 1],
                'upper_tail_dependence'] = t_upper_tail_dependence

                trees_prespecification[str(tree)].loc[trees_prespecification[str(tree)].index[i - 1],
                'target_correlation_sign'] = t_target_correlation_sign

                trees_prespecification[str(tree)].loc[trees_prespecification[str(tree)].index[i - 1],
                'monotone_dependence'] = t_monotone_dependence

                ## ==============================================================================
                ## STEP 3 — BUILD FALLBACK LISTS
                ## These lists are used by run_vine_optimization: if the optimizer fails to hit
                ## the correlation target with the primary family, it tries alternatives from
                ## these lists without violating the tail dependence constraints.
                ## ==============================================================================

                ## --- Tight fallback list: families matching (ℓ, u, m) ---
                ## Filter the ordinary catalog for same lower/upper tail AND same monotonicity
                t_tt_monotone = copulas_specifications.index[
                    (copulas_specifications.lower_tail_dependence == t_lower_tail_dependence) & \
                    (copulas_specifications.upper_tail_dependence == t_upper_tail_dependence) \
                    & (copulas_specifications.monotone_dependence == t_monotone_dependence)
                    ]

                ## Separate the candidates by correlation sign
                pos_corr_tt = [t for t in t_tt_monotone if copulas_specifications.loc[
                    t, 'target_correlation_sign'] == 1]
                neg_corr_tt = [t for t in t_tt_monotone if copulas_specifications.loc[
                    t, 'target_correlation_sign'] == -1]
                posneg_corr_tt = [t for t in t_tt_monotone if copulas_specifications.loc[
                    t, 'target_correlation_sign'] == -11]
                pos_corr_tt = tuple(pos_corr_tt[0]) if len(pos_corr_tt) > 0 else []
                neg_corr_tt = tuple(neg_corr_tt[0]) if len(pos_corr_tt) > 0 else []

                ## Order the fallback list: put the family matching the target correlation sign first,
                ## the opposite sign second. This way the optimizer tries the most natural family first.
                if t_target_correlation_sign == -1:
                    if len(pos_corr_tt) > 0:
                        t_tt_monotone = [neg_corr_tt] + [pos_corr_tt]
                    else:
                        t_tt_monotone = []
                elif t_target_correlation_sign == 1:
                    if len(pos_corr_tt) > 0:
                        t_tt_monotone = [pos_corr_tt] + [neg_corr_tt]
                    else:
                        t_tt_monotone = []
                else:
                    if len(posneg_corr_tt) > 0:
                        t_tt_monotone = posneg_corr_tt  # + list(copulas_specifications.index[:4])
                    else:
                        t_tt_monotone = []

                ## Remove the already-selected family from the fallback list (no point retrying it)
                if len(t_tt_monotone) > 0:
                    if (t_copulas_bic <= t_copulas_ncs_bic) and (t_lower_tail_dependence != t_upper_tail_dependence) and \
                            (t_copulas_bic != np.inf):
                        i_keep_it = [np.mean(copulas_specifications.loc[val] == \
                                             copulas_specifications.loc[
                                                 (t_copulas.family, int(t_copulas.rotation), False)]) != 1 if \
                                         t_copulas.family.name not in ['joe', 'student'] else \
                                         np.mean(copulas_specifications.loc[val] == \
                                                 copulas_specifications_joe_t.loc[
                                                     (t_copulas.family, int(t_copulas.rotation), False)]) != 1 \
                                     for val in t_tt_monotone]

                        t_tt_monotone = [v for v, m in zip(t_tt_monotone, i_keep_it) if m]

                    testable_copulas[tree - 1, i - 1 + (tree - 1)] = t_tt_monotone

                else:
                    ## No monotone alternatives exist — fall back to mixture alternatives
                    t_tt_nonmonotone = copulas_mixture_specifications.index[
                        (copulas_mixture_specifications.lower_tail_dependence == t_lower_tail_dependence) & \
                        (copulas_mixture_specifications.upper_tail_dependence == t_upper_tail_dependence)]

                    if len(t_tt_nonmonotone) == 0:
                        t_tt_nonmonotone = list(copulas_mixture_specifications.index)

                    testable_copulas[tree - 1, i - 1 + (tree - 1)] = list(t_tt_nonmonotone)

                ## --- Broad fallback list: families matching (ℓ, u) only ---
                ## Same tail dependence, but relaxing monotonicity and correlation sign.
                ## This gives the optimizer maximum freedom when the tight list is exhausted.
                t_tt = copulas_specifications.index[
                    (copulas_specifications.lower_tail_dependence == t_lower_tail_dependence) & \
                    (copulas_specifications.upper_tail_dependence == t_upper_tail_dependence)]
                pos_corr_tt = [t for t in t_tt if copulas_specifications.loc[t, 'target_correlation_sign'] == 1]
                pos_corr_tt = tuple(pos_corr_tt[0]) if len(pos_corr_tt) > 0 else []
                neg_corr_tt = [t for t in t_tt if copulas_specifications.loc[t, 'target_correlation_sign'] == -1]
                neg_corr_tt = tuple(neg_corr_tt[0]) if len(pos_corr_tt) > 0 else []
                posneg_corr_tt = [t for t in t_tt if copulas_specifications.loc[t, 'target_correlation_sign'] == -11]
                if t_target_correlation_sign == -1:
                    t_tt = [neg_corr_tt] + [pos_corr_tt]
                elif t_target_correlation_sign == 1:
                    t_tt = [pos_corr_tt] + [neg_corr_tt]
                else:
                    t_tt = posneg_corr_tt  # + list(copulas_specifications.index[:4])
                # if (t_copulas_bic <= t_copulas_ncs_bic) and (t_lower_tail_dependence != t_upper_tail_dependence) and \
                #         (t_copulas_bic != np.inf):
                #     i_keep_it = [np.mean(copulas_specifications.loc[val] == \
                #                          copulas_specifications.loc[
                #                              (t_copulas.family, int(t_copulas.rotation), False)]) != 1 if \
                #                      t_copulas.family.name not in ['joe', 'student'] else \
                #                      np.mean(copulas_specifications.loc[val] == \
                #                              copulas_specifications_joe_t.loc[
                #                                  (t_copulas.family, int(t_copulas.rotation), False)]) != 1 \
                #                  for val in t_tt]
                #     t_tt = [v for v, m in zip(t_tt, i_keep_it) if m]

                testable_copulas_only_tail[tree - 1, i - 1 + (tree - 1)] = t_tt

                ## ==============================================================================
                ## STEP 4 — MODEL SELECTION BY BIC AND H-FUNCTION PROPAGATION
                ## Compare ordinary, NCS, and mixture BICs. Pick the winner.
                ## Then compute the h-function of the winner to produce the conditional
                ## pseudo-observations that feed into the next tree level.
                ## ==============================================================================
                ii = ii + 1

                ## --- CASE 1: ordinary copula wins (or ties) ---
                if ((t_copulas_bic <= t_copulas_ncs_bic) and (t_copulas_bic <= t_copula_mix_bic)) and \
                        t_copulas_bic != np.inf:

                    ## Store parameters: primary parameter, optional second parameter, no NCS params
                    thetas_final_1p[tree - 1, i - 1 + (tree - 1)] = t_copulas.parameters[0, 0]
                    if len(t_copulas.parameters) == 2:
                        thetas_final_2p[tree - 1, i - 1 + (tree - 1)] = t_copulas.parameters[1, 0]
                    a1_final[tree - 1, i - 1 + (tree - 1)] = np.nan
                    a2_final[tree - 1, i - 1 + (tree - 1)] = np.nan
                    fams_cops[tree - 1, i - 1 + (tree - 1)] = t_copulas.family
                    fams_ncscops_status[tree - 1, i - 1 + (tree - 1)] = False
                    fams_rots[tree - 1, i - 1 + (tree - 1)] = t_copulas.rotation
                    fams_mixture_status[tree - 1, i - 1 + (tree - 1)] = False
                    fams_status[tree - 1, i - 1 + (tree - 1)] = 'ordinary'

                    ## Compute h-function and store with appropriate label
                    if tree == 1:
                        trees_copulas[str(tree)][str(i + 1) + ',' + str(tree)] = t_copulas

                        trees_hfunc[str(tree)][str(i + 1) + '|1'] = \
                            t_copulas.hfunc1(trees_hfunc[str(tree - 1)].iloc[:, [0, i]])

                    elif tree == 2:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|1'] = t_copulas

                        trees_hfunc[str(tree)][str(ii) + '|12'] = \
                            t_copulas.hfunc1(trees_hfunc[str(tree - 1)].iloc[:, [0, i]])

                    elif tree >= 3:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|' + ''.join(
                            str(num) for num in np.arange(1, tree))] = t_copulas

                        trees_hfunc[str(tree)][str(ii) + '|' + ''.join(str(num) for num in np.arange(1, tree + 1))] = \
                            t_copulas.hfunc1(trees_hfunc[str(tree - 1)].iloc[:, [0, i]])

                ## --- CASE 2: NCS copula wins ---
                elif ((t_copulas_bic > t_copulas_ncs_bic) and (t_copula_mix_bic > t_copulas_ncs_bic)) and \
                        t_copulas_ncs_bic != np.inf:

                    ## Store NCS parameters: full 3-element vector [base_param, a1, a2]
                    # thetas_final_1p[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_params[0]
                    thetas_final_1p[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_params
                    a1_final[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_params[1]
                    a2_final[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_params[2]
                    fams_cops[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_family
                    fams_ncscops_status[tree - 1, i - 1 + (tree - 1)] = True
                    fams_rots[tree - 1, i - 1 + (tree - 1)] = t_copulas_ncs_rotation
                    fams_mixture_status[tree - 1, i - 1 + (tree - 1)] = False
                    fams_status[tree - 1, i - 1 + (tree - 1)] = 'ncscopula'

                    ## Compute h-function via the NCS-specific h-function implementation
                    if tree == 1:
                        trees_copulas[str(tree)][str(i + 1) + ',' + str(tree)] = t_copulas_ncs_est

                        trees_hfunc[str(tree)][str(i + 1) + '|1'] = \
                            self.hfunc1_ncs_general(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                    u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                    params=t_copulas_ncs_params,
                                                    family=t_copulas_ncs_family,
                                                    rotation=t_copulas_ncs_rotation)

                    elif tree == 2:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|1'] = t_copulas_ncs_est

                        trees_hfunc[str(tree)][str(ii) + '|12'] = \
                            self.hfunc1_ncs_general(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                    u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                    params=t_copulas_ncs_params,
                                                    family=t_copulas_ncs_family,
                                                    rotation=t_copulas_ncs_rotation)

                    elif tree >= 3:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|' + ''.join(
                            str(num) for num in np.arange(1, tree))] = t_copulas_ncs_est

                        trees_hfunc[str(tree)][str(ii) + '|' + ''.join(str(num) for num in np.arange(1, tree + 1))] = \
                            self.hfunc1_ncs_general(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                    u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                    params=t_copulas_ncs_params,
                                                    family=t_copulas_ncs_family,
                                                    rotation=t_copulas_ncs_rotation)

                ## --- CASE 3: mixture copula wins ---
                elif ((t_copulas_bic > t_copula_mix_bic) and (t_copulas_ncs_bic > t_copula_mix_bic)) and \
                        t_copula_mix_bic != np.inf:

                    ## Store mixture parameters: weight + component parameters; family is a list of Bicop objects
                    thetas_final_1p[tree - 1, i - 1 + (tree - 1)] = t_copulas_mixture_est['best_params'].copy()
                    fams_cops[tree - 1, i - 1 + (tree - 1)] = t_copulas_mixture_est['best_copula'].copy()
                    fams_ncscops_status[tree - 1, i - 1 + (tree - 1)] = False
                    fams_mixture_status[tree - 1, i - 1 + (tree - 1)] = True
                    fams_status[tree - 1, i - 1 + (tree - 1)] = 'mixture'

                    ## Compute h-function via the mixture-specific h-function implementation
                    if tree == 1:
                        trees_copulas[str(tree)][str(i + 1) + ',' + str(tree)] = t_copulas_mixture_est

                        trees_hfunc[str(tree)][str(i + 1) + '|1'] = \
                            self.hfunc1_mixture(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                params=t_copulas_mixture_est['best_params'],
                                                families_w_rotations=t_copulas_mixture_est['best_copula'])


                    elif tree == 2:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|1'] = t_copulas_mixture_est

                        trees_hfunc[str(tree)][str(ii) + '|12'] = \
                            self.hfunc1_mixture(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                params=t_copulas_mixture_est['best_params'],
                                                families_w_rotations=t_copulas_mixture_est['best_copula'])

                    elif tree >= 3:
                        trees_copulas[str(tree)][str(ii) + ',' + str(tree) + '|' + ''.join(
                            str(num) for num in np.arange(1, tree))] = t_copulas_mixture_est

                        trees_hfunc[str(tree)][str(ii) + '|' + ''.join(str(num) for num in np.arange(1, tree + 1))] = \
                            self.hfunc1_mixture(u1=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 0],
                                                u2=trees_hfunc[str(tree - 1)].iloc[:, [0, i]].iloc[:, 1],
                                                params=t_copulas_mixture_est['best_params'],
                                                families_w_rotations=t_copulas_mixture_est['best_copula'])

                        # print(tree)
                ##break

        ## ==============================================================================
        ## Return all fitted parameters, h-function values, and fallback lists
        ## ==============================================================================
        return {'thetas_final_1p': thetas_final_1p,
                'thetas_final_2p': thetas_final_2p,
                'a1_final': a1_final,
                'a2_final': a2_final,
                'fams_cops': fams_cops,
                'fams_ncscops_status': fams_ncscops_status,
                'fams_rots': fams_rots,
                'trees_copulas': trees_copulas,
                'trees_hfunc': trees_hfunc,
                'trees_prespecification': trees_prespecification,
                'copulas_specifications': copulas_specifications,
                'copulas_specifications_joe_t': copulas_specifications_joe_t,
                'copulas_mixture_specifications': copulas_mixture_specifications,
                'fams_mixture_status': fams_mixture_status,
                'fams_status': fams_status,
                'testable_copulas': testable_copulas,
                'testable_copulas_only_tail': testable_copulas_only_tail}

    def simulate_CVine(self, thetas_1p, thetas_2p, a1_s, a2_s, fams, rotations, ncsstatus, mixturestatus, familystatus,
                       n=1000):

        W = np.random.uniform(0, 1, size=[n, thetas_1p.shape[0] + 1])

        U = W.copy() * 0
        U[:, 0] = W[:, 0].copy()

        for column_id in range(U.shape[1] - 1):
            t_theta_final_1p = thetas_1p[:column_id + 1, column_id]  ## [C_{1,x}, C_{2|1, x|1}, C_{3|12, x|12}...]
            t_theta_final_2p = thetas_2p[:column_id + 1, column_id]  ## [C_{1,x}, C_{2|1, x|1}, C_{3|12, x|12}...]
            t_a1 = a1_s[:column_id + 1, column_id]
            t_a2 = a2_s[:column_id + 1, column_id]
            t_fams = fams[:column_id + 1, column_id]
            t_rots = rotations[:column_id + 1, column_id]
            t_ncsstatus = ncsstatus[:column_id + 1, column_id]
            t_mixturestatus = mixturestatus[:column_id + 1, column_id]
            t_familystatus = familystatus[:column_id + 1, column_id]

            ##print(t_rots)
            ##print(rotations)
            temp_U = self.U_last(
                temp_W=W[:, :(column_id + 2)],
                temp_rots=t_rots,
                temp_fams=t_fams,
                temp_theta_final_1p=t_theta_final_1p,
                temp_theta_final_2p=t_theta_final_2p,
                # temp_a1_final=t_a1,
                # temp_a2_final=t_a2,
                # temp_ncsstatus=t_ncsstatus,
                # temp_mixturestatus=t_mixturestatus,
                temp_familystatus=t_familystatus)

            U[:, column_id + 1] = temp_U.copy()

        return U

    def U_last(self, temp_W, temp_rots, temp_fams, temp_theta_final_1p, temp_theta_final_2p
               # , temp_a1_final, temp_a2_final, temp_ncsstatus, temp_mixturestatus
              , temp_familystatus):

        ind_u = temp_W.shape[1]
        temp_U = temp_W[:, (ind_u - 1)]

        for ii in range(len(temp_theta_final_1p)):
            kk = len(temp_theta_final_1p) - ii

            if temp_familystatus[kk - 1] == 'ordinary':
                if temp_fams[kk - 1] != pv.BicopFamily.student:
                    temp_U = pv.Bicop(family=temp_fams[kk - 1],
                                      rotation=int(temp_rots[kk - 1]),
                                      parameters=[temp_theta_final_1p[kk - 1]]
                                      ).hinv1(np.array([temp_W[:, kk - 1], temp_U]).T)

                elif temp_fams[kk - 1] == pv.BicopFamily.student:
                    temp_U = pv.Bicop(family=temp_fams[kk - 1],
                                      rotation=int(temp_rots[kk - 1]),
                                      parameters=[temp_theta_final_1p[kk - 1], temp_theta_final_2p[kk - 1]]
                                      ).hinv1(np.array([temp_W[:, kk - 1], temp_U]).T)

            elif temp_familystatus[kk - 1] == 'ncscopula':
                temp_U = self.hinv1_ncs_general_vec_fast(
                    u1=temp_W[:, kk - 1],
                    p=temp_U,
                    # params=[temp_theta_final_1p[kk - 1],
                    #         temp_a1_final[kk - 1],
                    #         temp_a2_final[kk - 1]],  # [theta, a1, a2]
                    params=temp_theta_final_1p[kk - 1],  # [theta, a1, a2]
                    family=temp_fams[kk - 1],
                    rotation=int(temp_rots[kk - 1]),
                    tol=1e-5,
                    eps=1e-10,
                    n_iter=60)

            elif temp_familystatus[kk - 1] == 'mixture':
                ###print(f"kk-1 : {kk-1}")
                temp_U = self.hinv1_mixture(
                    u1=temp_W[:, kk - 1],
                    p=temp_U,
                    params=temp_theta_final_1p[kk - 1],  # [theta, a1, a2]
                    families_w_rotations=temp_fams[kk - 1])

        return temp_U

    def thetas_for_Corr_all_in_CVine(self,
                                     temp_theta_final_1p_,
                                     args):

        temp_W_ = args[0].copy()
        temp_rots_ = args[1].copy()
        temp_fams_ = args[2].copy()
        temp_theta_2p_ = args[3].copy()
        temp_a1_final_ = args[4].copy()
        temp_a2_final_ = args[5].copy()
        temp_ncsstatus_ = args[6].copy()
        temp_mixturestatus_ = args[7].copy()
        temp_familystatus_ = args[8].copy()
        R_ = args[9].copy()
        optimal_params_ = args[10].copy()
        X_sim_ = args[11].copy()
        Y_sim_ = args[12].copy()

        temp_theta_final_1p_reorg_ = []
        curr_end = 0
        for i in range(len(temp_familystatus_)):
            if temp_familystatus_[i] in ['ordinary']:
                temp_theta_final_1p_reorg_.append(temp_theta_final_1p_[curr_end])
                curr_end = curr_end+1
            elif temp_familystatus_[i] in ['mixture', 'ncscopula']:
                temp_theta_final_1p_reorg_.append(temp_theta_final_1p_[curr_end:(curr_end+3)].copy())
                curr_end = curr_end+3

        U_last_ = self.U_last(
                temp_W=temp_W_,
                temp_rots=temp_rots_,
                temp_fams=temp_fams_,
                temp_theta_final_1p=temp_theta_final_1p_reorg_,
                temp_theta_final_2p=temp_theta_2p_,
                # temp_a1_final=temp_a1_final_,
                # temp_a2_final=temp_a2_final_,
                # temp_ncsstatus=temp_ncsstatus_,
                # temp_mixturestatus=temp_mixturestatus_,
                temp_familystatus=temp_familystatus_
        )

        X_sim_[R_.columns[-1]] = norm.ppf(U_last_)

        if optimal_params_.loc[R_.columns[-1], 'Distr'] == 'JSU':
            Y_sim_last_ = self.sim_JSU_with_U(U_last_,
                                              params=optimal_params_.loc[R_.columns[-1],
                                              ['a', 'b', 'c', 'd']].values)

        Y_sim_[R_.columns[-1]] = Y_sim_last_.copy()
        corr_Y_sim_ = np.corrcoef(Y_sim_.T.values)
        res_ = np.linalg.norm(corr_Y_sim_[:, -1] - R_.values[:, -1])  # /len(temp_theta_final_1p_)

        return res_

    def __init__(self, tol_opt, n_samples, use_ncs_on_deepertrees, use_ncs_on_firsttree,
                 use_mixture_on_firsttree, use_mixture_on_deepertrees, force_try_ncscopula,
                 tol_for_optimization_func=5e-8):
        """Initialize the complete simulation system."""

        # Global variables for compatibility with original code
        self.tol_ = tol_opt
        self.n_samples = n_samples
        self.use_ncs_on_deepertrees = use_ncs_on_deepertrees
        self.use_ncs_on_firsttree = use_ncs_on_firsttree
        self.use_mixture_on_deepertrees = use_mixture_on_deepertrees
        self.use_mixture_on_firsttree = use_mixture_on_firsttree
        self.tol_for_optimization_func = tol_for_optimization_func  # paper value: 5e-8
        self.force_try_ncscopula = force_try_ncscopula

    def _fit_jsu_parameters(self, targeted_mom3, targeted_mom4):
        """Fit JSU distribution parameters for each asset."""
        print("Fitting JSU distribution parameters...")

        optimal_params = pd.DataFrame(np.zeros([len(targeted_mom3), 6]),
                                      index=targeted_mom3.index,
                                      columns=['a', 'b', 'c', 'd', 'fun', 'Distr'])

        for i, asset in enumerate(targeted_mom3.index):
            try:
                params, res = self.find_params_for_moments_matching_JSU(
                    [0, 1, targeted_mom3.loc[asset], targeted_mom4.loc[asset] - 3],
                    x0=[0, 1, 1.5, 1],
                    method='Nelder-Mead'
                )
                optimal_params.iloc[i, :-2] = params
                optimal_params.iloc[i, -2] = res.fun
                optimal_params.iloc[i, -1] = 'JSU'

                if res.fun > 1e-5:
                    print(f"Warning: Poor fit for {asset}, fun = {res.fun}")

            except Exception as e:
                print(f"Error fitting {asset}: {e}")
                # Use default parameters
                optimal_params.iloc[i, :-2] = [0, 1, 1.5, 1]
                optimal_params.iloc[i, -2] = 1e-3
                optimal_params.iloc[i, -1] = 'JSU'

        return optimal_params

    def fit_and_structure_CVine(self, setup_data):

        historical_data = setup_data['historical_data'].copy()
        asset_order = setup_data['asset_order'].copy()
        LTCMAs_corr = setup_data['ltcma_corr'].copy()

        # out_CVine = self.estimate_CVine(input_data=historical_data[asset_order], tryncs=True)
        # out_CVine = self.estimate_CVine_preselected(input_data=historical_data, target_corr=LTCMAs_corr,
        #                                             use_ncs_on_deepertrees=self.use_ncs_on_deepertrees,
        #                                             use_ncs_on_firsttree=self.use_ncs_on_firsttree)

        out_CVine = self.estimate_CVine_preselected_V2(
            input_data=historical_data,
            target_corr=LTCMAs_corr,
            use_ncs_on_deepertrees=self.use_ncs_on_deepertrees,
            use_ncs_on_firsttree=self.use_ncs_on_firsttree,
            use_mixture_on_deepertrees=self.use_mixture_on_deepertrees,
            use_mixture_on_firsttree=self.use_mixture_on_firsttree,
            force_try_ncscopula=self.force_try_ncscopula)

        fams_cops = out_CVine['fams_cops'].copy()
        fams_cops_name = np.zeros(fams_cops.shape, dtype=object) * np.nan
        for i in range(fams_cops_name.shape[0]):
            for j in range(fams_cops_name.shape[1]):
                if not isinstance(fams_cops[i, j], list):
                    try:
                        fams_cops_name[i, j] = fams_cops[i, j].name
                    except:
                        pass
                elif isinstance(fams_cops[i, j], list):
                    fams_cops_name[i, j] = [val.family.name for val in fams_cops[i, j]]

        CVinefitresults = {'thetas_final_1p': out_CVine['thetas_final_1p'],
                           'thetas_final_2p': out_CVine['thetas_final_2p'],
                           'a1_final': out_CVine['a1_final'],
                           'a2_final': out_CVine['a2_final'],
                           'fams_cops': out_CVine['fams_cops'],
                           'fams_cops_name': fams_cops_name.copy(),
                           'fams_ncscops_status': out_CVine['fams_ncscops_status'],
                           'fams_rots': out_CVine['fams_rots'],
                           'LTCMAs_corr': LTCMAs_corr,
                           'trees_copulas': out_CVine['trees_copulas'],
                           'trees_hfunc': out_CVine['trees_hfunc'],
                           'trees_prespecification': out_CVine['trees_prespecification'],
                           'copulas_specifications': out_CVine['copulas_specifications'],
                           'testable_copulas': out_CVine['testable_copulas'],
                           'fams_mixture_status': out_CVine['fams_mixture_status'],
                           'fams_status': out_CVine['fams_status'],
                           'copulas_specifications_joe_t': out_CVine['copulas_specifications_joe_t'],
                           'copulas_mixture_specifications': out_CVine['copulas_mixture_specifications'],
                           'testable_copulas_only_tail': out_CVine['testable_copulas_only_tail']
                           }

        return CVinefitresults

    def _make_bound_callback(self, bounds, eps=1e-7):
        state = {"x_last": None}

        def cb(xk):
            state["x_last"] = xk.copy()
            for xi, (lo, hi) in zip(xk, bounds):
                if lo is not None and abs(xi - lo) <= eps:
                    raise self._EarlyStopSLSQP("hit lower bound")
                if hi is not None and abs(xi - hi) <= eps:
                    raise self._EarlyStopSLSQP("hit upper bound")

        return cb, state

    def run_vine_optimization(self, setup_data, CVinefitresults):
        """
        Run the complete vine copula parameter optimization.

        This implements the first code block from the original notebook.

        Parameters:
        -----------
        setup_data : dict
            Data and parameters from setup phase

        Returns:
        --------
        dict
            Optimization results
        """
        print("\n" + "=" * 80)
        print("VINE COPULA PARAMETER OPTIMIZATION")
        print("=" * 80)

        start_time = time.time()

        # Extract setup data
        ltcma_corr = setup_data['ltcma_corr']
        asset_order = setup_data['asset_order']
        targeted_mom1 = setup_data['targeted_mom1']
        targeted_mom2 = setup_data['targeted_mom2']
        targeted_mom3 = setup_data['targeted_mom3']
        targeted_mom4 = setup_data['targeted_mom4']
        optimal_params = setup_data['optimal_params']

        thetas_final_1p = CVinefitresults['thetas_final_1p']
        thetas_final_2p = CVinefitresults['thetas_final_2p']
        a1_final = CVinefitresults['a1_final']
        a2_final = CVinefitresults['a2_final']
        fams_rots = CVinefitresults['fams_rots']
        fams_cops = CVinefitresults['fams_cops']
        fams_ncscops_status = CVinefitresults['fams_ncscops_status']
        fams_mixture_status = CVinefitresults['fams_mixture_status']
        fams_status = CVinefitresults['fams_status']
        LTCMAs_corr = CVinefitresults['LTCMAs_corr']
        trees_copulas = CVinefitresults['trees_copulas']
        trees_hfunc = CVinefitresults['trees_hfunc']
        trees_prespecification = CVinefitresults['trees_prespecification']
        copulas_specifications = CVinefitresults['copulas_specifications']
        testable_copulas = CVinefitresults['testable_copulas']
        testable_copulas_only_tail = CVinefitresults['testable_copulas_only_tail']

        # Initialize random uniform variables (EXACTLY as in original code)
        #np.random.seed(1)
        W = np.random.uniform(0, 1, size=[self.n_samples, len(asset_order)])
        U = W.copy()

        # Initialize first asset transformation (EXACTLY as in original code)
        X_sim = norm.ppf(U[:, 0])
        X_sim = pd.DataFrame(X_sim, columns=[asset_order[0]])
        Y_sim = X_sim * 0

        if optimal_params.loc[asset_order[0], 'Distr'] == 'Fleishman':
            Y_sim_last_ = targeted_mom1.loc[asset_order[0]] + \
                          self.cubic_transform(X_sim.iloc[:, -1],
                                                  params=optimal_params.loc[
                                                      asset_order[0], ['a', 'b', 'c', 'd']].values) * \
                          targeted_mom2.loc[asset_order[0]] ** (1 / 2)

        elif optimal_params.loc[asset_order[0], 'Distr'] == 'JSU':
            Y_sim_last_ = targeted_mom1.loc[asset_order[0]] + \
                          self.sim_JSU_with_U(U[:, 0],
                                         params=optimal_params.loc[
                                             asset_order[0], ['a', 'b', 'c', 'd']].values) * \
                          targeted_mom2.loc[asset_order[0]] ** (1 / 2)

        Y_sim[asset_order[0]] = Y_sim_last_.copy()

        # MAIN OPTIMIZATION LOOP (EXACTLY as in original code)
        for i in range(1, U.shape[1]):
            best_resfun_val, t_resfun_val = np.inf, np.inf

            print(f"\n{'-' * 60}")
            print(f"OPTIMIZING ASSET {i + 1}/{U.shape[1]}: {asset_order[i]}")
            print(f"{'-' * 60}")

            tt_ = time.time()

            # Set up optimization parameters (EXACTLY as in original)
            t_x0 = thetas_final_1p[:i, i - 1].copy()
            t_x0 = np.concatenate([np.atleast_1d(v).astype(float) for v in t_x0])
            t_fam_cops = fams_cops[:i, i - 1].copy()
            t_fam_rots = fams_rots[:i, i - 1].copy()
            t_fam_thetas_2p = thetas_final_2p[:i, i - 1].copy()
            t_fam_a1 = a1_final[:i, i - 1].copy()
            t_fam_a2 = a2_final[:i, i - 1].copy()
            t_fam_ncscops_status = fams_ncscops_status[:i, i - 1].copy()
            t_fam_mixture_status = fams_mixture_status[:i, i - 1].copy()
            t_fam_status = fams_status[:i, i - 1].copy()

            bnds0 = []
            for ii in range(len(t_fam_status)):
                # if not isinstance(t_fam_cops[ii], list) and\
                #         (t_fam_cops[ii] in [pv.BicopFamily.gaussian, pv.BicopFamily.student]):
                if t_fam_status[ii] in ['ordinary'] and \
                        (t_fam_cops[ii] in [pv.BicopFamily.gaussian, pv.BicopFamily.student]):
                    bnds0.append((-0.99999, 0.99999))

                elif t_fam_status[ii] in ['ordinary'] and \
                        (t_fam_cops[ii] in [pv.BicopFamily.gumbel]):
                    bnds0.append((1.00001, 25))

                elif t_fam_status[ii] in ['ordinary'] and \
                        (t_fam_cops[ii] in [pv.BicopFamily.joe]):
                    bnds0.append((1.00001, 25))

                elif t_fam_status[ii] in ['ordinary'] and \
                        (t_fam_cops[ii] in [pv.BicopFamily.clayton]):
                    bnds0.append((0.00001, 25))

                elif t_fam_status[ii] in ['mixture']:
                    sub_bnds0 = [(1e-10, 1-1e-10)]   ### for the mixture weight
                    for ii_2 in range(len(t_fam_cops[ii])):
                        if t_fam_cops[ii][ii_2].family in [pv.BicopFamily.clayton]:
                            sub_bnds0.append((0.00001, 25))
                        elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.joe]:
                            sub_bnds0.append((1.00001, 25))
                        elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.gumbel]:
                            sub_bnds0.append((1.00001, 25))
                        elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
                            sub_bnds0.append((-0.99999, 0.99999))
                    bnds0.append(sub_bnds0)

                elif t_fam_status[ii] in ['ncscopula']:
                    if t_fam_cops[ii] in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
                        bnds0.append((-0.99999, 0.99999))

                    elif t_fam_cops[ii] in [pv.BicopFamily.gumbel]:
                        bnds0.append((1.00001, 25))

                    elif t_fam_cops[ii] in [pv.BicopFamily.joe]:
                        bnds0.append((1.00001, 25))

                    elif t_fam_cops[ii] in [pv.BicopFamily.clayton]:
                        bnds0.append((0.00001, 25))

                    bnds0.append((1e-10, 3 - 1e-10))  ## a1
                    bnds0.append((1e-10, 3 - 1e-10))  ## a2

            bnds0 = [tuple(b) for item in bnds0 for b in (item if isinstance(item, list) else [item])]

            # First optimization attempt (EXACTLY as in original)
            cb, cb_state = self._make_bound_callback(bnds0, eps=1e-7)
            try:
                res = minimize(fun=self.thetas_for_Corr_all_in_CVine,
                               x0=t_x0.copy(),
                               args=[W[:, 0:(i + 1)].copy(),
                                     t_fam_rots.copy(),
                                     t_fam_cops.copy(),
                                     t_fam_thetas_2p.copy(),
                                     t_fam_a1.copy(),
                                     t_fam_a2.copy(),
                                     t_fam_ncscops_status.copy(),
                                     t_fam_mixture_status.copy(),
                                     t_fam_status.copy(),
                                     ltcma_corr.iloc[:(i + 1), :(i + 1)].copy(),
                                     optimal_params.iloc[:(i + 1), :].copy(),
                                     X_sim.copy(),
                                     Y_sim.copy()],
                               method="SLSQP",
                               # method="Nelder-Mead",
                               bounds=bnds0,
                               options={'maxiter': 100, 'disp': True},
                               tol=self.tol_)
            except self._EarlyStopSLSQP as e:
                x_last = cb_state["x_last"]
                print(f"SLSQP early-stopped ({e}).")
                # You can evaluate objective at x_last to keep logic consistent:
                f_last = self.thetas_for_Corr_all_in_CVine(
                     x_last,
                    [W[:, 0:(i + 1)].copy(),
                     t_fam_rots.copy(),
                     t_fam_cops.copy(),
                     t_fam_thetas_2p.copy(),
                     t_fam_a1.copy(),
                     t_fam_a2.copy(),
                     t_fam_ncscops_status.copy(),
                     t_fam_mixture_status.copy(),
                     t_fam_status.copy(),
                     ltcma_corr.iloc[:(i + 1), :(i + 1)].copy(),
                     optimal_params.iloc[:(i + 1), :].copy(),
                     X_sim.copy(),
                     Y_sim.copy()]
                )
                # Build a minimal result object
                res = type("Res", (), {})()
                res.x = x_last
                res.fun = f_last
                res.success = False
                res.message = f"Early stop: {e}"

            t_resfun_val = res.fun
            best_resfun_val = t_resfun_val
            best_res = res

            best_t_x0 = t_x0.copy()
            best_t_fam_cops = t_fam_cops.copy()
            best_t_fam_rots = t_fam_rots.copy()
            best_t_fam_thetas_2p = t_fam_thetas_2p.copy()
            best_t_fam_a1 = t_fam_a1.copy()
            best_t_fam_a2 = t_fam_a2.copy()
            best_t_fam_ncscops_status = t_fam_ncscops_status.copy()
            best_t_fam_mixture_status = t_fam_mixture_status.copy()
            best_t_fam_status = t_fam_status.copy()

            print(f'0 - best_resfun_val {best_resfun_val}\n')
            print(f'Best func val {best_resfun_val} - i = {i}')

            if (best_resfun_val >= self.tol_for_optimization_func) and not (best_resfun_val < 1e-7 and i >= 5):
                #and fams_cops[:i, i - 1][-1].name != 'student':
                # print(f"np.max([2, len(testable_copulas_only_tail[:i, i - 1])]) =\
                # {np.max([2, len(testable_copulas_only_tail[:i, i - 1])])}\n")
                # print(testable_copulas_only_tail[:i, i - 1])
                # for i_inv in range(1, np.min([10, np.max([2, len(testable_copulas_only_tail[:i, i - 1])])])):
                for i_inv in range(1, np.max([2, len(testable_copulas_only_tail[:i, i - 1])])):

                    ntry = 0
                    t_x0 = best_t_x0.copy()
                    t_fam_cops = best_t_fam_cops.copy()
                    t_fam_rots = best_t_fam_rots.copy()
                    t_fam_thetas_2p = best_t_fam_thetas_2p.copy()
                    t_fam_a1 = best_t_fam_a1.copy()
                    t_fam_a2 = best_t_fam_a2.copy()
                    t_fam_ncscops_status = best_t_fam_ncscops_status.copy()
                    t_fam_mixture_status = best_t_fam_mixture_status.copy()
                    t_fam_status = best_t_fam_status.copy()

                    t_higher_tree_copulas = testable_copulas_only_tail[:i, i - 1][-i_inv]

                    for fam_rot in t_higher_tree_copulas :

                        t_fam_cops[-i_inv] = fam_rot[0] ## ce que je change
                        if t_fam_status[-i_inv] not in ['ordinary']:
                            t_x0 = np.delete(t_x0, np.arange((-2 - i_inv), (-i_inv)))
                        t_fam_status[-i_inv] = 'ordinary'

                        if t_fam_cops[-i_inv] == pv.BicopFamily.student:
                            t_fam_cops[-i_inv] = pv.BicopFamily.gaussian

                        if t_fam_cops[-i_inv] == pv.BicopFamily.gaussian:
                            t_x0[-i_inv] = 0
                        elif t_fam_cops[-i_inv] in [pv.BicopFamily.gumbel, pv.BicopFamily.joe]:
                            t_x0[-i_inv] = 2
                        elif t_fam_cops[-i_inv] == pv.BicopFamily.clayton:
                            t_x0[-i_inv] = 1
                        t_fam_rots[-i_inv] = int(fam_rot[1])
                        t_fam_mixture_status[-i_inv] = False
                        t_fam_thetas_2p[-i_inv] = np.nan
                        t_fam_a1[-i_inv] = np.nan
                        t_fam_a2[-i_inv] = np.nan
                        t_fam_ncscops_status[-i_inv] = False

                        bnds0 = []
                        for ii in range(len(t_fam_status)):
                            if t_fam_status[ii] in ['ordinary'] and \
                                    (t_fam_cops[ii] in [pv.BicopFamily.gaussian, pv.BicopFamily.student]):
                                bnds0.append((-0.99999, 0.99999))

                            elif t_fam_status[ii] in ['ordinary'] and \
                                    (t_fam_cops[ii] in [pv.BicopFamily.gumbel]):
                                bnds0.append((1.00001, 25))

                            elif t_fam_status[ii] in ['ordinary'] and \
                                    (t_fam_cops[ii] in [pv.BicopFamily.joe]):
                                bnds0.append((1.00001, 25))

                            elif t_fam_status[ii] in ['ordinary'] and \
                                    (t_fam_cops[ii] in [pv.BicopFamily.clayton]):
                                bnds0.append((0.00001, 25))

                            elif t_fam_status[ii] in ['mixture']:
                                sub_bnds0 = [(1e-10, 1 - 1e-10)]  ### for the mixture weight
                                for ii_2 in range(len(t_fam_cops[ii])):
                                    if t_fam_cops[ii][ii_2].family in [pv.BicopFamily.clayton]:
                                        sub_bnds0.append((0.00001, 25))
                                    elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.joe]:
                                        sub_bnds0.append((1.00001, 25))
                                    elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.gumbel]:
                                        sub_bnds0.append((1.00001, 25))
                                    elif t_fam_cops[ii][ii_2].family in [pv.BicopFamily.gaussian,
                                                                         pv.BicopFamily.student]:
                                        sub_bnds0.append((-0.99999, 0.99999))
                                bnds0.append(sub_bnds0)

                            elif t_fam_status[ii] in ['ncscopula']:
                                if t_fam_cops[ii] in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
                                    bnds0.append((-0.99999, 0.99999))

                                elif t_fam_cops[ii] in [pv.BicopFamily.gumbel]:
                                    bnds0.append((1.00001, 25))

                                elif t_fam_cops[ii] in [pv.BicopFamily.joe]:
                                    bnds0.append((1.00001, 25))

                                elif t_fam_cops[ii] in [pv.BicopFamily.clayton]:
                                    bnds0.append((0.00001, 25))

                                bnds0.append((1e-10, 3 - 1e-10))  ## a1
                                bnds0.append((1e-10, 3 - 1e-10))  ## a2
                        bnds0 = [tuple(b) for item in bnds0 for b in (item if isinstance(item, list) else [item])]

                        # First optimization attempt (EXACTLY as in original)
                        res = minimize(fun=self.thetas_for_Corr_all_in_CVine,
                                       x0=t_x0.copy(),
                                       args=[W[:, 0:(i + 1)].copy(),
                                             t_fam_rots.copy(),
                                             t_fam_cops.copy(),
                                             t_fam_thetas_2p.copy(),
                                             t_fam_a1.copy(),
                                             t_fam_a2.copy(),
                                             t_fam_ncscops_status.copy(),
                                             t_fam_mixture_status.copy(),
                                             t_fam_status.copy(),
                                             ltcma_corr.iloc[:(i + 1), :(i + 1)].copy(),
                                             optimal_params.iloc[:(i + 1), :].copy(),
                                             X_sim.copy(),
                                             Y_sim.copy()],
                                       method="SLSQP",
                                       # method="Nelder-Mead",
                                       bounds=bnds0,
                                       options={'maxiter': 100, 'disp': True},
                                       tol=self.tol_)

                        t_resfun_val = res.fun
                        if t_resfun_val < best_resfun_val:
                            best_resfun_val = t_resfun_val
                            best_res = res
                            best_t_x0 = t_x0.copy()
                            best_t_fam_cops = t_fam_cops.copy()
                            best_t_fam_rots = t_fam_rots.copy()
                            best_t_fam_thetas_2p = t_fam_thetas_2p.copy()
                            best_t_fam_a1 = t_fam_a1.copy()
                            best_t_fam_a2 = t_fam_a2.copy()
                            best_t_fam_ncscops_status = t_fam_ncscops_status.copy()
                            best_t_fam_mixture_status = t_fam_mixture_status.copy()
                            best_t_fam_status = t_fam_status.copy()

                        ntry = ntry + 1
                        print(f'Tried to improve optmization {ntry} - best function value {best_resfun_val}\n')

                        if best_resfun_val < self.tol_for_optimization_func:
                            break

                    print(f'Tried to improve optimization {ntry} - best func val {best_resfun_val} - i_inv = {i_inv} - i = {i}')
                    if (best_resfun_val < self.tol_for_optimization_func) or\
                            (best_resfun_val < 1e-5 and i_inv>=5) or (best_resfun_val < 1e-5 and i>=4):
                        break

            # Update parameters and calculate new U (EXACTLY as in original)

            best_t_params = best_res.x.copy()
            best_t_params_reorg = []
            curr_end = 0
            for i_i in range(len(best_t_fam_status)):
                if best_t_fam_status[i_i] in ['ordinary']:
                    best_t_params_reorg.append(best_t_params[curr_end])
                    curr_end = curr_end + 1
                elif best_t_fam_status[i_i] in ['mixture', 'ncscopula']:
                    best_t_params_reorg.append(best_t_params[curr_end:(curr_end + 3)].copy())
                    curr_end = curr_end + 3

            thetas_final_1p[:i, i - 1] = best_t_params_reorg.copy()
            fams_cops[:i, i - 1] = best_t_fam_cops.copy()
            fams_rots[:i, i - 1] = best_t_fam_rots.copy()
            thetas_final_2p[:i, i - 1] = best_t_fam_thetas_2p.copy()
            a1_final[:i, i - 1] = best_t_fam_a1.copy()
            a2_final[:i, i - 1] = best_t_fam_a2.copy()
            fams_ncscops_status[:i, i - 1] = best_t_fam_ncscops_status.copy()
            fams_mixture_status[:i, i - 1] = best_t_fam_mixture_status.copy()
            fams_status[:i, i - 1] = best_t_fam_status.copy()

            U[:, i] = self.U_last(W[:, 0:(i + 1)].copy(),
                                  fams_rots[:i, i - 1].copy(),
                                  fams_cops[:i, i - 1].copy(),
                                  thetas_final_1p[:i, i - 1].copy(),
                                  thetas_final_2p[:i, i - 1].copy(),
                                  # a1_final[:i, i - 1].copy(),
                                  # a2_final[:i, i - 1].copy(),
                                  # fams_ncscops_status[:i, i - 1].copy(),
                                  # fams_mixture_status[:i, i - 1].copy(),
                                  fams_status[:i, i - 1].copy()
                                  ).copy()

            X_sim = norm.ppf(U[:, :(i + 1)])
            X_sim = pd.DataFrame(X_sim, columns=ltcma_corr.columns[:(i + 1)])
            Y_sim = pd.DataFrame(np.zeros(X_sim.shape), columns=ltcma_corr.columns[:(i + 1)])

            for ii, asset in enumerate(Y_sim.columns):
                if optimal_params.loc[ltcma_corr.columns[ii], 'Distr'] == 'JSU':
                    Y_sim.loc[:, asset] = targeted_mom1.loc[asset] + \
                                          self.sim_JSU_with_U(U[:, ii],
                                                         params=optimal_params.loc[
                                                             asset, ['a', 'b', 'c', 'd']].values) * \
                                          targeted_mom2.loc[asset] ** (1 / 2)

            corr_Y_sim = np.corrcoef(Y_sim.T.values)

            # Display correlation error (EXACTLY as in original)
            corr_error = corr_Y_sim - ltcma_corr.iloc[:(i + 1), :(i + 1)]
            print(corr_error.abs().max())

            print('\n')
            elapsed = time.time() - tt_
            print('Iteration : {} - elapsed time : {}s '.format(i, np.round(elapsed, 2)))
            print('\n\n')

            # # Save intermediate results (modified path for current environment)
            # if i >= 10:
        # output_dir = r'/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion'
        # pd.DataFrame(thetas_final_1p, index=asset_order[:-1], columns=asset_order[1:]).to_csv(
        #     f'{output_dir}/theta_final_1p__{i}.csv')
        # pd.DataFrame(a1_final, index=asset_order[:-1], columns=asset_order[1:]).to_csv(
        #     f'{output_dir}/theta_final_2p__{i}.csv')
        # pd.DataFrame(fams_code_, index=asset_order[:-1], columns=asset_order[1:]).to_csv(
        #     f'{output_dir}/fams_code___{i}.csv')
        # pd.DataFrame(fams_vine_, index=asset_order[:-1], columns=asset_order[1:]).to_csv(
        #     f'{output_dir}/fams_vine___{i}.csv')
        # pd.DataFrame(W, columns=asset_order).to_csv(f'{output_dir}/W___.csv')
        # pd.DataFrame(U, columns=asset_order).to_csv(f'{output_dir}/U___.csv')
        # Y_sim.to_csv(f'{output_dir}/Y___.csv')
        # print(f"Intermediate results saved for iteration {i}")


        total_elapsed = time.time() - start_time
        print(f"{'=' * 80}")
        print(f"VINE OPTIMIZATION COMPLETED")
        print(f"Total optimization time: {np.round(total_elapsed, 2)}s")
        print(f"{'=' * 80}")

        return {
            'thetas_final_1p': thetas_final_1p,
            'thetas_final_2p': thetas_final_2p,
            'a1_final': a1_final,
            'a2_final': a2_final,
            'fams_cops': fams_cops,
            'fams_rots': fams_rots,
            'fams_ncscops_status':fams_ncscops_status,
            'fams_mixture_status': fams_mixture_status,
            'fams_status': fams_status,
            'trees_copulas': trees_copulas,
            'trees_hfunc': trees_hfunc,
            'trees_prespecification':trees_prespecification,
            'copulas_specifications':copulas_specifications,
            'testable_copulas': testable_copulas,
            'testable_copulas_only_tail': testable_copulas_only_tail,
            'U': U,
            'W': W,
            'Y_sim': Y_sim,
            'ltcma_corr': ltcma_corr,
            'asset_order': asset_order,
            'optimal_params': optimal_params,
            'targeted_mom1': targeted_mom1,
            'targeted_mom2': targeted_mom2,
            'targeted_mom3': targeted_mom3,
            'targeted_mom4': targeted_mom4
        }

    def run_multi_year_simulation(self, vine_results, n_year=10, n_per_year=5000, corr_tol=5e-2):
        """
        Run the multi-year simulation with quality controls.

        This implements the second code block from the original notebook.

        Parameters:
        -----------
        vine_results : dict
            Results from vine optimization
        n_year : int, default=10
            Number of years to simulate
        n_per_year : int, default=5000
            Number of scenarios per year

        Returns:
        --------
        list
            List of DataFrames containing simulated data for each year
        """
        print("\n" + "=" * 80)
        print("MULTI-YEAR SCENARIO GENERATION")
        print("=" * 80)

        start_time = time.time()

        # Extract parameters from vine results
        thetas_final_1p = vine_results['thetas_final_1p']
        thetas_final_2p = vine_results['thetas_final_2p']
        a1_final = vine_results['a1_final']
        a2_final = vine_results['a2_final']
        fams_cops = vine_results['fams_cops']
        fams_ncscops_status = vine_results['fams_ncscops_status']
        fams_mixture_status = vine_results['fams_mixture_status']
        fams_status = vine_results['fams_status']
        fams_rots = vine_results['fams_rots']
        ltcma_corr = vine_results['ltcma_corr']
        optimal_params = vine_results['optimal_params']
        targeted_mom1 = vine_results['targeted_mom1']
        targeted_mom2 = vine_results['targeted_mom2']
        asset_order = vine_results['asset_order']

        # Set up target moments for multi-year simulation
        targeted_mom3 = vine_results['targeted_mom3']
        targeted_mom4 = vine_results['targeted_mom4'] #+ 3
        targets_correlation = ltcma_corr

        targets = pd.concat([targeted_mom1, targeted_mom2**0.5, targeted_mom3, targeted_mom4], axis=1)
        targets.columns = ['mean (target)', 'volatility (target)', 'skewness (target)', 'kurtosis (target)']

        n_total = n_year * n_per_year
        list_df_sim = []
        list_calibrated = []

        # np.random.seed(123)  # EXACTLY as in original

        print(f"Generating {n_year} years with {n_per_year} samples per year")
        print(f"Total samples: {n_total}")

        # FIRST YEAR SIMULATION (EXACTLY as in original code)
        max_error_temp_mean_vol = 5
        max_error_temp_skew_kurt = 5
        max_error_temp_corr = 5
        dont_move_to_next = True

        while (dont_move_to_next == True):
            # Generate standard normal random variable with correlation equal to the intermediate correlation matrix
            # cop = pv.Vinecop(self.mat_CVine(thetas_final_1p.shape[1] + 1),
            #                  self.pcs_RVine(fams_code_, thetas_final_1p, thetas_final_2p))
            # U_sim = cop.simulate(n=n_per_year)
            U_sim = self.simulate_CVine(
                thetas_1p=thetas_final_1p,
                thetas_2p=thetas_final_2p,
                a1_s=a1_final,
                a2_s=a2_final,
                fams=fams_cops,
                rotations=fams_rots,
                ncsstatus=fams_ncscops_status,
                mixturestatus=fams_mixture_status,
                familystatus=fams_status,
                n=n_per_year)

            optimal_params_post = optimal_params.iloc[:, :-1].copy()

            for i_, asset_ in enumerate(targeted_mom1.index):
                res = 0
                optimal_params_post.iloc[i_, :-1], res = self.find_params_for_moments_matching_JSU_with_U(
                    [0, 1, targeted_mom3[asset_], targeted_mom4[asset_] - 3],
                    x0=optimal_params.iloc[i_, :-2],
                    U=U_sim[:, i_],
                    method='SLSQP')
                optimal_params_post.iloc[i_, -1] = res.fun

            # Apply the nonnormal transformation to the generated standard normals
            Y_AR_year0 = pd.DataFrame(np.zeros(U_sim.shape), columns=ltcma_corr.columns)
            for ii, asset in enumerate(Y_AR_year0.columns):
                Y_AR_year0.loc[:, asset] = targeted_mom1.loc[ltcma_corr.columns[ii]] + \
                                           self.sim_JSU_with_U(U_sim[:, ii],
                                                          params=optimal_params_post.loc[
                                                              asset, ['a', 'b', 'c', 'd']].values) * \
                                           targeted_mom2.loc[asset] ** (1 / 2)

            temp_diff = np.round([np.max(np.abs(Y_AR_year0.mean(0).values - targeted_mom1.values)),
                                  np.max(np.abs(Y_AR_year0.std(0).values - targeted_mom2.values ** (1 / 2))),
                                  np.max(np.abs(skew(Y_AR_year0) - targeted_mom3.values)),
                                  np.max(np.abs(kurtosis(Y_AR_year0) + 3 - targeted_mom4.values)),
                                  np.max(
                                      np.abs(np.corrcoef(Y_AR_year0.to_numpy().T) - targets_correlation.to_numpy()))],
                                 5)

            max_error_temp_mean_vol = np.round(np.max(temp_diff[:2]), 5)
            max_error_temp_skew_kurt = np.round(np.max(temp_diff[2:-1]), 5)
            max_error_temp_corr = temp_diff[-1]

            if (max_error_temp_mean_vol <= 2e-4) and (max_error_temp_skew_kurt <= 5e-2) and (
                    max_error_temp_corr <= corr_tol):
                dont_move_to_next = False
                calibrated_moments = pd.DataFrame(
                    index=targets.index,
                    columns=['mean (target)', 'mean (calibrated)', 'diff mean',
                             'volatility (target)', 'volatility (calibrated)', 'diff volatility',
                             'skewness (target)', 'skewness (calibrated)', 'diff skewness',
                             'kurtosis (target)', 'kurtosis (calibrated)', 'diff kurtosis'])
                calibrated_moments.loc[targets.index, targets.columns] = targets.values
                calibrated_moments['mean (calibrated)'] = Y_AR_year0.mean(0).values
                calibrated_moments['diff mean'] = Y_AR_year0.mean(0).values - targeted_mom1.values
                calibrated_moments['volatility (calibrated)'] = Y_AR_year0.std(0).values
                calibrated_moments['diff volatility'] = Y_AR_year0.std(0).values - targeted_mom2.values ** (1 / 2)
                calibrated_moments['skewness (calibrated)'] = skew(Y_AR_year0)
                calibrated_moments['diff skewness'] = skew(Y_AR_year0) - targeted_mom3.values
                calibrated_moments['kurtosis (calibrated)'] = kurtosis(Y_AR_year0) + 3
                calibrated_moments['diff kurtosis'] = kurtosis(Y_AR_year0) + 3 - targeted_mom4.values
                correlation_diff =  pd.DataFrame((np.corrcoef(Y_AR_year0.to_numpy().T) - targets_correlation.to_numpy()),
                                                 index=targets_correlation.index, columns=targets_correlation.columns)
                calibrated_moments = pd.concat([calibrated_moments, correlation_diff], axis=1)

            print('ii = {} ; max_error_mean_vol = {} ; max_error_skew_kurt = {} ; temp_diff = {}'.format(
                0, max_error_temp_mean_vol, max_error_temp_skew_kurt, temp_diff))

        list_df_sim.append(Y_AR_year0.copy())
        list_calibrated.append(calibrated_moments.copy())

        # REMAINING YEARS SIMULATION (EXACTLY as in original code)
        for ii in range(1, n_year):
            print(f"\nSimulating year {ii + 1}/{n_year}")
            max_error_temp_mean_vol = 5
            max_error_temp_skew_kurt = 5
            max_error_temp_corr = 5
            dont_move_to_next = True

            while (dont_move_to_next == True):
                # cop = pv.Vinecop(self.mat_CVine(thetas_final_1p.shape[1] + 1),
                #                  self.pcs_RVine(fams_code_, thetas_final_1p, thetas_final_2p))
                # U_sim = cop.simulate(n=n_per_year)
                U_sim = self.simulate_CVine(
                    thetas_1p=thetas_final_1p,
                    thetas_2p=thetas_final_2p,
                    a1_s=a1_final,
                    a2_s=a2_final,
                    fams=fams_cops,
                    rotations=fams_rots,
                    ncsstatus=fams_ncscops_status,
                    mixturestatus=fams_mixture_status,
                    familystatus=fams_status,
                    n=n_per_year)
                optimal_params_post = optimal_params.iloc[:, :-1].copy()

                for i_, asset_ in enumerate(targeted_mom1.index):
                    res = 0
                    optimal_params_post.iloc[i_, :-1], res = self.find_params_for_moments_matching_JSU_with_U(
                        [0, 1, targeted_mom3[asset_], targeted_mom4[asset_] - 3],
                        x0=optimal_params.iloc[i_, :-2],
                        U=U_sim[:, i_],
                        method='SLSQP')
                    optimal_params_post.iloc[i_, -1] = res.fun

                temp_Y_AR_year = pd.DataFrame(np.zeros(U_sim.shape), columns=ltcma_corr.columns)
                for i, asset in enumerate(temp_Y_AR_year.columns):
                    temp_Y_AR_year.loc[:, asset] = targeted_mom1.loc[ltcma_corr.columns[i]] + \
                                                   self.sim_JSU_with_U(U_sim[:, i],
                                                                  params=optimal_params_post.loc[
                                                                      asset, ['a', 'b', 'c', 'd']].values) * \
                                                   targeted_mom2.loc[asset] ** (1 / 2)

                temp_diff = np.round([np.max(np.abs(temp_Y_AR_year.mean(0).values - targeted_mom1.values)),
                                      np.max(
                                          np.abs(temp_Y_AR_year.std(ddof=1).values - targeted_mom2.values ** (1 / 2))),
                                      np.max(np.abs(skew(temp_Y_AR_year) - targeted_mom3.values)),
                                      np.max(np.abs(kurtosis(temp_Y_AR_year) + 3 - targeted_mom4.values)),
                                      np.max(np.abs(
                                          np.corrcoef(temp_Y_AR_year.to_numpy().T) - targets_correlation.to_numpy()))],
                                     5)

                max_error_temp_mean_vol = np.round(np.max(temp_diff[:2]), 5)
                max_error_temp_skew_kurt = np.round(np.max(temp_diff[2:-1]), 5)
                max_error_temp_corr = temp_diff[-1]

                if (max_error_temp_mean_vol <= 2e-4) and (max_error_temp_skew_kurt <= 5e-2) and (
                        max_error_temp_corr <= corr_tol):
                    dont_move_to_next = False

                print('ii = {} ; max_error_mean_vol = {} ; max_error_skew_kurt = {} ; temp_diff = {}'.format(
                    ii, max_error_temp_mean_vol, max_error_temp_skew_kurt, temp_diff))

            list_df_sim.append(temp_Y_AR_year.copy())


        total_elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"MULTI-YEAR SIMULATION COMPLETED")
        print(f"Generated {len(list_df_sim)} years of data")
        print(f"Total simulation time: {np.round(total_elapsed, 2)}s")
        print(f"{'=' * 80}")

        return list_df_sim, list_calibrated, targets

    # ------------------------------------------------------------------
    # Inputs as DataFrames (package version of load_data_and_setup)
    # ------------------------------------------------------------------
    def load_data_and_setup(self, ltcma, historical_data, asset_order=None):
        """
        Set up the targets of the simulation (Algorithm 1 of the paper).

        ltcma           : DataFrame indexed by asset, with columns 'Arithmetic Mean'
                          and 'Volatility' (optionally 'Geometric Mean') followed by
                          the correlation-matrix columns, named after the assets.
        historical_data : DataFrame of returns, one column per asset. Used for the
                          target skewness and kurtosis and for copula family selection.
        asset_order     : list of assets; the first one is the central node of the
                          C-vine. Defaults to the columns of historical_data.
        """
        print("Loading data and setting up simulation parameters...")
        historical_data = historical_data.ffill().dropna()
        ltcma = ltcma.copy()
        if 'Geometric Mean' not in ltcma.columns:
            ltcma.insert(0, 'Geometric Mean', ltcma['Arithmetic Mean'] - 0.5 * ltcma['Volatility'] ** 2)
        ltcma_moments = ltcma.loc[:, ['Geometric Mean', 'Arithmetic Mean', 'Volatility']].copy()
        ltcma_moments['Variance'] = ltcma_moments['Volatility'] ** 2
        ltcma_corr = ltcma.loc[:, [c for c in ltcma.columns if c in ltcma.index]].copy()
        ltcma_corr_mat = ltcma_corr.values.astype(float)
        ltcma_corr_mat[np.triu_indices(ltcma_corr_mat.shape[0])] = np.tril(ltcma_corr_mat).T[
            np.triu_indices(ltcma_corr_mat.shape[0])]
        ltcma_corr[:] = ltcma_corr_mat

        if asset_order is None:
            asset_order = list(historical_data.columns)

        # ---- from here on: the original body, unchanged ----
        ltcma_corr = ltcma_corr.loc[asset_order, asset_order].copy()
        ltcma_moments = ltcma_moments.loc[asset_order, :]

        targeted_mom1 = ltcma_moments.loc[asset_order, 'Arithmetic Mean']  # Target means
        targeted_mom2 = ltcma_moments.loc[asset_order, 'Variance']  # Target variances
        targeted_GM = ltcma_moments.loc[asset_order, 'Geometric Mean']  # Target geometric means
        targeted_SD = ltcma_moments.loc[asset_order, 'Volatility']  # Target standard deviations

        targeted_mom3 = historical_data[asset_order].skew()  # Skewness from historical data
        targeted_mom4 = historical_data[asset_order].kurtosis() + 3  # Kurtosis from historical data

        targeted_mom3.name = 'Skewness'
        targeted_mom4.name = 'Kurtosis'

        optimal_params = self._fit_jsu_parameters(targeted_mom3, targeted_mom4)

        df_target_stats = targeted_GM.to_frame().join(targeted_SD)
        df_target_stats = df_target_stats.merge(targeted_mom1.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom2.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom3.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom4.to_frame(), left_index=True, right_index=True)

        ltcma_corr_PD = self.make_positive_definite(ltcma_corr)
        historical_data = historical_data.loc[:, asset_order].copy()
        df_target_stats = df_target_stats.loc[asset_order, :].copy()

        return {
            'ltcma_corr': ltcma_corr_PD,
            'historical_data': historical_data,
            'asset_order': asset_order,
            'targeted_mom1': targeted_mom1,
            'targeted_mom2': targeted_mom2,
            'targeted_mom3': targeted_mom3,
            'targeted_mom4': targeted_mom4,
            'optimal_params': optimal_params,
            'df_target_stats': df_target_stats,
        }

    def run_complete_simulation(self, ltcma, historical_data, asset_order=None,
                                n_year=1, n_per_year=25000, corr_tol=2e-2):
        """
        Complete pipeline (Algorithm 6 of the paper): setup, family selection and
        vine fit, correlation-targeting calibration, scenario generation.
        Returns a dict with 'setup_data', 'CVinefitresults', 'vine_results',
        'simulated_years', 'calibration' (target vs simulated moments and the
        correlation differences of the first period) and 'targets'.
        """
        print("FINANCIAL ASSET RETURN SIMULATION SYSTEM")
        print("Using Vine Copulas and Moment Matching")
        print("=" * 80)

        print("STEP 1: Loading data and setting up parameters...")
        setup_data = self.load_data_and_setup(ltcma, historical_data, asset_order)

        print("\nSTEP 2: Copula family selection and vine fit...")
        CVinefitresults = self.fit_and_structure_CVine(setup_data)

        print("\nSTEP 3: Running vine copula parameter optimization...")
        vine_results = self.run_vine_optimization(setup_data, CVinefitresults)

        print(f"\nSTEP 4: Running multi-year simulation ({n_year} years)...")
        simulated_years, calibration, targets = self.run_multi_year_simulation(
            vine_results, n_year, n_per_year, corr_tol)

        print(f"\n{'=' * 80}")
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print(f"Generated {len(simulated_years)} years of correlated asset return scenarios")
        print(f"Each year contains {simulated_years[0].shape[0]} samples for {simulated_years[0].shape[1]} assets")
        print(f"{'=' * 80}")

        return {
            'setup_data': setup_data,
            'CVinefitresults': CVinefitresults,
            'vine_results': vine_results,
            'simulated_years': simulated_years,
            'calibration': calibration,
            'targets': targets,
        }

    # ------------------------------------------------------------------
    # Synthetic markets from a known C-vine (used by the examples)
    # ------------------------------------------------------------------
    def make_vine_spec(self, asset_order, edges):
        """
        Build the matrices consumed by simulate_CVine from a dict of edges
        {(i, k): spec}, with 1-based variable indices and k < i, i.e. the
        copula C_{ik|1:k-1} of the paper (tree k, variable i).

        spec = (family, rotation, theta)                                  single family
        spec = ('mixture', [(fam1, rot1, th1), (fam2, rot2, th2)], w)    two-component mixture,
                                                                          w = weight on the first component
        Families: 'gaussian', 'clayton', 'gumbel', 'joe', 'frank'.
        """
        d = len(asset_order)
        shape = (d - 1, d - 1)
        thetas_1p = np.full(shape, np.nan, dtype=object)
        thetas_2p = np.full(shape, np.nan)
        a1 = np.full(shape, np.nan)
        a2 = np.full(shape, np.nan)
        fams = np.full(shape, np.nan, dtype=object)
        rots = np.zeros(shape, dtype=int)
        ncs = np.full(shape, False, dtype=object)
        mix = np.full(shape, False, dtype=object)
        status = np.full(shape, np.nan, dtype=object)
        for (i, k), spec in edges.items():
            if not (1 <= k < i <= d):
                raise ValueError(f"edge {(i, k)}: need 1 <= k < i <= {d}")
            r, c = k - 1, i - 2
            if spec[0] == 'mixture':
                comps = [pv.Bicop(self._FAMILY[f], int(rot)) for f, rot, _ in spec[1]]
                thetas_1p[r, c] = np.array([spec[2]] + [th for _, _, th in spec[1]], float)
                fams[r, c] = comps
                mix[r, c] = True
                status[r, c] = 'mixture'
            else:
                fam, rot, th = spec
                thetas_1p[r, c] = float(th)
                fams[r, c] = self._FAMILY[fam]
                rots[r, c] = int(rot)
                status[r, c] = 'ordinary'
        missing = [(i, k) for k in range(1, d) for i in range(k + 1, d + 1) if (i, k) not in edges]
        if missing:
            raise ValueError(f"missing edges: {missing}")
        return dict(thetas_1p=thetas_1p, thetas_2p=thetas_2p, a1_s=a1, a2_s=a2, fams=fams,
                    rotations=rots, ncsstatus=ncs, mixturestatus=mix, familystatus=status)

    def simulate_known_vine(self, spec, jsu_params, mu, sigma, asset_order, n, seed=None):
        """
        Simulate n returns from a known C-vine (spec from make_vine_spec) with
        Johnson SU marginals jsu_params[asset] = (gamma, xi, delta, lambda), then
        rescale to mean mu[asset] and volatility sigma[asset].
        Returns a DataFrame with one column per asset.
        """
        if seed is not None:
            np.random.seed(seed)
        U = self.simulate_CVine(n=n, **spec)
        out = pd.DataFrame(np.zeros((n, len(asset_order))), columns=asset_order)
        for j, a in enumerate(asset_order):
            out[a] = mu[a] + sigma[a] * self.sim_JSU_with_U(U[:, j], np.asarray(jsu_params[a], float))
        return out

    def true_edge_table(self, asset_order, edges):
        """Readable table of the true families of a synthetic market (for the examples)."""
        rows = []
        for (i, k), spec in sorted(edges.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if spec[0] == 'mixture':
                fam = ' + '.join(f"{f} {rot}°" for f, rot, _ in spec[1])
                par = f"w={spec[2]}, theta=({', '.join(str(th) for _, _, th in spec[1])})"
            else:
                fam, par = f"{spec[0]} {spec[1]}°", f"theta={spec[2]}"
            cond = '' if k == 1 else ' | ' + ', '.join(asset_order[:k - 1])
            rows.append({'tree': k, 'edge': f"{asset_order[i - 1]} , {asset_order[k - 1]}{cond}",
                         'true family': fam, 'true parameters': par})
        return pd.DataFrame(rows)

    def selected_edge_table(self, asset_order, CVinefitresults, vine_results=None):
        """
        Readable table of the selected family per edge after Algorithm 3, and of
        the calibrated parameters after Algorithm 4 when vine_results is given.
        """
        fams = CVinefitresults['fams_cops']
        rots = CVinefitresults['fams_rots']
        status = CVinefitresults['fams_status']
        th = (vine_results if vine_results is not None else CVinefitresults)['thetas_final_1p']
        rows = []
        d = len(asset_order)
        for k in range(1, d):
            for i in range(k + 1, d + 1):
                r, c = k - 1, i - 2
                s = status[r, c]
                if s == 'ordinary':
                    fam, par = f"{fams[r, c].name} {int(rots[r, c])}°", f"theta={float(th[r, c]):.3f}"
                elif s == 'mixture':
                    p = np.asarray(th[r, c], float)
                    fam = ' + '.join(f"{b.family.name} {int(b.rotation)}°" for b in fams[r, c])
                    par = f"w={p[0]:.2f}, theta=({p[1]:.3f}, {p[2]:.3f})"
                else:
                    fam, par = str(s), ''
                cond = '' if k == 1 else ' | ' + ', '.join(asset_order[:k - 1])
                rows.append({'tree': k, 'edge': f"{asset_order[i - 1]} , {asset_order[k - 1]}{cond}",
                             'selected family': fam, 'parameters': par})
        return pd.DataFrame(rows)
