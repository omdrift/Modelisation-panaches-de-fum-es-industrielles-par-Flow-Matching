"""Validation-only plateau decisions, independent of raw checkpoint selection."""
from __future__ import annotations

import copy
import math


DEFAULTS = dict(enabled=False, patience=10, min_delta=0.0001, min_updates=5000)


def settings(cfg: dict) -> dict:
    """An absent policy keeps an existing frozen protocol disabled."""
    overrides = cfg['training'].get('early_stopping', {})
    if not isinstance(overrides, dict) or overrides.keys() - DEFAULTS.keys():
        raise ValueError('Unknown or malformed early stopping policy')
    result = DEFAULTS | overrides
    if type(result['enabled']) is not bool:
        raise ValueError('Early stopping enabled must be a boolean')
    for name, minimum in (('patience', 1), ('min_updates', 0)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f'Early stopping {name} must be an integer >= {minimum}')
    delta = result['min_delta']
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError('Early stopping min_delta must be finite and nonnegative')
    return result


class EarlyStopping:
    """Count completed validations since the last significant LPIPS decrease.

    Before min_updates, the reference score is tracked but patience is zero.
    The reference changes only on a decrease greater than min_delta, so small
    gains can accumulate. Raw best.pth selection remains a separate decision.
    """

    def __init__(self, cfg: dict, state: dict | None = None):
        self.policy = settings(cfg)
        self._state = dict(settings=self.policy, best_score=None, best_step=0,
                           bad_checks=0, last_step=0, stopped=False)
        if state is not None:
            self._restore(state)

    def _restore(self, state: dict) -> None:
        if not isinstance(state, dict) or state.keys() != self._state.keys() or state['settings'] != self.policy:
            raise ValueError('Early stopping state does not match the configured policy')
        score = state['best_score']
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('Early stopping reference must be finite')
        for key in ('best_step', 'last_step', 'bad_checks'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('Invalid early stopping counter or step')
        if (state['best_step'] > state['last_step'] or (score is None) != (state['last_step'] == 0)
                or (score is not None and state['best_step'] < 1)
                or state['bad_checks'] > state['last_step']):
            raise ValueError('Inconsistent early stopping history')
        expected = (self.policy['enabled'] and state['last_step'] >= self.policy['min_updates']
                    and state['bad_checks'] >= self.policy['patience'])
        if type(state['stopped']) is not bool or state['stopped'] != expected:
            raise ValueError('Early stopping decision is inconsistent with its counters')
        self._state = copy.deepcopy(state)

    @property
    def stopped(self) -> bool:
        return self._state['stopped']

    def observe(self, score: float, step: int) -> None:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Early stopping requires a finite validation LPIPS')
        if type(step) is not int or step <= self._state['last_step'] or self.stopped:
            raise ValueError('Early stopping validations must advance and cannot follow a stop')
        previous = self._state['best_score']
        improved = previous is None or score < previous - self.policy['min_delta']
        if improved:
            self._state.update(best_score=score, best_step=step, bad_checks=0)
        elif self.policy['enabled'] and step >= self.policy['min_updates']:
            self._state['bad_checks'] += 1
        else:
            self._state['bad_checks'] = 0
        self._state['last_step'] = step
        self._state['stopped'] = self.policy['enabled'] and self._state['bad_checks'] >= self.policy['patience']

    def state_dict(self) -> dict:
        return copy.deepcopy(self._state)

    def metrics(self) -> dict:
        return {f'early_stopping/{name}': value for name, value in
                (self.policy | {key: self._state[key] for key in ('bad_checks', 'best_score', 'best_step', 'stopped')}).items()}


def completion_valid(cfg: dict, completion: dict) -> bool:
    """A short run is complete only with a policy-bound plateau decision."""
    completed, planned = completion.get('completed_updates'), completion.get('planned_updates')
    if (completion.get('status') != 'complete' or type(completed) is not int or type(planned) is not int
            or not 1 <= completed <= planned):
        return False
    if completed == planned:
        return True
    if completion.get('termination_reason') != 'early_stopping' or planned != cfg['training']['max_updates']:
        return False
    try:
        stopper = EarlyStopping(cfg, completion['early_stopping'])
    except (KeyError, TypeError, ValueError):
        return False
    return stopper.stopped and stopper.state_dict()['last_step'] == completed
