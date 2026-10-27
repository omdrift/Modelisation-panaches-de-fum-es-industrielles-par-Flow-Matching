"""Configure a stable named W&B workspace without modifying run history."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4


def reading_guide(method: str) -> str:
    """Keep interpretation beside the curves, including their limits."""
    common = '''### Lire les courbes
Chaque point de qualité utilise les mêmes exemples fixes, en mode évaluation. L'axe horizontal compte les mises à jour de l'optimiseur.

| Question | Lecture et limite |
| --- | --- |
| La qualité progresse-t-elle ? | **LPIPS ↓** mesure l'écart perceptuel. Comparer apprentissage et validation ; aucune valeur ne garantit à elle seule un bon modèle. |
| La fumée est-elle conservée ? | **MSE fumée ↓**, sur RGB dans [0,1], complète la MSE du fond. Les masques sont estimés ; une région absente est exclue, pas comptée comme erreur nulle. |
| Le modèle généralise-t-il ? | **Écart LPIPS = validation − apprentissage**. Une hausse durable de validation avec baisse en apprentissage est un signal à examiner, pas une preuve automatique. Les dates et les vues peuvent différer. |
| Quand l'entraînement s'arrête-t-il ? | Si activée, la **patience** compte les validations sans baisse significative de LPIPS. L'arrêt intervient lorsque le compteur atteint le seuil, après le minimum de mises à jour et la phase initiale ; les valeurs effectives figurent dans la configuration du run. |
| Que regarder visuellement ? | Les images fixes montrent RGB réel, reconstruction/prédiction, erreur et masque estimé. Vérifier disparition, artefacts de fond et contours. |

Le drapeau « surapprentissage » utilise cinq validations et un seuil exploratoire de 10 %. Il n'arrête pas l'entraînement et ne certifie pas l'absence de surapprentissage. Le meilleur checkpoint minimise la LPIPS de validation ; les tests restent réservés à l'évaluation finale.
'''
    if method == 'vqgan':
        common += '''
**VQGAN :** perplexité = exp(entropie des codes), fraction utilisée = codes actifs / taille du dictionnaire. Elles décrivent ce panneau, pas tout le dataset ; les maximiser n'est pas un objectif. Une perte discriminateur proche de 1 ne démontre pas l'équilibre du GAN. La perte de reconstruction d'entraînement utilise [-1,1], donc vaut quatre fois la MSE équivalente dans [0,1]. Les pertes VQ, LPIPS et GAN n'ont pas la même échelle.

**Réglages exploratoires :** le LR VQGAN est défini dans `training.autoencoder`, séparément du LR des prédicteurs. Les valeurs effectivement exécutées sont exportées dans `effective_training.json`. Une calibration sur la fumée reste nécessaire ; reconstruire une image ne valide pas la prédiction de 15 images futures.
'''
    else:
        common += '\n**Prédicteur :** la perte d’entraînement et la LPIPS de validation mesurent des objectifs différents. Les réglages exécutés figurent dans `effective_training.json` ; leur présence dans une configuration ne constitue pas une calibration.\n'
    return common


def section_plan(method: str) -> list[dict]:
    """Comparable quantities share a chart; detailed diagnostics stay collapsed."""
    quality = [
        ('LPIPS ↓ : apprentissage et validation', ['train_eval/lpips', 'validation/lpips', 'selection/best_validation_lpips']),
        ('MSE fumée estimée ↓ — RGB [0,1]', ['train_eval/smoke_mse', 'validation/smoke_mse']),
        ('MSE fond ↓ — RGB [0,1]', ['train_eval/background_mse', 'validation/background_mse']),
        ('Écart validation − apprentissage', ['generalization/lpips_gap']),
    ]
    if method == 'vqgan':
        quality.extend([
            ('Perplexité du codebook', ['validation/codebook/perplexity', 'validation/smoke_codebook/perplexity']),
            ('Fraction des codes utilisés', ['validation/codebook/used_fraction', 'validation/smoke_codebook/used_fraction']),
        ])
    quality.append(('Patience : validations sans progrès / seuil', ['early_stopping/bad_checks', 'early_stopping/patience']))
    optimization = [('Pertes de reconstruction', ['train/reconstruction', 'train/perceptual']),
                    ('Quantification et discriminateur', ['train/vq', 'train/discriminator'])] if method == 'vqgan' else [
                    ('Objectif du prédicteur', ['train/loss'])]
    result = [dict(name='Comprendre cet entraînement', open=True, markdown=[reading_guide(method)]),
              dict(name='Qualité et généralisation', open=True, charts=quality),
              dict(name='Exemples fixes', open=True, media=['validation/errors_and_masks']),
              dict(name='Analyse par date et vue', open=False, regex=[
                  ('LPIPS par journée', r'^validation/by_date/.*/lpips$'),
                  ('LPIPS par caméra et vue', r'^validation/by_view_id/.*/lpips$')]),
              dict(name='Optimisation et ressources', open=False, charts=optimization + [
                  ('Norme des gradients', ['train/gradient_norm_before_clip']),
                  ('Temps par mise à jour', ['performance/update_seconds']),
                  ('Mémoire GPU en octets', ['gpu/allocated_bytes', 'gpu/reserved_bytes']),
              ]),
              dict(name='Sélection et référence', open=False, charts=[
                  ('Validation et référence simple', ['validation/lpips', 'validation/baseline_lpips']),
                  ('Étape du meilleur checkpoint', ['selection/best_step']),
                  ('Signal indicatif de surapprentissage', ['generalization/possible_overfit']),
              ])]
    return result


def build_sections(plan, ws, wr, *, serialize=True) -> list:
    sections = []
    for entry in plan:
        panels = [wr.LinePlot(title=title, x='global_step', y=keys,
                             title_x='Mise à jour', smoothing_type='none',
                             point_visualization_method='bucketing-gorilla')
                  for title, keys in entry.get('charts', [])]
        panels += [wr.LinePlot(title=title, x='global_step', metric_regex=pattern,
                              title_x='Mise à jour', smoothing_type='none')
                   for title, pattern in entry.get('regex', [])]
        panels += [wr.MediaBrowser(title='RGB réel | reconstruction/prédiction | erreur | masque estimé',
                                   media_keys=[key], num_columns=1)
                   for key in entry.get('media', [])]
        panels += [wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=14))
                   for markdown in entry.get('markdown', [])]
        section = ws.Section(name=entry['name'], panels=panels, is_open=entry['open'],
                             layout_settings=ws.SectionLayoutSettings(columns=2, rows=3))
        # The pinned SDK serializes panels to the app's wire representation.
        sections.append(section._to_model().model_dump(by_alias=True, exclude_none=True) if serialize else section)
    return sections


def curate_spec(original: dict, sections: list[dict]) -> dict:
    """Change only presentation; retain run filters, colors and selections."""
    result = copy.deepcopy(original)
    section = result['section']
    bank = section['panelBankConfig']
    bank['sections'] = copy.deepcopy(sections)
    bank['panelConfigOverrides'] = {}
    bank['panelPlacementOverrides'] = {}
    section.setdefault('workspaceSettings', {})['shouldAutoGeneratePanels'] = False
    section.setdefault('settings', {})['shouldAutoGeneratePanels'] = False
    return result


VIEWS_QUERY = '''query($entity:String,$project:String,$username:String){
  project(name:$project,entityName:$entity){
    allViews(viewType:"project-view",userName:$username){edges{node{
      id name type displayName spec updatedAt user{username}
    }}}
  }
}'''

SAVE_MUTATION = '''mutation($input:UpsertViewInput!){
  upsertView(input:$input){view{id name spec} inserted}
}'''


def load_sdk():
    try:
        import wandb
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces._graphql import execute_graphql
    except ImportError as exc:
        raise RuntimeError('Install tools/requirements_dashboard.txt in a separate environment') from exc

    return wandb.Api(timeout=30), ws, wr, execute_graphql


def configure(identity, method, directory, *, apply=False):
    api, ws, wr, execute_graphql = load_sdk()
    username = execute_graphql(api, '{viewer{username}}')['viewer']['username']
    variables = dict(entity=identity['entity'], project=identity['project'], username=username)
    response = execute_graphql(api, VIEWS_QUERY, variables)
    project = response.get('project')
    if project is None:
        raise ValueError('W&B project not found or inaccessible')
    views = [entry['node'] for entry in project['allViews']['edges']]
    if any(view['user']['username'] != username for view in views):
        raise ValueError('Expected only workspaces owned by the current user')
    title = f'Suivi scientifique - {method.upper()}'
    views = [view for view in views if view['displayName'] == title]
    if len(views) > 1:
        raise ValueError('Duplicate named workspaces; no workspace was modified')
    view = views[0] if views else None
    plan = section_plan(method)
    result = dict(applied=False, entity=identity['entity'], project=identity['project'],
                  view_id=view['id'] if view else None, view_name=view['name'] if view else None, plan=plan)
    if not apply:
        return result

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = directory / f'workspace_before_{stamp}_{uuid4().hex[:8]}.json'
    with backup.open('x') as stream:
        json.dump(dict(entity=identity['entity'], project=identity['project'], view=view), stream, indent=2)
    if view is None:
        workspace = ws.Workspace(entity=identity['entity'], project=identity['project'], name=title,
            sections=build_sections(plan, ws, wr, serialize=False), auto_generate_panels=False,
            runset_settings=ws.RunsetSettings(filters=f"Config('method') == '{method}'"))
        workspace.save()
        view_id = workspace._internal_id
        view_name = workspace._internal_name
        url = workspace.url
    else:
        updated_spec = curate_spec(json.loads(view['spec']), build_sections(plan, ws, wr))
        payload = dict(id=view['id'], entityName=identity['entity'], projectName=identity['project'],
                       type=view['type'], name=view['name'], displayName=view['displayName'],
                       lastUpdatedAt=view['updatedAt'], spec=json.dumps(updated_spec), createdUsing='WANDB_SDK')
        saved = execute_graphql(api, SAVE_MUTATION, dict(input=payload))['upsertView']['view']
        if saved['id'] != view['id'] or json.loads(saved['spec']) != updated_spec:
            raise RuntimeError(f'W&B workspace save was not verified; original is in {backup}')
        view_id, view_name = view['id'], view['name']
        root_url = (identity.get('url') or '').split('/runs/')[0] or f"https://wandb.ai/{identity['entity']}/{identity['project']}"
        slug = view_name.removeprefix('nw-').removesuffix('-v')
        url = root_url + '?' + urlencode({'nw': slug})

    # A fresh read verifies persistence independently of the mutation response.
    reread = execute_graphql(api, VIEWS_QUERY, variables)
    actual_view = next((entry['node'] for entry in reread['project']['allViews']['edges'] if entry['node']['id'] == view_id), None)
    if actual_view is None:
        raise RuntimeError(f'Saved workspace was not found; original is in {backup}')
    actual = json.loads(actual_view['spec'])
    sections = actual['section']['panelBankConfig']['sections']
    settings = actual['section'].get('workspaceSettings') or actual['section'].get('settings', {})
    if ([section['name'] for section in sections] != [entry['name'] for entry in plan]
            or settings.get('shouldAutoGeneratePanels') is not False
            or any(len(section['panels']) != sum(len(entry.get(kind, [])) for kind in ('charts', 'regex', 'media', 'markdown'))
                   or section.get('isOpen') != entry['open'] for section, entry in zip(sections, plan))):
        raise RuntimeError(f'Saved dashboard layout was not preserved; original is in {backup}')
    result.update(applied=True, backup=str(backup), view_id=view_id, view_name=view_name, url=url, spec=actual)
    (directory / f'workspace_{method}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return {key: value for key, value in result.items() if key != 'spec'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', required=True, help='Contains wandb.json and run.json')
    parser.add_argument('--apply', action='store_true', help='Create or update the named dashboard after backing up its previous configuration')
    args = parser.parse_args(argv)
    run = Path(args.run_directory)
    identity = json.loads((run / 'wandb.json').read_text())
    method = json.loads((run / 'run.json').read_text()).get('method', 'vqgan' if run.name == 'autoencoder' else None)
    if method not in ('vqgan', 'simvp', 'flow_alpha', 'flow_uniform'):
        parser.error('Select a VQGAN or predictor run directory')
    print(json.dumps(configure(identity, method, run.parent / 'dashboard', apply=args.apply), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
