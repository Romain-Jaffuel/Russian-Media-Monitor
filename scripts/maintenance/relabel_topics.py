"""Renomme les thèmes dont le libellé est inutilisable.

Quand l'API Mistral ne répond pas, `_generate_readable_label` retombe sur son
repli : les trois premiers mots-clés du thème, donc du cyrillique brut
(« мвд / евротранс / топ-менеджер »). Le petit modèle, lui, contourne parfois
la limite de mots en collant sa phrase avec des tirets. Dans les deux cas les
groupes sont bons, seul leur nom manque. Ce script les reprend sans toucher au
clustering.

Il ne relance donc rien de coûteux : un appel par thème concerné, contre un
réembedding complet du corpus si l'on repassait par --reset.

Usage :
  python scripts/maintenance/relabel_topics.py            # tout ce qui est illisible
  python scripts/maintenance/relabel_topics.py --actifs   # les thèmes actifs seuls
  python scripts/maintenance/relabel_topics.py --essai    # montre sans écrire
  python scripts/maintenance/relabel_topics.py --tout     # reprend TOUS les thèmes
"""
import argparse
import re
import sys

from src.topic_labels import (_LABEL_MAX_MOTS, _degrouper,
                              _generate_readable_label, _sans_markdown)
from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("relabel")

CYRILLIQUE = re.compile(r"[Ѐ-ӿ]")
# On remonte TOUT le groupe : c'est _generate_readable_label qui y prend un
# echantillon reparti. Une limite ici ne rendrait que le haut du classement,
# ou une meme emission decoupee en segments occupe toutes les places.
LIMITE_LECTURE = 400


def _a_reprendre(label):
    """Vrai si le libellé est inutilisable tel quel.

    Sert deux fois : à choisir les thèmes à reprendre, et à refuser un
    remplacement qui ne vaudrait pas mieux que l'existant.
    """
    if not label:
        return True
    if CYRILLIQUE.search(label):
        return True
    if _sans_markdown(label) != label:
        return True
    return any(_degrouper(m) != m for m in label.split())


def _ameliorable(label):
    """Vrai si le libellé mérite une reprise, défaut franc ou simple longueur.

    La longueur ne sert qu'à sélectionner : un remplacement encore un peu long
    reste préférable à l'ancien, donc elle ne fait pas rejeter le résultat.
    """
    return _a_reprendre(label) or len(label.split()) > _LABEL_MAX_MOTS


def _contexte(conn, cle):
    """Titres et débuts d'articles du thème, pour nommer sur pièces."""
    titres = [t for (t,) in conn.execute(
        """SELECT a.title FROM article_topics at_
           JOIN articles a ON a.id = at_.article_id
           WHERE at_.topic_key = ? AND a.title IS NOT NULL AND TRIM(a.title) <> ''
           ORDER BY at_.probability DESC LIMIT ?""", [cle, LIMITE_LECTURE]).fetchall()]
    extraits = [x for (x,) in conn.execute(
        """SELECT a.content FROM article_topics at_
           JOIN articles a ON a.id = at_.article_id
           WHERE at_.topic_key = ? AND a.content IS NOT NULL
           ORDER BY at_.probability DESC LIMIT ?""", [cle, LIMITE_LECTURE]).fetchall()]
    return titres, extraits


def run(actifs_seuls=False, essai=False, tout=False):
    conn = get_conn(read_only=essai)
    tous = conn.execute(
        "SELECT topic_key, label, top_words, portee, article_count FROM topics "
        "WHERE topic_key <> -1"
        + (" AND active" if actifs_seuls else "")
        + " ORDER BY article_count DESC").fetchall()
    cibles = tous if tout else [t for t in tous if _ameliorable(t[1])]
    log.info("%d thèmes à renommer%s.", len(cibles),
             " (actifs seulement)" if actifs_seuls else "")
    if tout and not essai:
        # Le corpus entier passe au modele : quinze minutes environ a 30
        # requetes/minute. On le dit, plutot que de laisser croire a un blocage.
        log.info("Reprise complète : comptez environ %d min.", len(cibles) // 30 + 1)

    repris = echoues = 0
    for cle, ancien, mots, portee, n_articles in cibles:
        titres, extraits = _contexte(conn, cle)
        neuf = _generate_readable_label(mots, titres, fallback=ancien,
                                        extraits=extraits,
                                        n_articles=n_articles)
        if neuf == ancien or (_a_reprendre(neuf) and not tout):
            # Le repli a resservi, ou le modele a rendu un libelle aussi
            # mauvais : on garde l'ancien plutot qu'un equivalent.
            echoues += 1
            log.warning("  #%-4s inchangé : %s", cle, ancien[:52])
            continue
        repris += 1
        log.info("  #%-4s [%-8s] %s  ->  %s", cle, portee or "?",
                 ancien[:34], neuf)
        if not essai:
            conn.execute("UPDATE topics SET label = ? WHERE topic_key = ?",
                         [neuf, cle])

    log.info("Terminé : %d renommés, %d inchangés.%s", repris, echoues,
             "  (essai, rien écrit)" if essai else "")
    conn.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actifs", action="store_true",
                    help="ne reprendre que les thèmes actifs")
    ap.add_argument("--essai", action="store_true",
                    help="affiche les nouveaux libellés sans les écrire")
    ap.add_argument("--tout", action="store_true",
                    help="reprend tous les thèmes, y compris ceux qui sont "
                         "déjà lisibles")
    a = ap.parse_args()
    sys.exit(run(actifs_seuls=a.actifs, essai=a.essai, tout=a.tout))
