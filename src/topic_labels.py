"""Nommage des themes : transforme un cluster en une formule francaise lisible.

Un appel au modele par theme NOUVEAU, pas par article. Le detail du procede et
les mesures qui ont fixe le modele et la consigne sont dans les commentaires de
_generate_readable_label.
"""
import re

# --- Label lisible par cluster (un appel Mistral par cluster, pas par --
# article : ~50-60 appels/run plutot que ~1000, cout negligeable). Les mots
# c-TF-IDF ("всу / нпз / бпла") restent dans top_words pour le detail ; le
# label affiche en priorite devient une phrase courte lisible.
_LABEL_SYSTEM_PROMPT = """Vous nommez, en francais, un groupe d'articles de presse russophone portant sur la Russie.

Ce groupe a ete constitue automatiquement : ses articles partagent un
vocabulaire recurrent, qui vous est donne. Votre travail est de dire a quoi ce
vocabulaire renvoie concretement, en vous appuyant sur les titres et les
extraits pour comprendre de quel evenement il s'agit.

Procedez en deux temps.

1. « sujet » : en une phrase francaise complete, dites ce qui se passe dans ces
   documents, qui fait quoi et ou, et ce qui en fait un seul dossier. Cette
   phrase sert a verifier que vous avez compris ; elle ne sera pas affichee.

2. « label » : a partir de cette phrase, ecrivez le nom du groupe. Il doit se
   lire comme un intitule qu'un lecteur francais comprend seul, sans avoir vu
   les articles.

Exemples de bons labels :
   « Drones ukrainiens sur le port d'Oust-Louga »
   « Poutine autorise des reservistes inaptes contre les drones »
   « Detournement de carburant au ministere de l'Interieur »
   « Reinhumation du nationaliste ukrainien Konovalets »

Exemples de mauvais labels, a ne jamais produire :
   « Attaques drones VSU port Ust-Luga Leningradskaya Oblast »  termes empiles
   « Initiatives Poutine drones categorie D »                   incomprehensible
   « Debats elections 2024 Russie Unie ridicule »               jugement porte
   « Actualites internationales »                               vide

Regles :
- une formule nominale francaise de 4 a 8 mots, avec ses articles et ses
  prepositions (de, sur, a, contre) pour qu'elle se lise comme du francais
- francais correct et accentue, jamais de caractere cyrillique. TRADUIRE les
  mots communs (хищение топлива -> detournement de carburant, мост -> pont) et
  ne transcrire que les noms propres (Туманная -> Toumannaia)
- aucun terme empile sans lien grammatical, aucun tiret pour coller des mots.
  Les traits d'union sont reserves aux noms propres composes (Saxe-Anhalt)
- n'inventer aucune date, aucun chiffre, aucun sigle, aucun fait absent des
  documents. Dans le doute, omettre l'element plutot que de le deviner
- aucun jugement de valeur : nommer le sujet, pas ce qu'on en pense
- le groupe a TOUJOURS un fil directeur. Certains extraits peuvent parler
  d'autre chose : revues de presse et resumes quotidiens melangent des sujets
  sans rapport. Les ignorer et nommer le fil que designent les mots-cles
- ne JAMAIS repondre que le groupe est divers, varie, general, heterogene ou
  sans lien commun, ni le nommer « Actualites ». Ce sont des non-reponses

Repondez en JSON : {"sujet": "...", "label": "..."}"""


# Extraits de contenu joints aux titres. Les titres de Telegram et des
# transcriptions sont souvent absents ou inutilisables (« Обзорная сводка ») ;
# quelques lignes du texte disent alors ce dont le groupe parle vraiment.
_CYRILLIQUE = re.compile(r"[Ѐ-ӿ]")

# Transcription mecanique du cyrillique, en dernier recours. Le modele rend
# souvent un bon libelle francais dont un seul nom propre reste en russe
# (« Vostok Oyl » ecrit Восток Ойл). Rejeter tout le libelle pour cela
# renvoyait au repli, c'est-a-dire aux mots-cles russes -- pire que le defaut
# qu'on voulait corriger. On transcrit donc le fragment fautif.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "io",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "ou",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "tch", "ш": "ch", "щ": "chtch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "iou", "я": "ia",
}


def _transcrire(texte):
    """Passe le cyrillique restant en alphabet latin, lettre a lettre."""
    sortie = []
    for c in texte:
        bas = c.lower()
        if bas in _TRANSLIT:
            lat = _TRANSLIT[bas]
            sortie.append(lat.capitalize() if c.isupper() and lat else lat)
        else:
            sortie.append(c)
    return "".join(sortie)
# Prefixes qui forment un mot compose francais et non une phrase collee :
# « pro-ukrainien » et « anti-missile » doivent garder leur tiret, sans quoi
# on reecrit des libelles corrects.
_PREFIXES = {
    "anti", "auto", "co", "contre", "ex", "extra", "inter", "mi", "multi",
    "neo", "néo", "non", "post", "pre", "pré", "pro", "semi", "sous",
    "sur", "ultra", "vice",
    # Composes de nationalite, meme role : « russo-chinois », « franco-russe »
    "afro", "euro", "franco", "germano", "greco", "indo", "russo", "sino",
}


def _degrouper(mot):
    """Rend les espaces a une chaine de mots collee par des tirets."""
    bouts = mot.split("-")
    if len(bouts) < 2:
        return mot
    if len(bouts) == 2 and bouts[0].lower() in _PREFIXES:
        return mot
    if len(bouts) > 3 or any(b[:1].islower() for b in bouts[1:] if b):
        return " ".join(b for b in bouts if b)
    return mot


# Lettres accentuees admises en francais. Le modele produit parfois une marque
# combinante aberrante (« Gala̧tasaray », cedille posee sur un a) : elle passe
# les controles puisqu'elle est en alphabet latin, mais s'affiche comme une
# coquille dans le tableau de bord.
_ACCENTS_FR = set("àâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ")


def _nettoyer_diacritiques(texte):
    """Retire les marques combinantes qui ne forment pas une lettre francaise."""
    import unicodedata
    decompose = unicodedata.normalize("NFD", texte)
    sortie, base = [], ""
    for c in decompose:
        if unicodedata.combining(c):
            if base and unicodedata.normalize("NFC", base + c) in _ACCENTS_FR:
                sortie.append(c)
            continue
        base = c
        sortie.append(c)
    return unicodedata.normalize("NFC", "".join(sortie))


# Longueur visee. Au-dela, on redemande une formule plus courte.
_LABEL_MAX_MOTS = 9


def _sans_markdown(texte):
    """Retire les marques Markdown, ou qu'elles soient dans la chaine.

    Le modele met en valeur les titres d'oeuvres et les noms de navires
    (« le *Professeur Molchanov* »). Un strip des extremites laissait passer
    ces marques au milieu du libelle.
    """
    for marque in ("**", "*", "__", "`"):
        texte = texte.replace(marque, "")
    return " ".join(texte.split()).strip("_ ").strip()


def _resserrer(message, label):
    """Redemande une formule plus courte, sans changer de sujet."""
    from src.llm_mistral import complete_json, MODEL_LARGE
    consigne = (
        message + "\n\nVotre proposition « " + label + " » fait plus de "
        + str(_LABEL_MAX_MOTS) + " mots. Reecrivez-la en " + str(_LABEL_MAX_MOTS)
        + " mots au maximum, en gardant l'element le plus distinctif (le nom "
        "propre, le lieu ou l'affaire) et en supprimant les enumerations et "
        "les qualificatifs. Elle doit rester une formule francaise lisible.")
    court = _extraire_label(complete_json(_LABEL_SYSTEM_PROMPT, consigne,
                                          model=MODEL_LARGE, max_tokens=300))
    if not court or _CYRILLIQUE.search(court):
        return ""
    court = " ".join(_degrouper(m) for m in court.split())
    # On ne garde le resserrage que s'il raccourcit vraiment.
    return court if len(court.split()) < len(label.split()) else ""


def _extraire_label(data):
    """Sort le libelle de la reponse, quelle que soit sa forme.

    Le modele respecte le schema demande la plupart du temps, mais rend
    parfois {"label": {"label": "..."}} ou une liste. Laisser passer ces cas
    faisait tomber toute la passe sur un .strip() applique a un dict.
    """
    vu = 0
    while isinstance(data, (dict, list)) and vu < 4:
        if isinstance(data, list):
            data = data[0] if data else ""
        else:
            data = data.get("label", data.get("sujet", ""))
        vu += 1
    return data.strip().strip(".") if isinstance(data, str) else ""


def _repartir(elements, n):
    """Prend n elements etales sur toute la liste, pas les n premiers.

    Les membres d'un theme arrivent ordonnes par probabilite d'appartenance,
    et les premiers se ressemblent souvent enormement : une seule emission
    decoupee en soixante segments occupe tout le haut du classement. Prendre
    les n premiers donne alors un echantillon qui ne represente qu'un coin du
    theme, et un libelle qui ne vaut que pour ce coin.
    """
    elements = [e for e in elements if e]
    if len(elements) <= n:
        return elements
    pas = len(elements) / n
    return [elements[int(i * pas)] for i in range(n)]


# Mots-outils sur lesquels un libelle tronque ne doit pas s'arreter.
_MOTS_OUTILS = {"de", "du", "des", "la", "le", "les", "et", "a", "au", "aux",
                "en", "sur", "pour", "dans", "par", "avec", "contre", "un",
                "une", "d", "l"}

# Volume envoye au modele pour nommer. Large a dessein : sur un theme de
# plusieurs centaines d'articles, une poignee de titres ne prouve rien et le
# nom sort faux. A ~4 500 jetons par appel on reste tres loin du debit
# autorise sur le 14b (937 500 jetons/minute).
_LABEL_N_TITRES = 120
_LABEL_N_EXTRAITS = 25
_LABEL_LONGUEUR_EXTRAIT = 500


def _generate_readable_label(keywords_str, example_titles, fallback,
                             extraits=None, n_articles=None):
    """Nomme un groupe d'articles. Renvoie `fallback` si l'appel echoue.

    Le 14b, mesure contre le 8b sur douze themes le 09/09/2026. Un precedent
    essai donnait le 8b gagnant, mais il portait sur l'ancienne consigne, qui
    demandait « 3 a 7 mots, Format Titre » a partir des mots-cles : les deux
    modeles empilaient alors des termes et le 14b y ajoutait de la
    translitteration. Avec la consigne en deux temps, le 14b rend la formule
    la plus lisible (« Transfert de Batrakov du Lokomotiv au Galatasaray » la
    ou le 8b rend « Transfert d'Alekseï Batrakov vers Galatasaray », et surtout
    « Poutine autorise des reservistes inaptes a defendre contre les drones »
    la ou le 8b melange deux sujets). Son debit, 30 requetes/minute, suffit
    largement : un appel par theme NOUVEAU, pas par article.

    L'ordre du message compte autant que la consigne. Les mots-cles viennent en
    tete parce qu'ils sont la signature statistique du groupe ; en les
    releguant a la fin, le modele se laissait entrainer par des extraits de
    revues de presse et repondait « divers sujets russes ».
    """
    from src.llm_mistral import complete_json, MODEL_LARGE

    titres = _repartir(list(example_titles or []), _LABEL_N_TITRES)
    morceaux = _repartir(list(extraits or []), _LABEL_N_EXTRAITS)

    parties = [f"Vocabulaire recurrent qui a constitue le groupe : {keywords_str}"]
    if n_articles:
        # Dire la taille du groupe et celle de l'echantillon : sans cela le
        # modele nomme ce qu'il voit comme s'il voyait tout, et un detail
        # present dans deux articles sur mille se retrouve dans le titre.
        parties.append(
            f"Le groupe compte {n_articles} articles. Les {len(titres)} titres "
            f"ci-dessous en sont un echantillon reparti sur l'ensemble du "
            f"groupe : ne retenez que ce qui y revient souvent.")
    if titres:
        bloc_titres = "\n".join(f"- {t}" for t in titres)
        parties.append(f"Titres des articles du groupe :\n{bloc_titres}")
    if morceaux:
        bloc = "\n\n".join((e or "")[:_LABEL_LONGUEUR_EXTRAIT] for e in morceaux)
        if bloc.strip():
            parties.append("Extraits, pour comprendre a quoi ce vocabulaire "
                           f"renvoie :\n{bloc}")
    # De la place pour le champ « sujet », qui n'est pas affiche mais que le
    # modele doit ecrire avant de nommer : c'est lui qui force la comprehension.
    message = "\n\n".join(parties)
    data = complete_json(_LABEL_SYSTEM_PROMPT, message,
                         model=MODEL_LARGE, max_tokens=300)
    if not data:
        return fallback
    label = _extraire_label(data)
    if not label:
        return fallback
    # Un libelle qui garde du cyrillique n'a pas rempli sa fonction : le
    # lecteur du tableau de bord ne lit pas le russe. Une seule reprise, en
    # le signalant au modele -- au-dela, on garde ce qu'il a rendu.
    if _CYRILLIQUE.search(label):
        rappel = (
            message + "\n\nVotre proposition « " + label + " » contient "
            "des caracteres cyrilliques. Reecrivez-la entierement en "
            "alphabet latin.")
        data = complete_json(_LABEL_SYSTEM_PROMPT, rappel,
                             model=MODEL_LARGE, max_tokens=300)
        relance = _extraire_label(data)
        if relance and not _CYRILLIQUE.search(relance):
            label = relance
        else:
            label = _transcrire(relance or label)
    # Le modele contourne parfois la limite de mots en collant sa phrase avec
    # des tirets (« Visite-Kiev-Kushner-cessation-bombardements »), qui compte
    # alors pour un seul mot. On degroupe avant de compter, sans casser les
    # noms propres composes : un vrai compose fait deux ou trois segments tous
    # capitalises (Saxe-Anhalt, Kim-Chen-Yn), la phrase collee en fait plus,
    # ou melange des minuscules.
    label = " ".join(_degrouper(m) for m in label.split())
    # La consigne demande 4 a 8 mots ; sur un theme touffu le modele deborde
    # quand meme (9,2 mots de moyenne mesures sur 439 themes). On lui redemande
    # une formule courte plutot que de couper : tronquer ampute la fin, qui
    # porte souvent le nom propre distinctif.
    if len(label.split()) > _LABEL_MAX_MOTS:
        court = _resserrer(message, label)
        if court:
            label = court
    label = _nettoyer_diacritiques(_sans_markdown(label))
    # Dernier filet, si le resserrage a echoue : on coupe, en retirant les
    # mots-outils que la coupe laisse en fin de formule.
    mots = label.split()
    if len(mots) > 12:
        mots = mots[:12]
        while mots and mots[-1].lower() in _MOTS_OUTILS:
            mots.pop()
        label = " ".join(mots)
    return label[:150] if label else fallback
