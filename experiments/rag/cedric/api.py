from rag import RAG
from typing import Dict
import requests

URL = "http://localhost:1234"
MODELS_ENDPOINT = "/v1/models"
COMPLETION_ENDPOINT = "/v1/chat/completions"


def get_available_models() -> list:
    """
    Returns the list of available model IDs from the LM server.
    Returns:
        list: A list of model IDs.
    """

    response = requests.get(f"{URL}{MODELS_ENDPOINT}")
    response.raise_for_status()
    models_info = response.json()
    return [model["id"] for model in models_info["data"]]


def prompt_str(model_id: str, text: str, context: Dict[str, str] = {}) -> str:
    """
    Sends a prompt string to the LM server and returns the response.
    Args:
        model_id (str): The model ID.
        text (str): The input text.
        context (dict): The context to include in the prompt. Map of role to content.
    Returns:
        str: The LM response string.
    """

    try:
        response = requests.post(
            f"{URL}{COMPLETION_ENDPOINT}",
            json={
                "model": model_id,
                "messages": [
                    { "role": role, "content": content.replace("\n", " ") } for role, content in context.items()
                ] + [{ "role": "user", "content": text.replace("\n", " ") }]
            }
        )
        response.raise_for_status()
        lm_result = response.json()["choices"][0]["message"]["content"].strip()
        return lm_result
    
    # Handle errors
    except Exception as e:
        return f"[Error] Could not get response from the LM server: {e}"


def main(default_texts: list = []):
    """
    Main function of the script.
    Args:
        default_texts (list): List of default texts to initialize the RAG system.
    """

    def print_commands():
        """
        Prints the available commands for the user.
        """

        print("  Available commands:\n"
            "    /help - Show this help message\n"
            "    /model - Change the current model\n"
            "    /add - Add a new text to the RAG system\n"
            "    /exit - Exit the program")

    def select_model() -> str:
        """
        Prompts the user to select a model from the available models.
        Returns:
            str: The selected model ID.
        """

        # Get and display available models
        models = get_available_models()
        print("Available models:")
        for i, model in enumerate(models):
            print(f"[{i + 1}]  {model}")

        # Prompt user to select a model
        while True:
            selected = input("Selected model: ")
            if selected in models:
                model = selected
                break
            if selected.isdigit() and 1 <= int(selected) <= len(models):
                model = models[int(selected) - 1]
                break
            # Handle invalid model selection
            print("Invalid model selection.")
        
        # Return the selected model
        return model
    
    def add_text():
        """
        Prompts the user to add a new text to the RAG system.
        """

        new_text = input("")
        if new_text.strip():
            rag.add_text(new_text)
            print("Text added to the RAG system.")
        else:
            print("Empty text. Nothing was added.")

    # Initialize values
    model = select_model()
    rag = RAG(default_texts)

    while True:
        print("\n------------------------\n")
        text = input(f"[{model}]  ")

        # Skip empty input
        if not text.strip():
            continue

        # Detect commands (starting with '/')
        if text.startswith("/"):

            # Help command
            if text.lower() == "/help":
                print_commands()
            # Model change command
            elif text.lower() == "/model":
                model = select_model()
            # Add text command
            elif text.lower() == "/add":
                add_text()
            # Exit command
            elif text.lower() == "/exit":
                break
            # Unknown command
            else:
                print("  Unknown command. Type /help for a list of commands.")

        # Else, it's a prompt
        else:
            response = prompt_str(model, text, rag.get_context(text)).replace("\n", "\n  ")
            print(response)


if __name__ == "__main__":
    main([
        # Semaine 1
        """1) Lundi :
a. Matin :
Présentation du département informatique et de ces différentes branches (création de contenu
pédagogique, back-end, front-end).
Configuration en autonomie de mon poste de travail (Wampserver, Symfony, VsCode) et découverte en
autonomie du langage PHP et du framework Symfony à l’aide de tuto sur internet.
Création d’un projet bateau visant à
b. Après-midi :
Découverte et apprentissage en autonomie du langage PHP et du framework Symfony à l’aide de tuto sur
internet.
Création d’un projet bateau visant qui consiste en une application web fonctionnant avec PHP qui interagir
avec une base de données MySQL pour avoir accès à diverses opérations (CRUD) sur la table des
utilisateurs.
2) Mardi :
a. Matin :
Présentation du projet que je devrais réaliser pendant ce stage.
L’entreprise utilisant l’API de ChatGPT dans ces divers modules de formations par exemple pour
apprendre l’anglais en dialoguant avec une IA. Elle s’est vite rendu compte de l’instabilité des réponses de
l’IA en raison de mise à jour de ChatGPT. Ainsi il souhaiterait conserver dans une base de données les
différentes réponses aux files du temps de l’IA pour un même prompt afin de voir à partir de quel moment
l’IA à commencer à dire n’importe quoi. Et ainsi pouvoir apporter des corrections au prompts qu’ils
envoient à l’API de ChatGPT.
Mon travail consistera donc à crée un serveur avec PHP, Symfony et MySQL pour lancer
automatiquement les prompts à des moments donnés de la semaine ou de la journée afin de sauvegarder
les réponses de l’IA dans une base de données pour pouvoir ensuite y accéder pour voir les réponses.
b. Après-midi :
Mise en place du projet et création de la BDD, du MCD et des premiers templates du site.
3) Mercredi :
a. Matin :
Créations des templates et des requêtes pour récupérer les informations en BDD.
b. Après-midi :
Présentation à Cédric Anus de l’état du projet et des visuels déjà présents. Refonte de la BDD avec l’arrivé
de nouvelles informations à prendre en compte comme par exemple le fait de devoir interagir avec une
API qui évolue très rapidement et donc qui rend une BDD trop spécifique, trop vite obsolète.
Refonte du site afin de prendre en compte les changements dans la BDD et les changements d’interfaces.
4) Jeudi :
J’ai fini de faire les changements liés à la modification de la BDD, j’ai agrandi la BDD pour rajouter une
table qui me permette de gérer les changements dans les réponses de chatgpt. J’ai fini de faire les vues il
ne me reste plus qu’à perfectionner les boutons qui supprimer des outils et de langues afin de faire un
delete de tous ce qui est lié.
5) Vendredi :
Finalisation du visuel de l’application web et ajout d’un filtre pour filtrer les logs. Echange avec Cédric et
présentation de mon travail. Il fallait que je rajoute un filtre et plus tard je devrais rajouter des requêtes
asyncrone pour éviter de recharger la page inutilement.
 Pré séntation dés diffé rénts mé tiérs au séins dé l’é quipé dé dé véloppémént dé l’éntréprisé ainsi qué lés
ta chés qu’ils accomplissént.
 Configuration én autonomié du posté dé travail avéc Wampsérvér, Symfony ét VsCodé. Appréntissagé én
autonomié du langagé PHP ét du framéwork Symfony via dés tutoriéls én ligné.
 Misé én placé d’un projét dé basé afin dé dé véloppér més compé téncés én PHP/Symfony.
 Pré séntation d’uné partié du projét dé stagé : cré ation d'un sérvéur avéc PHP, Symfony ét MySQL pour
sauvégardér lés ré ponsés dé l'IA dans uné basé dé donné és ét lés visualisér.
 Misé én placé du projét, cré ation du modé lé concéptuél dés donné és (MCD), dé la basé dé donné és ét
dés prémiérs témplatés du sité.
 Cré ation dés témplatés ét dés réqué tés pour ré cupé rér lés informations dé la basé dé donné és.
 Pré séntation dé l'é tat du projét ét dés visuéls éxistants a Cé dric. Réfonté dé la basé dé donné és pour
préndré én compté dé nouvéllés informations, notammént la géstion dés langués ét dés nivéaux pour
chaqué outil. Cétté modification a impliqué uné ré é valuation dé la structuré dé la basé dé donné és afin
dé garantir uné géstion éfficacé dés rélations éntré lés outils, lés langués ét lés nivéaux.
 Finalisation dés modifications dé la basé dé donné és. Ajout d'uné tablé pour gé rér lés changéménts
dans lés ré ponsés dé l'IA (changémént dé nom dé variablés, changémént d’émplacémént dans la
ré ponsé, ...).
 Aché vémént dés témplatés. Finalisation dés boutons dé suppréssion d'outils ét dé langués.
 É changé avéc Cé dric ét pré séntation du travail éfféctué . Discussion sur l'ajout d'un filtré supplé méntairé
pour é vitér dé réchargér inutilémént la pagé.
 Finalisation du désign dé l'application wéb. Ajout d'un filtré pour filtrér lés logs.
 Pré paration pour la phasé suivanté du projét du stagé : planification dé l'é crituré du script a lancér par
un CRON pour ré cupé rér lés prompts a téstér via l'API dé dé véloppémént dé l'éntréprisé.
\"Concéption ét dé véloppémént d'uné platéformé dé suivi dés é volutions dés modé lés dés intélligéncés
artificiéllés (IA), én particuliér ChatGPT, pour amé liorér én continu lés modulés dé formation én ligné. Cétté
platéformé énglobéra la misé én placé d'un sérvéur PHP avéc Symfony ét MySQL pour archivér lés ré ponsés dé
l'IA, ainsi qué lé dé véloppémént d'un script automatisé éxé cuté via CRON. Cé script intérrogéra la basé dé
donné és pour idéntifiér lés outils a téstér, lés langués ét lés nivéaux, puis utiliséra l’API dé l'éntréprisé pour
obténir lés prompt a téstér ét lés énvoyér a l’API dé ChatGPT pour obténir ét énrégistrér lés ré ponsés. L'objéctif
ést dé fournir un systé mé complét pérméttant d'analysér ét dé suivré l'é volution dés intéractions afin d'assurér
la qualité ét la pértinéncé dés formations én ligné.\"""",
        
        # Semaine 2
        """1) Lundi :
Création du script php pour récupérer des données du serveur de l’entreprise et de ChatGPT
et de les enregistrer dans ma BDD.
Problème rencontré : Mauvaise version de PHP installé
Création d’une commande qui permettrait de récupérer tous les logs
Modification de la commande pour qu’elle ait un system de fonctionnement similaire à
CRON mais sans avoir besoin de CRON.
Modification de la BDD pour que l’on puisse enregistrer et modifier les informations sur le
‘CRON’ que j’ai fait.
2) Mardi :
Tentative de ‘’transposer’’ le system de gestion des tâches de CRON avec mes proposes
fonctions mais échec car trop complexe et long à mettre en place.
Utilisation d’une librairie qui le fait. Modification de la BDD pour pouvoir changer les dates
d’exécutions du CRON sans avoir besoin de le changer en ligne de commande.
Ajout d’une table dans la BDD afin de pouvoir voir les erreurs qui sont survenue lors des
requêtes vers le LMS ou le serveur de développement de l’entreprise afin d’afficher un
historique sur le site pour faciliter la compréhension des erreurs.
3) Jeudi :
Ajouts de code AJAX pour des actions asyncrone de suppression et possibilité d’ajout d’outils
et de langue via un formulairePOST.
4) Vendredi :
Fin des modifications des fonctions non asyncrone en AJAX. Et ajout de message de
confirmations de suppriession, ...
Connection au GitLab de l’entreprise
Modification du CRON pour avoir plusieurs CRON lancer depuis la VM qui lance une
commande et non plus la commande qui lance un faux CRON.
Création machine virtuelle.
Discussion avec Cédric sur les autres points que je pourrais améliorer.
Besoin d’ajouter la possibilité d’ajouter des niveaux, de les supprimer et de les modifiers.
Ajout de bouton pour supprimer les logs avec des case à cocher et changement de
l’emplacement des erreurs de récupérations des logs dans les logs avec un message pour dire
qu’il y a eu une erreur sur le titre du log ou un truc du genre.
 Développement d'un script pour récupérer les prompts à tester depuis les serveurs de
l’entreprise, les soumettre à l’API de ChatGPT et enregistrer les réponses. Mise à jour
de la version de PHP pour assurer la compatibilité.
 Adaptation de la base de données pour stocker des informations sur les CRON
exécutés.
 Tentative de transposer la logique de CRON sur le script mais finalement abandonner
pour l’utilisation d’une librairie qui est moins difficile à utiliser.
 Ajout d’une table dans la base de données pour sauvegarder les éventuelles erreurs lors
des appelle à l’API de l’entreprise ou de ChatGPT. Après une rapide présentation de
l’avancement du projet à Cédric, cette table sera finalement supprimée et les
informations sur les erreurs seront stocké dans les logs afin de facilité la visualisation
et la recherche d’erreurs.
 Refonte des requêtes serveur pour utiliser AJAX côté client, minimisant le
rechargement de la page pour des actions telles que la suppression de logs, langues ou
outils.
 Intégration de boîtes d'informations pour notifier visuellement l'utilisateur du succès ou
de l'échec des opérations.
 Connection au GitLab de l’entreprise afin d’un déposer le projet.
 Présentation du projet à Cédric, discussion sur les étapes suivantes incluant la mise en
place d'une machine virtuelle pour le serveur web et les CRON, afin de gérer l'insertion
de nouveaux logs dans la base de données.""",
        
        # Semaine 3
        """1) Lundi :
J’ai continué de travailler sur le filtrage des logs pour que ce soit fonctionnel. Ainsi
que la suppression quand un filtre et activé.
Mise à jour du script PHP pour récupérer les prompt depuis l’API de l’entreprise.
Mise à jour de ma BDD pour pouvoir enregistrer l’alias qui est utiliser dans l’API de
l’entreprise.
Ajout d’une catégorie dans les paramètres pour pouvoir ajouter, supprimer ou
modifier des levels.
Ajouts d’un système de couleur pour visualiser les levels des logs mais aussi pour
voir combien d’erreurs sont survenu dans les logs de chaque langue et chaque outil.
Dépôt du projet sur le GitLab de l’entreprise (mieux vaut tard que jamais…).
Début de configuration de la machine virtuelle qui sera chargé de faire fonctionner le
site web (installation d’Apache2, MariaDB, MySQL et PHP).
Début de l’ajout des noms des outils à tester dans la BDD afin d’avoir un script
d’initialisation de la BDD propre et qui ressemble au donnée qui seront manipuler
par le site.
 Finalisation du développement du filtrage des logs pour assurer leur
fonctionnalité, ainsi que la suppression des logs en fonction des filtres actifs.
 Achèvement de l'écriture du script PHP récupérant les prompts à tester depuis
l’API de l’entreprise. Adaptation de la base de données pour enregistrer les
alias des langues.
 Complétion de la page des paramètres pour inclure une section permettant de
visualiser, modifier, supprimer et ajouter des niveaux. Intégration d'un système
de coloration pour différencier les niveaux dans l'historique des logs.
 Présentation des fonctionnalités et du visuel du site à Cédric, il m’a donné
l’accès au GitLab de l’entreprise pour y déposer mon projet et m’as suggéré
quelques améliorations dans le visuel et les fonctionnalités du site, telles que
l'optimisation de la fenêtre de confirmation.
 Configuration de la machine virtuelle (VM) Debian pour l'hébergement du
serveur PHP. Installation d'Apache2, MariaDB, MySQL, PHP et Composer. J’ai
rencontré un problème lors de la configuration, résolu en réalisant des
ajustements spécifiques au système Debian pour les requêtes vers la base de
données et les fichiers de configuration du serveur.""",
        
        # Semaine 4
        """Lundi :
J’ai essayé de configurer la machine virtuelle sans interface graphique mais sans
succès à cause de problèmes de réseaux sur mon poste de travail.
J’ai fini de modifier la fonction qui permet de trier les levels quand on en ajoute un
pour qu’ils soient par défaut trier dans l’ordre croissant de level.
J’ai modifié le code pour l’ajout d’un CRON pour qu’il y ait une route dans le
controller pour vérifier si une expression est juste ou pas. Et si la réponse et juste
alors on appelle une autre route pour enregistrer le CRON.
Il faudra aussi que j’ajoute une fonction qui vérifie que lorsque je lance le CRON ce
dernier soit bien lancé (systemctl start cron / restart)
J’ai ajouté un bouton et une page pour que lorsqu’on sélectionne 2 logs (pas moins
pas plus) on puisse les comparer dans une autre page.
J’ai ajouté un bouton dans la barre de navigation du site pour télécharger les logs qui
ont été cocher. Et j’ai commencé de faire le côté serveur pour récupérer un fichier et
l’envoyer au client.
Demain je vais finir d’écrire le code pour pouvoir crée mes différents fichiers et
ensuite les envoyer au client.
Mardi :
J’ai fini d’ajouter la fonctionnalité de téléchargement et j’ai même ajouter une fonction
pour supprimer les anciens téléchargements au bout d’un certain temps.
J’ai fait un point avec Cédric sur les améliorations à faire. Il faut que j’améliore le
visuel du site pour que ce soit un visuel plus moderne notamment et je dois faire une
interface plus facile à utiliser notamment en mettant une liste pour afficher les logs
d’une langue.
J’ai configuré le CRON pour s’exécuter tous les jours à 9h et à 16h.
J’ai modifié tous les boutons pour utiliser des icones afin de rendre la lecture de la
page plus rapide et j’ai commencer de modifier la page des logs afin de la rendre
plus lisible et d’améliorer les filtres.
Mercredi :
J’ai fini de modifier l’affichage des outils, des langues et des logs afin de voir
combien il y a eu d’erreur au cours de dernière 24h.
J’ai fini de faire le tableau pour visualiser les logs d’une langue et j’ai ajouter des
fonctions qui permettent de trier les logs par date ou niveau croissant et ce
directement dans le tableau via des petits icône à cliquer et j’ai perfectionner le filtre
afin de permettre l’affiche d’uniquement des erreurs.
J’ai commencé de rendre le visuel plus moderne et plus jolie notamment en faisant
en sorte que les card qui représente les différents outils soit toutes de la même taille
et j’ai rajouter un fil d’ariane en haut de la page se repérer plus facilement dans
l’arborescence du site.
J’ai commencé de faire une page pour les statistiques et j’ai fait le graphique qui
permet de voir le temps de réponse de l’API de ChatGPT en fonction de l’heure de la
journée.
Jeudi :
J’ai fait un point avec Cédric le matin pour lui montrer les avancer et il m’a proposé
plusieurs modifications comme le fait que les niveaux ne soient pas les même pour
tous les outils et qu’on puisse les customiser. Et je lui ai montrer la page statistique
que j’ai commencé de faire et je lui ai montrer les graphiques que je comptais
intégrer afin d’avoir un retour et de savoir s’il y avait des statistiques qu’il souhaitait
voir en plus. Il m’a aussi demandé de faire une détection plus poussée des erreurs
afin de pouvoir savoir s’il s’agissait d’une erreur de leur API ou de celle d’OpenAI
pour ensuite pouvoir améliorer les statistiques. Il a aussi
J’ai refait la BDD pour que les niveaux ne soient pas les mêmes pour tous les outils
mais que les niveaux dépendent d’un outil puisque certains outils n’ont qu’un seul
niveau par défaut.
J’ai ensuite modifié tous mon code afin qu’ils soient compatibles avec mes
modifications et que je puisse ajouter ou supprimer des niveaux directement depuis
les outils.
J’ai modifié la commande PHP de récupération des logs afin d’avoir la gestion
avancée des erreurs.
J’ai continué de travailler sur la page des statistiques.
Vendredi :
J’ai continué de travailler sur la pages des statistiques et j’ai rencontrer des
problèmes avec D3.js parce qu’il s’agissait de la première fois que je l’utilisait et
qu’en plus de ça le code que j’avais trouvé utilisais des fonctions qui était déprécié et
dont je ne trouvais pas l’équivalent.
 J'ai tenté de configurer la machine virtuelle sans interface graphique, mais des
problèmes de réseau sur mon poste de travail m'ont empêché de réussir.
 J'ai modifié la fonction de tri des niveaux pour qu'ils soient par défaut triés
dans l'ordre croissant. J'ai ajouté une route dans le contrôleur pour vérifier si
une expression CRON est valide et, si c'est le cas, enregistrer le CRON.
 J'ai implémenté une vérification du lancement correct des CRON avec
systemctl start cron et systemctl restart cron. J'ai ajouté une
fonctionnalité permettant de comparer deux logs sélectionnés et un bouton
pour télécharger les logs cochés, en commençant le développement côté
serveur.
 J'ai finalisé la fonctionnalité de téléchargement de logs et ajouté une fonction
de suppression automatique des anciens téléchargements. Après un point
avec Cédric, j'ai modernisé le visuel du site et facilité l'utilisation de l'interface,
notamment avec une liste pour afficher les logs d’une langue.
 J'ai configuré le CRON pour s'exécuter à des heures fixes et mis à jour tous
les boutons avec des icônes pour améliorer la lisibilité. J'ai également modifié
la page des logs pour la rendre plus lisible et optimiser les filtres.
 J'ai modifié l'affichage des outils, des langues et des logs pour visualiser les
erreurs des dernières 24 heures. J'ai terminé le tableau de visualisation des
logs d'une langue, avec des fonctions de tri et des filtres pour afficher
uniquement les erreurs.
 J'ai modernisé le visuel du site en harmonisant la taille des cartes des outils et
ajouté un fil d'Ariane pour faciliter la navigation. J'ai commencé une page de
statistiques avec un graphique du temps de réponse de l'API de ChatGPT.
 Après un point avec Cédric, j'ai ajusté la base de données et le code pour
permettre la customisation des niveaux par outil et une meilleure détection des
erreurs des API. J'ai modifié la commande PHP de récupération des logs pour
une gestion avancée des erreurs.
 J'ai continué à travailler sur la page de statistiques, rencontrant des difficultés
avec D3.js à cause de fonctions dépréciées.""",
        
        # Semaine 5
        """1) Mardi :
J’ai continué de travailler sur les statistiques et je suis parvenu à résoudre les
problèmes que j’avais eu la semaine dernière avec D3.js.
Céline (PAS SUR DU NOM) et venu me voir avec Cédric pour que je lui montre ce
que j’avais comme interface dans mon site web afin qu’elle me propose un templates
afin d’avoir un site plus uniforme et visuellement plus joli.
2) Mercredi :
J’ai continué de travailler sur la page des statistiques pour ajouter des graphes et
des filtres sur les graphes existants.
J’ai commencé de modifier le la page de visualisation des outils afin de proposer une
option qui permettent de voir pour tous les outils de la pages les langues qui sont
disponibles pour ces outils et les niveaux afin de pouvoir plus facilement les cocher
sans avoir besoin de revenir à la page principale pour changer d’outil.
3) Jeudi :
J’ai continué de travailler sur la page des statistiques en ajoutant des filtres sur les
graphes et en ajoutant un graphe pour visualiser sous la forme de cercle le nombre
de log afin de faire comme une sorte d’arborescence de fichier pour savoir qu’elles
outils, langues ou niveaux ont le plus de logs. Par exemple pour pouvoir libérer de la
place plus facilement si on souhaite avoir des statistiques toujours pertinentes en
enlevant les logs les plus anciens dans les outils, langues ou niveaux qui en ont le
plus.
J’ai fini de faire la page pour afficher toutes les langues des outils ainsi que leurs
niveaux. Il ne reste plus qu’à faire un visuel plus propre pour la page.
4) Vendredi :
Ajout d’un graphe pour afficher l’espace que prennent les différents outils, langues et
niveaux.
Ajout trie par latence dans l’affichage des logs.
Début de l’ajout de statistiques pour les langues d’un outil.
 J'ai continué de travailler sur les statistiques et résolu les problèmes
rencontrés la semaine précédente. J'ai également ajouté des filtres sur
plusieurs graphiques pour faciliter leur analyse, comme en enlevant les
données les plus anciennes.
 J'ai fait une démonstration du site à Céline pour lui montrer son
fonctionnement et son but afin qu'elle puisse ensuite proposer des idées de
templates plus modernes pour améliorer le design.
 J'ai ajouté une page permettant d'afficher toutes les langues et leurs niveaux.
Cette page facilite la visualisation des langues et des niveaux actifs, ainsi que
des erreurs rencontrées lors des dernières 24 heures.
 J'ai perfectionné la page d'affichage des logs en ajoutant une colonne pour
visualiser le temps de réponse de l'API d'OpenAI.""",

        # Semaine 6
        """1) Lundi :
J’ai continué de faire les statistiques en améliorant un peut le visuel.
J’ai fait une réunion d’une heure avec Cédric pour lui présenter l’avancement du site
et il m’a fait un retour. Il m’a fait des suggestions d’amélioration pour l’UX (interface
utilisateur) afin de la rendre plus intuitive. Mais il m’a aussi parler des assistants
virtuels qui venait d’être ajouter par ChatGPT et qu’ils sont entrain de développé de
leur côté. Il aimerait donc avoir en plus des conversations normales avec ChatGPT
avoir aussi accès aux assistants en plus des outils.
J’ai ensuite continué de travailler en faisant les améliorations et la correction de bug
sur le visuel d’un outil afin de rendre l’interface de l’utilisateur plus intuitive.
2) Mardi :
J’ai continué de modifier l’affichage de la page des logs afin d’avoir une interface qui
rende les informations essentielles rapidement. Et j’ai ajouté des textes
d’informations sur les boutons afin de savoir exactement ce qu’ils font.
J’ai rendu mes filtres dynamiques pour ne plus avoir à cliquer sur le logo pour
rechercher.
J’ai modifié le script qui permet de récupérer les logs pour intégrer une pause d’une
durée aléatoire entre chaque requête à l’API d’OpenAI. Je devais mettre en place un
script qui fasse les requêtes d’une même langue simultanément mais après avoir fait
des tests il s’est avérer que la durée de génération d’un seul log avait doublé. Bien
que cette méthode permettait d’exécuter plusieurs requêtes simultanément cela
faussait les statistiques puisse que la différence de performance de l’API d’OpenAI
était énorme.
3) Mercredi :
J’ai fini de travailler sur les améliorations demander par Cédric.
Je me suis documenté sur l’API d’OpenAI afin de savoir comment fonctionnais les
assistants. J’ai ensuite fait une ébauche de base de données pour pouvoir stocker
les informations sur ces assistants et j’ai montrer le résultat à Cédric afin de mieux
comprendre leur besoin en ce qui concerne les assistants. Ainsi j’ai modifié mon
MCD temporaire pour qu’il colle aux données dont j’avais besoin. J’ai ensuite
commencé de faire la partie du site qui permettras d’afficher ces assistants. Pour
l’instant je peux voir des assistants ainsi que les différents threads pour voir les
messages de ces threads.
4) Jeudi :
J’ai continué de travailler sur le visuel des assistants afin de pouvoir voire les
assistants, les ajouter, voir la liste de leurs threads, le filtré (les threads) et les voire
(les threads).
J’ai ajouté le script pour faire un back up des assistants via l’API d’OpenAI et j’ai
modifier la BDD pour sauvegarder ces assistants.
J’ai commencé de faire le script de récupération des threads.
5) Vendredi :
J’ai fini de faire le script de récupération des threads.
J’ai avancé dans l’interface de visualisation des assistants et sur la visualisation de
back up.
 J'ai amélioré le visuel des statistiques et ajouté des filtres dynamiques pour
une meilleure analyse.
 J'ai eu une réunion avec Cédric pour présenter l'avancement, recevoir des
suggestions pour l'UX et discuter de l'intégration des assistants virtuels de
ChatGPT.
 J'ai corrigé des bugs et amélioré l'interface utilisateur d'un outil spécifique.
 J'ai modifié l'affichage des logs pour une meilleure lisibilité et ajouté des textes
explicatifs sur les boutons.
 J'ai dynamisé les filtres pour éviter le rechargement manuel de la page.
 J'ai modifié le script de récupération des logs pour intégrer des pauses
aléatoires entre les requêtes à l'API d'OpenAI.
 Je me suis documenté sur l'API d'OpenAI et j'ai fait une ébauche de la base
de données pour stocker les informations sur les assistants virtuels.
 J'ai montré cette ébauche à Cédric et posé des questions pour mieux
comprendre leurs besoins. Nous avons fait un point sur ce que je devais faire.
 J'ai développé la partie du site pour afficher les assistants et les threads
associés.
 J'ai commencé à travailler sur la partie du site permettant de visualiser et de
faire des backups des assistants virtuels.
 J'ai ajouté un script pour sauvegarder les assistants via l'API d'OpenAI et
modifié la base de données pour stocker ces informations.
 J'ai finalisé le script de récupération des threads.
 J'ai avancé sur l'interface de visualisation des assistants virtuels et la
fonctionnalité de test des assistants.""",

        # Semaine 7
        """1) Lundi :
J’ai fini de faire la visualisation dans les backups.
J’ai ajouter la possibilité de faire des CRON en fonction de la commande qu’on
souhaite lancer.
2) Mardi :
J’ai corrigé de bug.
J’ai fait une démonstration de l’avancement du projet à Cédric.
J’ai refactorisé le code de la partie templates afin de séparer les templates des outils,
des assistants, … et j’ai fait de même avec le controller afin d’avoir un controller par
partie de mon site ce qui rend le code plus facile à retrouver et à lire.
J’ai ajouté de la documentation dans toutes les routes de mon controller pour faciliter
la reprise du projet par une autre personne.
3) Mercredi :
J’ai continué de travailler sur les statistiques. J’ai fait une refonte de la partie
templates des statistiques pour différencier les statistiques des outils de ceux des
assistants. Et j’ai ajouté des statistiques pour les assistants.
J’ai commencé de faire une partie dans les paramètres pour visualiser la taille de la
BDD et de chaque table et de permettre de l’optimiser pour gagner de la place sur le
disque dur du serveur.
4) Jeudi :
J’ai fini la partie de visualisation de la BDD.
J’ai corrigé des erreurs de visualisation mineurs et j’ai fini de crée les statistiques
pour les assistants.
5) Vendredi :
- Corrections erreurs visualisation statistiques
- Ajout documentation dans tout le projet
- Correction des fautes d'orthographes
- Début de la modification du visuel du site avec l'exemple donné par Céline.
 J’ai fini de faire la partie visualisation des backups.
 J’ai ajouté une fonctionnalité pour visualiser la taille de chaque table de ma
base de données ainsi que l’espace vide.
 J’ai ajouté la possibilité d’avoir des CRON pour chaque commande.
 J’ai fini les statistiques pour les assistants.
 J’ai fait une démonstration de l’avancement du projet à Cédric.
 J’ai ensuite fait une refonte du projet afin de séparer les différentes parties du
site dans plusieurs controllers et j’ai fait de même pour la partie templates. J’ai
également rajouté de la documentation sur toutes mes fonctions afin de
faciliter la reprise du projet par d’autres personnes.
 J’ai relu tout mon code afin d’enlever toutes les fautes d’orthographe. Enfin je
crois…
 J’ai commencé à faire une refonte du visuel en prenant comme exemple les
templates que Céline (une graphiste) a faits pour le site.""",

        # Semaine 8
        """1) Lundi :
J’ai continué de travailler sur le visuel du site.
J’ai montré ce que j’avais fait à Cédric. Il m’a donné plusieurs points à améliorer en
termes de visuel.
2) Mardi :
J’ai fini de faire les modifications visuelles demandé par Cédric.
3) Mercredi :
J’ai montré à Cédric les modifications visuelles et il m’a donné plusieurs points à
améliorer. Il m’a demandé aussi de faire une fonctionnalité de login.
J’ai fini les améliorations visuelles et je suis entrain de commencer la fonctionnalité
de login.
4) Jeudi :
J’ai continué de faire la fonctionnalité de login.
5) Vendredi :
 J’ai continué de travailler sur le visuel du site.
 J’ai montré à Cédric les modifications que j’avais faite. Il m’a ensuite de
demander de rajouter une fonctionnalité de login et m’as fait des suggestions
pour améliorer le visuel.""",

        # Semaine 9
        """1) Lundi :
J’ai fini de faire l’ajout de la fonctionnalité de login et je l’ai montré à Cédric ainsi que les
modifications du visuel.
Il m’a donné d’autres modifications à faire et m’a demandé de faire la fonctionnalité de login
autrement. Parce que j’utilisais une méthode « fait maisons » qui n’est pas forcément des plus
sécurisé. Il m’a donc conseillé d’utiliser une librairie qui permet d’intégrer facilement une
fonctionnalité de login.
2) Mardi :
Fin de l’implémentation de la fonctionnalité de login et CRUD sur les utilisateurs.
Début de modifications visuels sur le site.
 J’ai fini de faire les améliorations visuelles ainsi que la fonctionnalité de login. Après
l’avoir présenté à Cédric, il m’a dit d’utiliser la bibliothèque intégrer à Symfony pour
gérer les connexions au lieu de faire un fonction login à la main.
 Cédric m’a aussi donné plusieurs autres améliorations visuelles à faire.
 J’ai fini d’implémenter la fonctionnalité de login en utilisant la bibliothèque intégrée à
Symfony.
 Le dernier jour j’ai fait des tests de tous le site afin de trouver des erreurs.""",
    ])
