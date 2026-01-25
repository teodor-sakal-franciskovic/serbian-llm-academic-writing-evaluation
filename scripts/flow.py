#!/usr/bin/env python3
"""
optimized_paper_evaluator.py

Requirements:
- python-dotenv
- openai
- pymupdf4llm
- pandas

Usage examples:
    # Baseline: Rule Instruction Assessment (Standard / None)
    python flow.py --pdf-folder papers --out evaluations_none.csv --expansion none

    # Chain-of-Thought: Internal step-by-step reasoning
    python flow.py --pdf-folder papers --out evaluations_cot.csv --expansion chain_of_thought_expansion

    # ReAct: Internal Reasoning and Action cycle
    python flow.py --pdf-folder papers --out evaluations_react.csv --expansion react_expansion

    # Zero-Shot: Evaluation based only on rule names
    python flow.py --pdf-folder papers --out evaluations_zero.csv --expansion zero_shot_expansion

    # Self-Consistency: Multiple reasoning paths and consensus
    python flow.py --pdf-folder papers --out evaluations_sc.csv --expansion self_consistency_expansion

    # Self-Critique: Self-critical review and improved scoring
    python flow.py --pdf-folder papers --out evaluations_critique.csv --expansion self_critique_expansion

    # Rubric Decomposition: Decomposing the score into smaller sub-steps
    python flow.py --pdf-folder papers --out evaluations_decomposed.csv --expansion decomposed_expansion

    # Deliberative: Generating arguments for and against before the final decision
    python flow.py --pdf-folder papers --out evaluations_deliberative.csv --expansion deliberative_expansion

    # Active Prompting: Internally formulating questions to enhance context understanding
    python flow.py --pdf-folder papers --out evaluations_active.csv --expansion active_expansion

    # Dry-run example: Only parsing PDFs, no model calls
    python flow.py --pdf-folder papers --out dry_run_check.csv --expansion none --dry-run

"""

from __future__ import annotations

import argparse


# from google import genai
# from google.genai import types
# from google.oauth2 import service_account
import json
import logging

# from openai import OpenAI
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

import pandas as pd
import pymupdf4llm

from dotenv import load_dotenv
# from openai import AzureOpenAI

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------------
# Load environment and client
# -------------------------------

load_dotenv()
""" GPT-4o
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. API calls will fail if attempted.")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://galton.openai.azure.com/",
    api_key=OPENAI_API_KEY,
)
"""

"""
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "global")
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH", "google_creds.json")

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_CREDS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = genai.Client(
    vertexai=True,
    project=GOOGLE_PROJECT_ID,
    location=GOOGLE_LOCATION,
    credentials=credentials,
)
"""

"""
endpoint = "https://claude-east-us-2-resource.openai.azure.com/anthropic"
deployment_name = "claude-sonnet-4-5"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

client = AnthropicFoundry(api_key=CLAUDE_API_KEY, base_url=endpoint)
"""

endpoint = "https://gpt-east-us-2-resource.openai.azure.com/openai/v1/"
deployment_name = "gpt-5.2-chat"
GPT_5_API_KEY = os.getenv("GPT_5_API_KEY")

client = OpenAI(base_url=endpoint, api_key=GPT_5_API_KEY)

# -------------------------------
# Templates and expansions
# -------------------------------
BASE_SYSTEM_PROMPT = """
Ti si recenzent koji treba da oceni naučne radove i da da razlog iza datih ocena.

Potrebno je da odgovori budu formalni i da na osnovu pravila, koja ću ti priložiti, revidiraš uneseni tekst i daš ocenu za svako pravilo.
Potrebno je da svako pravilo oceniš sa ocenom 0, 1 ili 2.
Ocena 2 predstavlja potpuno poštovanje pravila,
ocena 1 predstavlja delimično poštovanje pravila (ekvivalent mašenja pravila dva puta, gde ga je moguće više puta omašiti),
a ocena 0 predstavlja potpuno mašenje pravila (ekvivalent mašenja pravila tri ili više puta, gde ga je moguće više puta omašiti).

{expansion}

Prvo ću ti priložiti sva pravila na osnovu kojih treba da oceniš tekst, a zatim i tekst koji treba da oceniš.

Tvoj odgovor treba da prikaže ocenu za svako pravilo.

Za svako pravilo koje je dato dole u tekstu, tvoj odgovor treba da ima sledeći šablon i ništa van njega ne treba da postoji. Znači, treba kreirati listu JSON-a koja objedinjuje sva pravila:

[
  {{
    "naziv_pravila": "<naziv_pravila>",
    "ocena": <0 | 1 | 2>
  }},
  ...
]
"""

EXPANSIONS: Dict[str, str] = {
    "none": "",
    "zero_shot_expansion": (
        "Model treba da oceni svako pravilo isključivo na osnovu njegovog naziva, bez dodatnih instrukcija. "
        "Finalni izlaz mora sadržati isključivo JSON listu konačnih ocena."
    ),
    "few_shot_expansion": (
        "The model must use the provided examples (Few-Shot) as a reference "
        "when evaluating each rule. The final output must contain exclusively "
        "a JSON list of final scores."
    ),
    "chain_of_thought_expansion": (
        "Model treba interno da **formuliše detaljno rezonovanje** korak po korak za svaku ocenu pre nego što donese finalnu odluku. "
        "Ovo unutrašnje rezonovanje NE SME biti prikazano u odgovoru. "
        "Finalni odgovor mora sadržati isključivo JSON listu konačnih ocena."
    ),
    "react_expansion": (
        "Model treba interno da koristi **strogi Reason-Act obrazac (Razmisli-Deluj)** za svako pravilo: "
        "1. **Razmisli (Reason):** Analiziraj pravilo i pronađi ključne dokaze iz teksta. "
        "2. **Deluj (Act):** Odluči da li je pravilo prekršeno na osnovu pronađenog dokaza. "
        "Ovo unutrašnje rezonovanje i akcije NE SMEJU biti prikazani. Finalni odgovor mora sadržati samo JSON listu."
    ),
    "self_consistency_expansion": (
        "Model treba interno da **generiše tri (ili više) nezavisne putanje rezonovanja** za svako pravilo. "
        "Konačna ocena za svako pravilo mora biti izabrana na osnovu **većine** (konsenzusa) ovih internih putanja. "
        "Finalni odgovor mora sadržati samo jednu konačnu ocenu za svako pravilo u JSON formatu."
    ),
    "self_critique_expansion": (
        "Model treba interno da primeni **Self-Critique (Samokritika) proces**: "
        "1. Generiši početno rezonovanje i ocenu. "
        "2. Kritički preispitaj tu početnu ocenu, tražeći potencijalne greške ili propuste. "
        "3. Na osnovu kritike, generiši konačnu, poboljšanu ocenu. "
        "Finalni izlaz mora sadržati samo konačnu ocenu za svako pravilo u JSON formatu."
    ),
    "decomposed_expansion": (
        "Model treba interno da primeni **Rubric Decomposition (Dekompozicija Rubrike)**: "
        "Podeli ocenu svakog pravila na manje, lakše proverljive podkorake. "
        "Oceni svaki podkorak pre nego što se izvede konačna, agregirana ocena za celo pravilo. "
        "Finalni izlaz mora sadržati samo konačnu ocenu za svako pravilo u JSON formatu."
    ),
    "deliberative_expansion": (
        "Model treba interno da primeni **Deliberativno Promptovanje (Deliberative Prompting)**: "
        "Generiši listu mogućih argumenata za i protiv kršenja pravila, i tek nakon te debate donesi dobro promišljenu konačnu odluku. "
        "Finalni izlaz mora sadržati samo konačnu ocenu za svako pravilo u JSON formatu."
    ),
    "active_expansion": (
        "Model treba interno da primeni **Active Prompting (Aktivno Promptovanje)**: "
        "Pre nego što oceni pravilo, model treba da formuliše jedno ili više **pitanja** koja bi poboljšala razumevanje pravila u kontekstu datog teksta, i interno odgovori na ta pitanja. "
        "Ova interna pitanja i odgovori NE SMEJU biti prikazani. Finalni izlaz mora sadržati samo konačnu ocenu za svako pravilo u JSON formatu."
    ),
}

# -------------------------------
# Rules (kept identical to your original structure)
# You can move this into a JSON/YAML file if preferred.
# -------------------------------
RULES = {
    "global": {
        "Gramatika i pravopis": {
            "instruction": "Tekst mora biti gramatički ispravan, bez pravopisnih grešaka.",
            "few-shot": "Ocena 2: 'Model je uspešno izveo klasifikaciju podataka.' | Ocena 1: 'Analiza rezultata *samo* što nije sprovedena.' (Manji kolokvijalizam) | Ocena 0: 'Rezultati su dobijeni, al je *tesko* objasniti ih, jel *nema* dovoljno podataka.' (Višestruke pravopisne i gramatičke greške)",
        },
        "Strane reči": {
            "instruction": "Strane reči treba da budu napisane kurzivom (italic). Ako je neka reč prevedena na srpski jezik, strani naziv je potrebno da se nađe u zagradi.",
            "few-shot": "Ocena 2: 'Korišćena je *Deep Learning* (duboko učenje) metoda.' (Kurziv i prevod) | Ocena 1: 'U radu smo implementirali *reinforcement learning* tehniku.' (Nedostaje kurziv) | Ocena 0: 'Koristili smo *baseline* rezultate u *machine learning* eksperimentu, što nije bilo dovoljno.' (Višestruki propusti u formatiranju)",
        },
        "Skraćenice": {
            "instruction": "Prilikom uvođenja skraćenice, mora biti naveden pun termin od kojeg je nastala. U daljem tekstu, mora se koristi skraćenica, a ne pun termin (izuzetak su naslovi poglavlja). Ne smeju biti definisane skraćenice koje kasnije nisu korišćene.",
            "few-shot": "Ocena 2: 'Korišćen je Model dubokog učenja (DL). Kasnije se koristi samo DL.' | Ocena 1: 'Definisana je skraćenica (ML), ali je tri puta kasnije korišćen pun termin *Machine Learning*.' (Nekonzistentnost) | Ocena 0: 'Definisali smo skraćenicu 'AI' u uvodu, ali ona više nigde nije upotrebljena u tekstu.' (Definisana, a nekorišćena skraćenica)",
        },
        "Argumentacija": {
            "instruction": "Sve tvrdnje moraju biti podržane citatima ili argumentovane rezultatima rada. Literatura mora biti citirana u okviru rečenice, najbliže tvrdnji koju podržava. Citati moraju biti deo rečenice.",
            "few-shot": "Ocena 2: 'Potvrđeno je da se ovim pristupom postiže visoka preciznost, što je dokumentovano u [1].' | Ocena 1: 'Visoka preciznost je ključna za ovaj domen. [1]' (Citat je prisutan, ali je pozicioniran predaleko od tvrdnje) | Ocena 0: 'Naš metod je najbolji pristup na svetu, što je opšte poznata činjenica.' (Neosnovana tvrdnja bez citata/dokaza)",
        },
        "Konciznost rečenice": {
            "instruction": "Svaka rečenica mora sadržati jednu i samo jednu poentu.",
            "few-shot": "Ocena 2: 'Prvo je sprovedena analiza ulaznih podataka. Zatim su podaci normalizovani.' | Ocena 1: 'Analiza je potvrdila da su podaci nekonzistentni, što je rezultiralo potrebom za dodatnom normalizacijom.' (Dve blisko povezane poente) | Ocena 0: 'Modeli su obučeni, testirani su, pri čemu je uočena niska preciznost, a to je verovatno zbog lošeg izbora hiperparametara.' (Previše poenti/klauza)",
        },
        "Jasnoća rečenica": {
            "instruction": "Svaka rečenica treba biti nedvosmislena, lako razumljiva i logično strukturirana, bez suvišne složenosti ili nepreciznih formulacija.",
            "few-shot": "Ocena 2: 'Metoda bazirana na grafovima omogućava efikasno procesiranje složenih relacija.' | Ocena 1: 'Uvođenje novog, naprednog, prilagodljivog mehanizma poboljšava efikasnost.' (Previše nepreciznih prideva) | Ocena 0: 'S obzirom na to da je bilo potrebno da se, u kontekstu rezultata, proveri validnost, sprovedena je kompleksna provera.' (Zapetljana i nejasna struktura)",
        },
        "Aktiv i pasiv": {
            "instruction": "Aktiv treba koristiti kao podrazumevanu formu, dok se pasiv upotrebljava samo kada je izvršilac radnje nebitan, očigledan ili kada je fokus na rezultatu ili opštoj činjenici.",
            "few-shot": "Ocena 2: 'Autori su predstavili novi algoritam.' (Aktiv) | Ocena 1: 'Potom je od strane algoritma izvršena provera.' (Pasiv gde je aktiv bolji) | Ocena 0: 'Od strane nas je izveden eksperiment. Od strane sistema je implementirana funkcionalnost.' (Preterana, nepotrebna upotreba pasiva)",
        },
        "Početak rečenice": {
            "instruction": "Rečenica ne sme početi sa rečima A, Ili, I, kao ni sa brojevima.",
            "few-shot": "Ocena 2: 'Struktura rešenja je opisana detaljno.' | Ocena 1: 'Iako je metod efikasan,...' (Počinje veznikom *Iako*, ali je prihvatljivo) | Ocena 0: 'I sprovedena je provera. A rezultati su bili loši. Ili je problem bio u podacima.' (Počinje sa I, A, Ili)",
        },
        "Sadržaj rečenice": {
            "instruction": "Rečenice treba da budu formalne. Ne treba da koriste žargone i slengove. Ne treba da se obraćaju čitaocu direktno.",
            "few-shot": "Ocena 2: 'Utvrđena je korelacija između varijabli i nivoa greške.' (Formalno) | Ocena 1: 'Možda bi *bolja fora* bila da se primeni drugi model.' (Manji žargon/neformalnost) | Ocena 0: 'Kao što vidite, *rezultati su baš kul*, pa *hajde da vidimo* kako dalje.' (Direktno obraćanje čitaocu i žargon)",
        },
        "Korišćenje vremena": {
            "instruction": "Prezent se koristi prilikom iskazivanja činjenica, dok se perfekt koristi prilikom spominjanja sopstvenih rezultata.",
            "few-shot": "Ocena 2: 'Model *pokazuje* preciznost (činjenica). Mi *smo dobili* rezultate (sopstveni rad).' | Ocena 1: 'Rezultati *pokazuju* visoku preciznost, ali *mi ćemo implementirati* poboljšanje.' (Nekonzistentan prelaz) | Ocena 0: 'Model *pokazivao* je visoku preciznost, iako je opšte poznata činjenica da *treba* da se koristi prezent.' (Široko rasprostranjena greška u upotrebi vremena)",
        },
        "Zarezi": {
            "instruction": "Zarez se obavezno koristi uz a i ali, zabranjen je uz i i ili, koristi se kod nabrajanja, za razdvajanje nezavisnih iskaza u istoj rečenici i obavezan je u apoziciji.",
            "few-shot": "Ocena 2: 'Rad je interesantan, ali zahteva doradu. Rezultati su dobri i pouzdani.' (Pravilna upotreba) | Ocena 1: 'Analiza je sprovedena, što je bilo neophodno.' (Nedostaje zarez pre *što*) | Ocena 0: 'Model je brz, i efikasan, a, rezultati, su dobri.' (Višestruke greške u upotrebi zareza)",
        },
        "Interpunkcija": {
            "instruction": "Izbegavati uzvičnike; duže crtice koriste se za umetnute komentare, kraće za spajanje reči, a tačka-zarez za pauzu dužu od zareza a kraću od tačke, posebno kada druga klauzula proširuje ili objašnjava prvu.",
            "few-shot": "Ocena 2: 'Rezultati (koji su bili iznenađujući) su prezentovani u Tabeli 1.' (Pravilna interpunkcija) | Ocena 1: 'Analiza je završena, međutim, dalja istraživanja su neophodna; (Tačka-zarez je mogao biti bolji).' | Ocena 0: 'To je zaista neverovatno! (Uzvičnik). To je bio ključni faktor – to jest, podatak.' (Upotreba uzvičnika i nepravilne crtice)",
        },
        "Konciznost paragrafa": {
            "instruction": "Jedan paragraf treba da opisuje jednu i samo jednu temu. Jedna tema ne treba da bude razbijena na više paragrafa.",
            "few-shot": "Ocena 2: Paragraf objašnjava samo metodologiju. Sledeći paragraf objašnjava samo rezultate. | Ocena 1: Paragraf počinje objašnjenjem rezultata i završava sa dve rečenice o budućem radu. (Manji prelazak na drugu temu) | Ocena 0: Paragraf pokriva definicije, metodologiju, rezultate i zaključak. (Grubo mešanje tema)",
        },
        "Organizacija paragrafa": {
            "instruction": "Paragraf treba da ima uvodnu rečenicu koja ističe glavnu ideju, rečenice koje je dosledno razrađuju objašnjenjima, primerima ili dokazima, i zaključnu rečenicu koja sumira implikacije ili povezuje sa narednim paragrafom.",
            "few-shot": "Ocena 2: Paragraf počinje glavnom idejom, detaljno je razrađuje i završava zaključkom. | Ocena 1: Paragrafu nedostaje jasna zaključna rečenica. (Manji nedostatak u strukturi) | Ocena 0: Paragraf je samo lista nepovezanih tvrdnji bez uvodne rečenice.",
        },
        "Konzistentnost": {
            "instruction": "Za određeni koncept se konzistentno koristi isti termin/fraza.",
            "few-shot": "Ocena 2: Kroz ceo rad se koristi isključivo termin *klasifikator*. | Ocena 1: Na dva mesta je korišćen termin *klasifikator*, a na jednom mestu *prediktor* za istu stvar. (Manja nekonzistentnost) | Ocena 0: Ista tehnika je nazivana *algoritam učenja*, *model*, i *mašina za predikciju* u jednom poglavlju. (Ozbiljna nekonzistentnost)",
        },
        "Repetitivnost": {
            "instruction": "Tekst ne sme biti repetitivan, odnosno ne treba ponavljati iste pojmove, objašnjenja ili informacije već jasno iznete drugde u tekstu ili prikazane na slici.",
            "few-shot": "Ocena 2: Informacija je izneta samo jednom u tekstu. | Ocena 1: Ista rečenica je ponovljena u paragrafu na početku i na kraju rada. (Manje ponavljanje) | Ocena 0: Ključni rezultat je opisan i u poglavlju *Rešenje*, i *Rezultati*, i *Zaključak* identičnim rečenicama. (Široko rasprostranjeno ponavljanje)",
        },
        "Bespotrebni detalji": {
            "instruction": "Tekst ne treba da sadrži nepotrebne detalje, poput trivijalnih isečaka koda ili definisanja koncepata koji nisu ključni za razumevanje teme.",
            "few-shot": "Ocena 2: Tekst sadrži samo relevantne detalje o implementaciji. | Ocena 1: Naveden je kratak isečak trivijalnog koda koji nije ključan. (Manji nepotrebni detalj) | Ocena 0: Uključen je opširan opis instalacije biblioteka koje su standardne u domenu. (Veliki nepotrebni detalj)",
        },
        "Odnos teksta i teme rada": {
            "instruction": "Sve što je izloženo mora biti povezano sa temom rada, odnosno, ne postoji tekst čija povezanost sa temom nije jasna.",
            "few-shot": "Ocena 2: Svaka rečenica se odnosi na primenjenu metodologiju ili rezultate. | Ocena 1: Jedan paragraf sadrži opšte informacije o istoriji informatike, čija veza s temom nije sasvim jasna. (Mala irelevantnost) | Ocena 0: Pola poglavlja *Teorijske osnove* je posvećeno temi koja je napuštena u fazi implementacije. (Gruba irelevantnost)",
        },
        "Nekonciznost": {
            "instruction": "Treba izbegavati korišćenje generičkih i bespotrebnih reči u tekstu.",
            "few-shot": "Ocena 2: Tekst je direktan i jasan, bez suvišnih reči. | Ocena 1: Često korišćenje fraza poput 'u suštini', 'izvesno je', 'opšte je poznato'. (Manja verbalna zagušenost) | Ocena 0: Svaka rečenica počinje sa 'Ono što je važno reći jeste...', 'Možemo konstatovati da...', itd. (Ozbiljna zagušenost)",
        },
        "Korišćenje ličnih zamenica": {
            "instruction": "Izbegavati korišćenje ličnih zamenica radi izbegavanja subjektivnosti rada.",
            "few-shot": "Ocena 2: Korišćene su samo objektivne forme. | Ocena 1: Na jednom mestu je napisano 'Mi mislimo da je...'. (Manji subjektivni glas) | Ocena 0: Tekst je pun rečenica: 'Mi smo sproveli eksperiment. Mi smo primetili. Mi smatramo. Naša analiza.' (Široko rasprostranjen subjektivni glas)",
        },
    },
    "chapters": {
        "Problem": {
            "Širi problem": {
                "instruction": "Širi problem koji rad obrađuje treba da bude jasno predstavljen tako da se odmah razume njegov kontekst i važnost. Čitalac ne bi trebalo da mora da istražuje dodatne izvore da bi shvatio zašto je tema relevantna.",
                "few-shot": "Ocena 2: 'Uvod jasno objašnjava kontekst *online toksičnosti* u društvenim medijima i zašto je to globalno bitno (uticaj na mentalno zdravlje i platforme).' | Ocena 1: 'Opisan je širi problem, ali čitaocu nedostaje jedan ključni termin da bi razumeo kontekst (npr. šta je *deepfake*). ' | Ocena 0: 'Problem je definisan samo tehničkim terminima (Nema dovoljno RAM-a za obradu podataka) bez objašnjenja društvene ili naučne važnosti.'",
            },
            "Osnovni koncepti": {
                "instruction": "Osnovni koncepti za razumevanje problema treba da budu jasno definisani tako da čitalac može da razume tekst bez dodatnog istraživanja. Istovremeno, svaki definisani koncept treba da bude neophodan, bez suvišnih pojmova koji ne doprinose razumevanju.",
                "few-shot": "Ocena 2: 'Koncepti *Klasifikacija*, *Skup podataka* i *Evaluacija* su definisani, i svi su korišćeni kasnije u tekstu.' | Ocena 1: 'Koncept *Klasterovanje* je definisan u uvodu, iako se nikada ne koristi u ostatku rada.' | Ocena 0: 'Nijedan ključni termin (kao što je *Mašinsko učenje*) nije definisan, a definisano je pet irelevantnih pojmova iz fizike.'",
            },
            "Značaj rešenja": {
                "instruction": "Istaknuto je zašto je priloženo rešenje značajno za društvo. Objašnjava se koja je motivacija iza rešenja i problema koji se rešava.",
                "few-shot": "Ocena 2: 'Eksplicitno se navodi da rešenje pomaže *moderatorima platformi* i *istraživačima* u borbi protiv dezinformacija, čime se poboljšava digitalna sigurnost.' | Ocena 1: 'Spomenuto je da će 'nekome rešenje biti korisno', ali bez navođenja konkretne motivacije ili šireg društvenog uticaja.' | Ocena 0: 'Nema reči o tome zašto je rešenje značajno za šire društvo, fokus je samo na tehničkom izazovu (npr. 'rešavamo problem jer je težak').'",
            },
            "Pozicioniranje užeg problema u širem kontekstu": {
                "instruction": "tekst treba najpre da predstavi opšti okvir oblasti, a zatim jasno da prikaže kako se konkretan uži problem logično uklapa u taj širi kontekst. Ovo omogućava čitaocu da razume relevanciju i važnost problema bez potrebe za dodatnim istraživanjem.",
                "few-shot": "Ocena 2: 'Rad prvo opisuje *oblast NLP-a*, zatim problem *toksičnosti*, i na kraju precizira da se bavi *detekcijom sarkazma* unutar toksičnog sadržaja.' | Ocena 1: 'Opisana je oblast, ali se odmah prelazi na uži problem bez jasne rečenice koja ih logički povezuje.' | Ocena 0: 'Odmah se počinje sa opisom užeg problema (Model X), bez uvoda u širi kontekst mašinskog učenja ili NLP-a.'",
            },
            "Opis fokusa rada": {
                "instruction": "Jasno je očekivano ponašanje rešenja i kada/kako se koristi.",
                "few-shot": "Ocena 2: 'Očekuje se da sistem prima Twitter poruke i da u realnom vremenu vraća binarnu odluku (toksično/nije toksično).' | Ocena 1: 'Navedeno je da rešenje obrađuje poruke, ali nije jasno da li vraća binarni rezultat ili verovatnoću.' | Ocena 0: 'Fokus rada je opisan sa 'cilj je dobar model', ali nedostaje opis ulaznih i izlaznih podataka i uslova korišćenja.'",
            },
            "Opis koristi rešenja": {
                "instruction": "Istaknute su konkretne interesne grupe koje bi rešenje koristile i na koji način.",
                "few-shot": "Ocena 2: 'Konkretne interesne grupe su *moderatori foruma* (za automatsko filtriranje), *istraživači* (za analizu trendova) i *roditelji* (za nadzor).' | Ocena 1: 'Spomenuto je da će rešenje koristiti 'korisnicima interneta', što je previše generička grupa.' | Ocena 0: 'Nema opisa konkretnih interesnih grupa, već samo opšte tvrdnje o poboljšanju tehnologije.'",
            },
        },
        "Teorijske osnove": {
            "Opis problema domena": {
                "instruction": "Opis problema treba da pruži detaljan prikaz domena iz kojeg jasno proističu zahtevi koje rešenje mora da ispuni. Tekst treba da identifikuje ključne potrebe, izazove ili ciljeve koji određuju skup funkcionalnih, tehničkih ili metodoloških zahteva rešenja.",
                "few-shot": "Ocena 2: 'Detaljno su opisani zahtevi domena (npr. *potreba za brzim odzivom u realnom vremenu* i *visoka preciznost na neuravnoteženom skupu podataka*) iz kojih proističu svi tehnički zahtevi.' | Ocena 1: 'Opis problema je dobar, ali nedostaje objašnjenje jednog ključnog funkcionalnog zahteva (npr. *zahtev za skalabilnošću*).' | Ocena 0: 'Opis domena je generički i ne identifikuje nijedan konkretan zahtev koji rešenje mora da ispuni, fokus je na opštoj teoriji, ne na izazovima.'",
            },
            "Zahtevi": {
                "instruction": "Potrebno je navesti jasno definisane stavke koje opisuju šta tačno rešenje treba da omogući ili koji problem treba da otkloni.",
                "few-shot": "Ocena 2: 'Jasno je navedena lista zahteva, npr. *Brzina odziva ispod 100ms*, *Preciznost (F1) > 90%* i *Podrška za tri jezika*.' | Ocena 1: 'Navedena su tri zahteva, od kojih je jedan nejasan (npr. 'Sistem mora biti dobar i efikasan').' | Ocena 0: 'Nema jasno definisanih zahteva; umesto toga su navedene samo opšte želje ('želimo da sistem bude precizan').'",
            },
            "Opis drugačijih rešenja": {
                "instruction": "Tekst sažeto prikazuje moguće alternativne pristupe bez ulaska u preterane detalje, kako bi se obezbedio jasan pregled postojećih opcija.",
                "few-shot": "Ocena 2: 'Prikazana su tri alternativna pristupa (A, B, C) sa kratkim opisom prednosti/mana, bez suvišnih tehničkih detalja.' | Ocena 1: 'Prikazan je samo jedan alternativni pristup (A) iako u domenu postoje barem još dva relevantna. ' | Ocena 0: 'Opisana su samo alternativna rešenja (sa previše detalja), bez njihovog sažetog poređenja ili pregleda.' ",
            },
            "Argumentacija odabranog rešenja": {
                "instruction": "Prisutan je razlog zbog kojeg je odabrano predstavljeno rešenje u radu.",
                "few-shot": "Ocena 2: 'Eksplicitno je navedeno: 'LSTM model je odabran zbog dokazane sposobnosti da efikasno obradi sekvencijalne podatke, što je ključno za NLP.' | Ocena 1: 'Navedeno je 'Odabran je model X jer je najbolji', ali bez konkretnog tehničkog argumenta zašto je bolji za ovaj domen.' | Ocena 0: 'Potpuno nedostaje objašnjenje zašto je odabrana metoda X, samo se nastavlja opis metode X.'",
            },
            "Opis koncepata rešenja": {
                "instruction": "Jasno predstavljanje tehnologija, modela i pristupa koji se koriste, tako da čitalac nakon čitanja poseduje sve neophodno znanje za razumevanje rezultata rešenja.",
                "few-shot": "Ocena 2: 'Opisuje se rad *Transfer Learning* tehnike i kako ona funkcioniše u kontekstu jezičkih modela, što je ključno za razumevanje implementacije.' | Ocena 1: 'Opisana je tehnologija A, ali je preskočeno objašnjenje tehnologije B koja je jednako ključna za rešenje.' | Ocena 0: 'Umesto opisa koncepata, dat je samo generički pregled istorije razvoja NLP-a.'",
            },
            "Definisani koncepti": {
                "instruction": "Svi pojmovi su jasno objašnjeni tako da čitalac može da razume problem i rešenje bez dodatnih pitanja ili nejasnoća.",
                "few-shot": "Ocena 2: 'Svi pojmovi (kao što su *tokenizacija* i *embedding*) su definisani jasno i precizno u skladu sa literaturom.' | Ocena 1: 'Većina pojmova je definisana, ali jedan ključni pojam (*F1-score*) nije definisan, iako se koristi u rezultatima.' | Ocena 0: 'Nijedan tehnički pojam nije definisan, pretpostavlja se da je čitaocu sve poznato.'",
            },
            "Višak koncepata": {
                "instruction": "Ne postoje koncepti koji nisu potrebni za razumevanje rada.",
                "few-shot": "Ocena 2: 'Svi predstavljeni koncepti (A, B, C) su direktno iskorišćeni u implementaciji ili evaluaciji rešenja.' | Ocena 1: 'Opisan je koncept D, iako se on odnosi na napuštenu metodu koja nema veze sa finalnim rešenjem.' | Ocena 0: 'Pola poglavlja posvećeno je opisu *mašinskog učenja* i *veštačke inteligencije*, što je previše širok i nepotreban kontekst.'",
            },
            "Opis rešenja koncepata": {
                "instruction": "Jasno je kako je svaki od zahteva sistema realizovan.",
                "few-shot": "Ocena 2: 'Eksplicitno je rečeno: 'Zahtev za brzom odzivom (Zahtev #1) rešen je korišćenjem *GPU akceleracije* i *kvantizacije modela*.' | Ocena 1: 'Opisani su zahtevi, ali za jedan zahtev nije opisano *kako* je realizovan, već samo da je realizovan.' | Ocena 0: 'Zahtevi su navedeni na početku poglavlja, ali se nigde u tekstu ne objašnjava njihova realizacija.'",
            },
        },
        "Rešenje": {
            "Opis rešenja na opštijem nivou": {
                "instruction": "Pojednostavljen pregled toga koje potrebe rešenje ispunjava i kakav se ulaz i izlaz očekuje, bez ulaska u detaljne korake procesa. Tekst treba da omogući čitaocu da razume osnovni način funkcionisanja rešenja na visokom nivou.",
                "few-shot": "Ocena 2: 'Rešenje prihvata neobrađen tekst (ulaz) i vraća kategorizovanu ocenu toksičnosti (izlaz), čime ispunjava zahtev za automatskom moderacijom.' | Ocena 1: 'Opis obuhvata ulaz i izlaz, ali ulazi u detalje implementacije i spominje previše koraka procesa (npr. *tokenizacija* i *vektorizacija*).' | Ocena 0: 'Potpuno nedostaje objašnjenje ulaza i izlaza rešenja, već se odmah prelazi na tehničke detalje obuke modela.'",
            },
            "Inicijalno predstavljanje strukture sistema": {
                "instruction": "Kratko i pregledno prikazivanje glavnih komponenti, modula ili faza rešenja, bilo kroz sažet paragraf (za jednostavne pristupe) ili kroz odgovarajući dijagram u slučaju složenije obrade. Cilj je da čitalac odmah stekne osnovni uvid u organizaciju i tok sistema.",
                "few-shot": "Ocena 2: 'Kratak paragraf i dijagram objašnjavaju da se rešenje sastoji od modula A (predobrada), B (klasifikacija) i C (izveštavanje).' | Ocena 1: 'Struktura je opisana, ali je dijagram loše formatiran i nerazumljiv, ili se oslanja samo na listu bez vizuelnog prikaza/toka.' | Ocena 0: 'Nema kratkog pregleda strukture sistema; čitalac je odmah primoran da čita detalje komponenti, bez uvida u celinu.'",
            },
            "Opis svake celine strukture rešenja": {
                "instruction": "Opis svake celine strukture rešenja treba da bude predstavljen tek nakon što su jasno definisani cilj i ukupna struktura sistema, a obuhvata detaljno objašnjenje procesa poput prikupljanja, obrade i analize podataka ili izgradnje i evaluacije sistema. Tekst mora sistematski razložiti svaku komponentu tako da je njen doprinos celini jasno razumljiv.",
                "few-shot": "Ocena 2: 'Svaka komponenta (Modul A, B, C) je opisana detaljno tek nakon što je predstavljena ukupna arhitektura sistema, objašnjavajući tačan doprinos svakog modula.' | Ocena 1: 'Modul A je opisan pre nego što je u radu predstavljena ukupna arhitektura sistema (kršenje logičkog redosleda).' | Ocena 0: 'Komponente su opisane, ali je logički redosled pogrešan (npr. opis Modula C pre Modula A) ili nedostaje objašnjenje doprinosa celini.'",
            },
            "Preciznost": {
                "instruction": "Tekst daje dovoljno relevantnih detalja da čitalac može jasno da razume postupak bez dodatnih pitanja ili nejasnoća.",
                "few-shot": "Ocena 2: 'Tekst objašnjava da je korišćen *LSTM model* sa *128 neurona* i *Adam optimizatorom* pri *stoppoing kriterijumu* od 10 epoha.' | Ocena 1: 'Navedeno je da je korišćen *LSTM*, ali bez ključnih hiperparametara (broj slojeva, veličina reči) ili optimizatora.' | Ocena 0: 'Tekst samo navodi: 'Korišćen je dobar model dubokog učenja za klasifikaciju, na osnovu skupa podataka.''",
            },
            "Visok nivo apstrakcije": {
                "instruction": "Rešenje treba opisati na konceptualnom i opštem nivou, bez ulaska u implementacione detalje kao što su isečci koda ili tehničke sitnice.",
                "few-shot": "Ocena 2: 'Opisan je protok podataka kroz komponente i njihova funkcionalna uloga, bez navođenja sintakse Pythona ili C++ klase.' | Ocena 1: 'Na jednom mestu je uključen isečak koda od 5 linija koji pokazuje trivijalno definisanje varijable.' | Ocena 0: 'Uključeni su veliki isečci koda, detalji instalacije biblioteka i tehničke sitnice koje nisu konceptualno bitne.'",
            },
            "Teorijske osnove": {
                "instruction": "Ne treba opisivati kako neka procedura, algoritam ili nešto treće funkcioniše, već kako je korišćeno.",
                "few-shot": "Ocena 2: 'Fokus je na tome *kako je NLP model primenjen* u sistemu za detekciju toksičnosti, a ne *kako NLP modeli opšte funkcionišu*.' | Ocena 1: 'Uključeno je pola stranice opšteg objašnjenja šta je NLP, pre nego što se prešlo na primenu.' | Ocena 0: 'Poglavlje u potpunosti objašnjava teoriju iza *Tokenizacije* i *Vektorskih prostora* bez objašnjenja *kako su tačno korišćeni u ovom rešenju*.'",
            },
            "Problem": {
                "instruction": "Potrebno je opisati koji su problemi nastali tokom izrade rada. Ukoliko ih nije bilo, potrebno je eksplicitno navesti tu konstataciju.",
                "few-shot": "Ocena 2: 'Eksplicitno navedeno: 'Najveći problem tokom izrade bio je obezbeđivanje kvalitetnog *dataset*-a.' (Identifikovan i naveden problem) | Ocena 1: 'Navedeno je 'Nije bilo većih problema tokom izrade rešenja.', ali bez eksplicitne konstatacije (nedostaje *eksplicitno*). ' | Ocena 0: 'Problemi nisu spomenuti, već se fokus nastavlja samo na opis rešenja.'",
            },
        },
        "Rezultati": {
            "Konciznost": {
                "instruction": "Jasno je objašnjen poželjan, odnosno nepoželjan ishod evaluacije.",
                "few-shot": "Ocena 2: 'Jasno je objašnjeno da je cilj *visoka F1 mera*, dok je *niska preciznost* u našem kontekstu nepoželjna.' | Ocena 1: 'Poželjan ishod je objašnjen u dve rečenice, a nepoželjan ishod samo jednom kratkom. ' | Ocena 0: 'Nije objašnjeno koji rezultat se smatra uspehom, a koji neuspehom; navode se samo brojevi.'",
            },
            "Dovoljno detalja": {
                "instruction": "Na osnovu datog opisa eksperimenta u tekstu, moguće ga je reprodukovati.",
                "few-shot": "Ocena 2: 'Opis eksperimenta sadrži sve detalje o hardveru (GPU, CPU), korišćenim bibliotekama (verzija 1.2.1), i ključnim hiperparametrima modela.' | Ocena 1: 'Navedene su korišćene biblioteke i model, ali nedostaju verzije ili detalji hardvera, što otežava potpunu reprodukciju.' | Ocena 0: 'Opisano je samo da je 'eksperiment sproveden', bez navođenja modela, hardvera ili optimizatora, nemoguće je reprodukovati.'",
            },
            "Vreme pisanja": {
                "instruction": "Potrebno je koristiti prošlo vreme u opisu dobijenih rezultata.",
                "few-shot": "Ocena 2: 'Konačni F1 skor **iznosio** je 0.92.' (Perfekt) | Ocena 1: 'Model **će postići** preciznost 0.92.' (Korišćenje budućeg vremena) | Ocena 0: 'Model **postiže** preciznost 0.92 i to je naš rezultat.' (Korišćenje prezenta)",
            },
            "Struktura opisa rezultata": {
                "instruction": "Tekst logično i jasno tumači predstavljene rezultate. Ukoliko postoji više rezultata, sortirani su po značaju ili hronološki.",
                "few-shot": "Ocena 2: 'Rezultati su sortirani po značaju: prvo ključni F1 skor, zatim matrica konfuzije i na kraju sporedni parametri (vreme odziva).' | Ocena 1: 'Rezultati su prikazani, ali nisu logički sortirani, već su izmešani, počevši od najmanje bitne tabele.' | Ocena 0: 'Rezultati su prezentovani kao sirovi brojevi bez logičnog tumačenja ili strukture.'",
            },
            "Diskusija o rezultatima": {
                "instruction": "Komentarisane su prednosti i ograničenja rezultata. Istaknuto je u kojim kontekstima je rešenja pouzdano, a u kojima nije. Istaknuto je koji zahtevi nisu pokriveni rešenjem, ako takvi zahtevi postoje. Rešenje se poredi sa ostalim rešenjima, ukoliko takva rešenja postoje.",
                "few-shot": "Ocena 2: 'Komentarisana su ograničenja (slabo radi na *sarkazmu*) i prednosti (brzina), te je upoređeno sa radom [1] i [2].' | Ocena 1: 'Diskusija je prisutna, ali nedostaje poređenje sa konkurentskim rešenjima iz literature.' | Ocena 0: 'Naveden je samo rezultat (0.92) bez ikakve diskusije o prednostima, manama, ili poređenju.'",
            },
            "Finalni paragraf": {
                "instruction": "Potrebno je sintezirati sve rezultate u jednom paragrafu na kraju poglavlja. Navodi se u budući rad.",
                "few-shot": "Ocena 2: 'Poslednji paragraf sumira ključne nalaze (F1 0.92) i predlaže *proširenje skupa podataka* i *implementaciju interpretativnosti* kao budući rad.' | Ocena 1: 'Finalni paragraf sumira rezultate, ali ne spominje budući rad.' | Ocena 0: 'Poglavlje se završava prikazom poslednje tabele, bez sumirajućeg zaključka i bez spominjanja budućeg rada.'",
            },
        },
    },
}

# -------------------------------
# Section parsing (split by heading number/size)
# -------------------------------
ROMAN_HEAD_RE = re.compile(r"(?m)^\s*(I|II|III|IV)\.\s*(.*)$")
ARABIC_HEAD_RE = re.compile(r"(?m)^\s*([1-4])\.\s*(.*)$")

# -------------------------------
# Helpers
# -------------------------------


def build_system_prompt(expansion_key: str) -> str:
    """Return the filled system prompt using a single expansion key.

    expansion_key must be one of keys in EXPANSIONS. If unknown, 'none' is used.
    """
    if not expansion_key or expansion_key not in EXPANSIONS:
        expansion = EXPANSIONS["none"]
        logger.info(f"Expansion: {expansion}")
    else:
        expansion = EXPANSIONS[expansion_key]
    return BASE_SYSTEM_PROMPT.format(expansion=expansion)


def generate_rules_prompt(
    rules: Dict[str, Any],
    include_global: bool = True,
    include_chapters: Optional[List[str]] = None,
    include_instructions: bool = True,
    include_few_shot: bool = False,
) -> str:
    """Create the user prompt fragment that enumerates the rules in plain text."""
    parts: List[str] = ["Pravila na osnovu kojih treba da se evaluira tekst:", ""]

    def format_rule(name: str, meta: Dict[str, Any]) -> str:
        """Helper to format a single rule based on flags."""
        instr = meta.get("instruction", "")
        few_shot_ex = meta.get("few-shot", "")

        line_parts = [f"- {name}"]

        if include_instructions and instr:
            line_parts.append(f": {instr}")

        if include_few_shot and few_shot_ex:
            line_parts.append(f" (Primeri: {few_shot_ex})")

        return "".join(line_parts)

    # --- Global Rules ---
    if include_global and rules.get("global"):
        parts.append("=== GLOBALNA PRAVILA ===")
        for name, meta in rules["global"].items():
            parts.append(format_rule(name, meta))
        parts.append("")

    # --- Chapter Rules ---
    if include_chapters is not None and rules.get("chapters"):
        chapters_to_use = (
            include_chapters
            if len(include_chapters) > 0
            else list(rules["chapters"].keys())
        )
        for ch in chapters_to_use:
            if ch not in rules["chapters"]:
                continue
            parts.append(f"=== PRAVILA ZA POGLAVLJA: {ch} ===")
            for name, meta in rules["chapters"][ch].items():
                parts.append(format_rule(name, meta))
            parts.append("")

    return "\n".join(parts).strip()


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown text using pymupdf4llm. Returns empty string on failure."""
    try:
        return pymupdf4llm.to_markdown(pdf_path)
    except Exception as e:
        logger.exception("Failed to parse PDF %s: %s", pdf_path, e)
        return ""


def extract_json_from_model_output(text: str) -> List[Dict[str, Any]]:
    """Robustly extract JSON list of rule evaluations from model text.

    Handles fenced JSON (```json ... ```), plain arrays, or multiple top-level objects.
    Raises ValueError if extraction/parsing fails.
    """
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()

    # Remove code fences
    s = re.sub(r"^```json\\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s)
    s = s.strip()

    # Try to find the first JSON array in the text
    match_array = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if match_array:
        candidate = match_array.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # fall through to other heuristics
            pass

    # If it's a single JSON object
    match_obj = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match_obj and not match_array:
        # Could be multiple adjacent objects like {..}{..} or {..}, {..}
        # Try to split into individual top-level objects
        objs: List[str] = []
        depth = 0
        buf = ""
        for ch in s:
            if ch == "{":
                depth += 1
            if depth > 0:
                buf += ch
            if ch == "}":
                depth -= 1
                if depth == 0 and buf:
                    objs.append(buf)
                    buf = ""
        parsed = []
        for o in objs:
            try:
                parsed.append(json.loads(o))
            except json.JSONDecodeError:
                # if any single object fails, abort
                parsed = []
                break
        if parsed:
            return parsed

    # As a last resort try to parse the whole response as JSON
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse model output as JSON. Last error: {e}\nOutput was:\n{s[:1000]}"
        )

    raise ValueError("Unable to extract JSON from model output")


def parse_sections_by_number(text: str):
    """
    Split text into top-level numbered sections (Roman numerals I, II, III, IV).

    Rules:
    - Map I → Problem, II → Teorijske osnove, III → Resenje, IV → Rezultati
    - If multiple sections with the same numeral exist, take the one with the most words.
    - Returns a dict: {'Problem': body, 'Teorijske osnove': body, 'Resenje': body, 'Rezultati': body}
    """
    if not text:
        return {}

    # Find all Roman numeral headings
    matches = list(ROMAN_HEAD_RE.finditer(text))
    if not matches:
        # fallback: whole text as 'Problem'
        return {"Problem": text.strip()}

    # Temporary dict to store all candidates per numeral
    candidates = {}

    for i, m in enumerate(matches):
        start = m.start()
        marker = m.group(1).strip()
        # Determine end: next match start or end of text
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Add candidate for this numeral
        if marker not in candidates:
            candidates[marker] = []
        candidates[marker].append(body)

    # Mapping of Roman numeral → chapter name
    numeral_to_chapter = {
        "I": "Problem",
        "II": "Teorijske osnove",
        "III": "Rešenje",
        "IV": "Rezultati",
    }

    # Pick the candidate with most words for each numeral
    result = {}
    for numeral, chapter in numeral_to_chapter.items():
        if numeral in candidates:
            # choose the body with the most words
            best_body = max(candidates[numeral], key=lambda b: len(b.split()))
            result[chapter] = best_body

    return result


def evaluate_sections(md_text: str, expansion_key: str) -> List[Dict[str, Any]]:
    """
    Perform evaluation in 'sections' mode:
      1) Global rules on the whole text (one call)
      2) Chapter-level rules for each of the top 4 chapters in order:
         Problem -> Teorijske osnove -> Resenje -> Rezultati

    Returns a single flattened list of rule evaluation objects (naziv_pravila + ocena).
    """
    all_evals: List[Dict[str, Any]] = []

    # --- 1) Global evaluation ---
    try:
        logger.info("Sections mode: calling model for GLOBAL rules (full text).")
        global_eval = evaluate_markdown(md_text, expansion_key)
        all_evals.extend(global_eval)
    except Exception as e:
        logger.exception("Global evaluation failed: %s", e)

    # --- 2) Parse sections ---
    sections = parse_sections_by_number(
        md_text
    )  # now returns dict keyed by chapter names
    if not sections:
        logger.warning(
            "No numbered sections found in document; sections mode will only include global evaluation."
        )
        return all_evals

    # --- 3) Evaluate each chapter in order ---
    chapter_order = ["Problem", "Teorijske osnove", "Rešenje", "Rezultati"]

    for chapter_key in chapter_order:
        body = sections.get(chapter_key)
        if not body:
            logger.warning(
                "No content found for chapter '%s'. Skipping evaluation.", chapter_key
            )
            continue

        try:
            logger.info(
                "Sections mode: calling model for chapter '%s', size=%d chars",
                chapter_key,
                len(body),
            )

            # Build prompt including only rules for this chapter
            rules_prompt = generate_rules_prompt(
                RULES,
                include_global=False,
                include_chapters=[chapter_key],
                include_instructions=(expansion_key != "zero_shot_expansion"),
                include_few_shot=(expansion_key == "few_shot_expansion"),
            )

            logger.info(f"Rules prompt: {rules_prompt}")

            user_prompt = f"""
                {rules_prompt}

                Ovo je sadržaj rada koji treba da se evaluira:

                {body}

                Evaluirati rad na osnovu šeme iz system prompta i pravila koja sam ti dao.
            """.strip()

            system_prompt = build_system_prompt(expansion_key)
            raw = call_model(system_prompt, user_prompt)
            parsed = extract_json_from_model_output(raw)
            logger.info(f"Parsed: {parsed}")
            all_evals.extend(parsed)

            # polite pause to avoid rate limits
            time.sleep(1.0)

        except Exception as e:
            logger.exception("Failed to evaluate chapter '%s': %s", chapter_key, e)
            continue

    logger.info(f"Final output: {all_evals}")
    return all_evals


# -------------------------------
# Chat / API wrapper with retries
# -------------------------------


def call_model(
    system_prompt: str, user_prompt: str, max_retries: int = 3, backoff: float = 1.0
) -> str:
    """Call the chat completion API and return the assistant text. Retries on transient errors.

    Note: we use the same `client.chat.completions.create` shape as in your original code. If your SDK is
    different in your environment, you may need to adapt this function.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            completion = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1,
            )

            text = completion.choices[0].message.content
            if not text:
                raise ValueError("Empty response from GPT")

            return text

        except Exception as e:
            logger.exception("Model call failed on attempt %d: %s", attempt, e)
            if attempt >= max_retries:
                raise
            sleep_time = backoff * (2 ** (attempt - 1))
            logger.info("Retrying in %0.1f seconds...", sleep_time)
            time.sleep(sleep_time)
            """GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                top_p=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # response.choices[0].message.content is what your earlier code used
            text = response.choices[0].message.content
            """

            """ GEMINI
            full_prompt = f"""
            {system_prompt}

            {user_prompt}
            """.strip()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            # Gemini vraća listu kandidata; tekst je u .text
            text = response.text

            message = client.messages.create(
                model=os.getenv("CLAUDE_DEPLOYMENT", "claude-sonnet-4-5"),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                temperature=0,
            )

            if isinstance(message.content, list) and len(message.content) > 0:
                block = message.content[0]
                # block može biti dict ili objekat sa .text
                text = (
                    block.get("text")
                    if isinstance(block, dict)
                    else getattr(block, "text", "")
                )
            else:
                text = ""

            if not text:
                raise ValueError("Empty response from Claude")
"""


# -------------------------------
# High-level evaluation
# -------------------------------


def evaluate_markdown(markdown_text: str, expansion_key: str) -> List[Dict[str, Any]]:
    """Form the prompts, call model, and return parsed JSON evaluation list."""

    include_instructions = expansion_key != "zero_shot_expansion"

    include_few_shot = expansion_key == "few_shot_expansion"

    rules_prompt = generate_rules_prompt(
        RULES,
        include_global=True,
        include_chapters=[],
        include_instructions=include_instructions,
        include_few_shot=include_few_shot,
    )

    user_prompt = f"""
{rules_prompt}

This is the content of the scientific paper:

{markdown_text}

Evaluate it based on the schema and rules provided.
""".strip()

    system_prompt = build_system_prompt(expansion_key)

    raw = call_model(system_prompt, user_prompt)
    logger.debug("Raw model output: %s", raw[:1000])

    parsed = extract_json_from_model_output(raw)
    return parsed


# -------------------------------
# CSV helpers
# -------------------------------


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of flattened dict rows to a pandas DataFrame, ensuring consistent columns."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def append_rows_to_csv(rows: List[Dict[str, Any]], csv_path: str):
    df = rows_to_dataframe(rows)
    if df.empty:
        logger.info("No rows to write to CSV.")
        return
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), csv_path)


# -------------------------------
# Main orchestration
# -------------------------------


def process_all_pdfs(
    pdf_folder: str,
    csv_path: str,
    expansion_key: str,
    dry_run: bool = False,
    mode: str = "full",
) -> None:
    pdf_folder_path = Path(pdf_folder)
    if not pdf_folder_path.exists() or not pdf_folder_path.is_dir():
        raise ValueError(f"PDF folder does not exist: {pdf_folder}")

    pdf_files = sorted(
        [p for p in pdf_folder_path.iterdir() if p.suffix.lower() == ".pdf"]
    )
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_folder)
        return

    all_rows: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        logger.info("Processing: %s", pdf_path.name)
        md = parse_pdf_to_markdown(str(pdf_path))
        if not md:
            logger.warning(
                "Skipping %s because parsing produced no markdown.", pdf_path.name
            )
            continue

        if dry_run:
            # In dry-run we only collect metadata
            row = {"paper_name": pdf_path.name, "note": "dry-run, not evaluated"}
            all_rows.append(row)
            logger.info("Dry-run: parsed %s (no model call)", pdf_path.name)
            continue

        try:
            if mode == "full":
                eval_list = evaluate_markdown(md, expansion_key)
            elif mode == "sections":
                eval_list = evaluate_sections(md, expansion_key)
            else:
                logger.warning("Unknown mode '%s' — defaulting to 'full'.", mode)
                eval_list = evaluate_markdown(md, expansion_key)

            logger.info(
                "Eval list received for %s (items=%d)", pdf_path.name, len(eval_list)
            )
            # Convert list of rule objects into a single flattened row
            row: Dict[str, Any] = {"paper_name": pdf_path.name}
            for rule_obj in eval_list:
                rule_name = rule_obj.get("naziv_pravila")
                score = rule_obj.get("ocena")
                if rule_name is None:
                    logger.warning(
                        "Skipping item without 'naziv_pravila' in %s: %s",
                        pdf_path.name,
                        rule_obj,
                    )
                    continue
                row[rule_name] = score

            all_rows.append(row)
            logger.info(
                "Completed %s — collected %d fields", pdf_path.name, len(row) - 1
            )
            print(row)

            # polite pause to avoid rate limits
            time.sleep(1.0)

        except Exception as e:
            logger.exception("Failed to evaluate %s: %s", pdf_path.name, e)
            continue

    if all_rows:
        append_rows_to_csv(all_rows, csv_path)
    else:
        logger.info("No rows to write to CSV.")


# -------------------------------
# CLI
# -------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-evaluate PDFs using an LLM and a fixed rubric."
    )
    p.add_argument(
        "--pdf-folder", required=True, help="Folder containing PDF files to process"
    )
    p.add_argument("--out", required=True, help="CSV output path")
    p.add_argument(
        "--expansion",
        default="none",
        help="Which single expansion to include in the system prompt. One of: "
        + ",".join(EXPANSIONS.keys()),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, parse PDFs but do not call the model",
    )
    p.add_argument(
        "--mode",
        choices=["full", "sections"],
        default="full",
        help="Whether to evaluate the whole text in one prompt ('full') or evaluate global + each numbered chapter separately ('sections')",
    )
    return p.parse_args()


def main():
    args = parse_args()
    expansion = args.expansion
    if expansion not in EXPANSIONS:
        logger.warning("Unknown expansion '%s' — defaulting to 'none'", expansion)
        expansion = "none"

    try:
        process_all_pdfs(
            args.pdf_folder, args.out, expansion, dry_run=args.dry_run, mode=args.mode
        )

    except Exception as e:
        logger.exception("Fatal error: %s", e)


if __name__ == "__main__":
    main()
