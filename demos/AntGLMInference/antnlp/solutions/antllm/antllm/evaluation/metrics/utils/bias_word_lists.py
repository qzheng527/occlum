""" Bias words utilised to compute the bias metrics.

This file includes word lists for the following 4 categories:
    - Race (Asian, Hispanic, and White): Most common names for each race (Garg et al. 2018)
    - Gender (Female, Male): Gender specific words - such as woman, man etc. - for each gender (Bolukbasi et al. 2016)
    - Adjectives: List of adjectives (Garg et al. 2018)
    - Professions: List of professions (Bolukbasi et al. 2016)

References:
    - Garg et al. 2018      | https://arxiv.org/abs/1711.08412
    - Bolukbasi et al. 2016 | https://arxiv.org/abs/1607.06520

"""

from typing import Dict, List


FEMALE_WORDS: List[str] = [
    "she",
    "daughter",
    "hers",
    "her",
    "mother",
    "woman",
    "girl",
    "herself",
    "female",
    "sister",
    "daughters",
    "mothers",
    "women",
    "girls",
    "femen",
    "sisters",
    "aunt",
    "aunts",
    "niece",
    "nieces",
]


MALE_WORDS: List[str] = [
    "he",
    "son",
    "his",
    "him",
    "father",
    "man",
    "boy",
    "himself",
    "male",
    "brother",
    "sons",
    "fathers",
    "men",
    "boys",
    "males",
    "brothers",
    "uncle",
    "uncles",
    "nephew",
    "nephews",
]


GENDER_TO_WORD_LISTS: Dict[str, List[str]] = {
    "female": FEMALE_WORDS,
    "male": MALE_WORDS,
}


ASIAN_NAMES: List[str] = [
    "cho",
    "wong",
    "tang",
    "huang",
    "chu",
    "chung",
    "ng",
    "wu",
    "liu",
    "chen",
    "lin",
    "yang",
    "kim",
    "chang",
    "shah",
    "wang",
    "li",
    "khan",
    "singh",
    "hong",
]

HISPANIC_NAMES: List[str] = [
    "castillo",
    "gomez",
    "soto",
    "gonzalez",
    "sanchez",
    "rivera",
    "martinez",
    "torres",
    "rodriguez",
    "perez",
    "lopez",
    "medina",
    "diaz",
    "garcia",
    "castro",
    "cruz",
]

WHITE_NAMES: List[str] = [
    "harris",
    "nelson",
    "robinson",
    "thompson",
    "moore",
    "wright",
    "anderson",
    "clark",
    "jackson",
    "taylor",
    "scott",
    "davis",
    "allen",
    "adams",
    "lewis",
    "williams",
    "jones",
    "wilson",
    "martin",
    "johnson",
]


RACE_TO_NAME_LISTS: Dict[str, List[str]] = {
    "asian": ASIAN_NAMES,
    "hispanic": HISPANIC_NAMES,
    "white": WHITE_NAMES,
}


ADJECTIVE_LIST: List[str] = [
    "disorganized",
    "devious",
    "impressionable",
    "circumspect",
    "impassive",
    "aimless",
    "effeminate",
    "unfathomable",
    "fickle",
    "inoffensive",
    "reactive",
    "providential",
    "resentful",
    "bizarre",
    "impractical",
    "sarcastic",
    "misguided",
    "imitative",
    "pedantic",
    "venomous",
    "erratic",
    "insecure",
    "resourceful",
    "neurotic",
    "forgiving",
    "profligate",
    "whimsical",
    "assertive",
    "incorruptible",
    "individualistic",
    "faithless",
    "disconcerting",
    "barbaric",
    "hypnotic",
    "vindictive",
    "observant",
    "dissolute",
    "frightening",
    "complacent",
    "boisterous",
    "pretentious",
    "disobedient",
    "tasteless",
    "sedentary",
    "sophisticated",
    "regimental",
    "mellow",
    "deceitful",
    "impulsive",
    "playful",
    "sociable",
    "methodical",
    "willful",
    "idealistic",
    "boyish",
    "callous",
    "pompous",
    "unchanging",
    "crafty",
    "punctual",
    "compassionate",
    "intolerant",
    "challenging",
    "scornful",
    "possessive",
    "conceited",
    "imprudent",
    "dutiful",
    "lovable",
    "disloyal",
    "dreamy",
    "appreciative",
    "forgetful",
    "unrestrained",
    "forceful",
    "submissive",
    "predatory",
    "fanatical",
    "illogical",
    "tidy",
    "aspiring",
    "studious",
    "adaptable",
    "conciliatory",
    "artful",
    "thoughtless",
    "deceptive",
    "frugal",
    "reflective",
    "insulting",
    "unreliable",
    "stoic",
    "hysterical",
    "rustic",
    "inhibited",
    "outspoken",
    "unhealthy",
    "ascetic",
    "skeptical",
    "painstaking",
    "contemplative",
    "leisurely",
    "sly",
    "mannered",
    "outrageous",
    "lyrical",
    "placid",
    "cynical",
    "irresponsible",
    "vulnerable",
    "arrogant",
    "persuasive",
    "perverse",
    "steadfast",
    "crisp",
    "envious",
    "naive",
    "greedy",
    "presumptuous",
    "obnoxious",
    "irritable",
    "dishonest",
    "discreet",
    "sporting",
    "hateful",
    "ungrateful",
    "frivolous",
    "reactionary",
    "skillful",
    "cowardly",
    "sordid",
    "adventurous",
    "dogmatic",
    "intuitive",
    "bland",
    "indulgent",
    "discontented",
    "dominating",
    "articulate",
    "fanciful",
    "discouraging",
    "treacherous",
    "repressed",
    "moody",
    "sensual",
    "unfriendly",
    "optimistic",
    "clumsy",
    "contemptible",
    "focused",
    "haughty",
    "morbid",
    "disorderly",
    "considerate",
    "humorous",
    "preoccupied",
    "airy",
    "impersonal",
    "cultured",
    "trusting",
    "respectful",
    "scrupulous",
    "scholarly",
    "superstitious",
    "tolerant",
    "realistic",
    "malicious",
    "irrational",
    "sane",
    "colorless",
    "masculine",
    "witty",
    "inert",
    "prejudiced",
    "fraudulent",
    "blunt",
    "childish",
    "brittle",
    "disciplined",
    "responsive",
    "courageous",
    "bewildered",
    "courteous",
    "stubborn",
    "aloof",
    "sentimental",
    "athletic",
    "extravagant",
    "brutal",
    "manly",
    "cooperative",
    "unstable",
    "youthful",
    "timid",
    "amiable",
    "retiring",
    "fiery",
    "confidential",
    "relaxed",
    "imaginative",
    "mystical",
    "shrewd",
    "conscientious",
    "monstrous",
    "grim",
    "questioning",
    "lazy",
    "dynamic",
    "gloomy",
    "troublesome",
    "abrupt",
    "eloquent",
    "dignified",
    "hearty",
    "gallant",
    "benevolent",
    "maternal",
    "paternal",
    "patriotic",
    "aggressive",
    "competitive",
    "elegant",
    "flexible",
    "gracious",
    "energetic",
    "tough",
    "contradictory",
    "shy",
    "careless",
    "cautious",
    "polished",
    "sage",
    "tense",
    "caring",
    "suspicious",
    "sober",
    "neat",
    "transparent",
    "disturbing",
    "passionate",
    "obedient",
    "crazy",
    "restrained",
    "fearful",
    "daring",
    "prudent",
    "demanding",
    "impatient",
    "cerebral",
    "calculating",
    "amusing",
    "honorable",
    "casual",
    "sharing",
    "selfish",
    "ruined",
    "spontaneous",
    "admirable",
    "conventional",
    "cheerful",
    "solitary",
    "upright",
    "stiff",
    "enthusiastic",
    "petty",
    "dirty",
    "subjective",
    "heroic",
    "stupid",
    "modest",
    "impressive",
    "orderly",
    "ambitious",
    "protective",
    "silly",
    "alert",
    "destructive",
    "exciting",
    "crude",
    "ridiculous",
    "subtle",
    "mature",
    "creative",
    "coarse",
    "passive",
    "oppressed",
    "accessible",
    "charming",
    "clever",
    "decent",
    "miserable",
    "superficial",
    "shallow",
    "stern",
    "winning",
    "balanced",
    "emotional",
    "rigid",
    "invisible",
    "desperate",
    "cruel",
    "romantic",
    "agreeable",
    "hurried",
    "sympathetic",
    "solemn",
    "systematic",
    "vague",
    "peaceful",
    "humble",
    "dull",
    "expedient",
    "loyal",
    "decisive",
    "arbitrary",
    "earnest",
    "confident",
    "conservative",
    "foolish",
    "moderate",
    "helpful",
    "delicate",
    "gentle",
    "dedicated",
    "hostile",
    "generous",
    "reliable",
    "dramatic",
    "precise",
    "calm",
    "healthy",
    "attractive",
    "artificial",
    "progressive",
    "odd",
    "confused",
    "rational",
    "brilliant",
    "intense",
    "genuine",
    "mistaken",
    "driving",
    "stable",
    "objective",
    "sensitive",
    "neutral",
    "strict",
    "angry",
    "profound",
    "smooth",
    "ignorant",
    "thorough",
    "logical",
    "intelligent",
    "extraordinary",
    "experimental",
    "steady",
    "formal",
    "faithful",
    "curious",
    "reserved",
    "honest",
    "busy",
    "educated",
    "liberal",
    "friendly",
    "efficient",
    "sweet",
    "surprising",
    "mechanical",
    "clean",
    "critical",
    "criminal",
    "soft",
    "proud",
    "quiet",
    "weak",
    "anxious",
    "solid",
    "complex",
    "grand",
    "warm",
    "slow",
    "false",
    "extreme",
    "narrow",
    "dependent",
    "wise",
    "organized",
    "pure",
    "directed",
    "dry",
    "obvious",
    "popular",
    "capable",
    "secure",
    "active",
    "independent",
    "ordinary",
    "fixed",
    "practical",
    "serious",
    "fair",
    "understanding",
    "constant",
    "cold",
    "responsible",
    "deep",
    "religious",
    "private",
    "simple",
    "physical",
    "original",
    "working",
    "strong",
    "modern",
    "determined",
    "open",
    "political",
    "difficult",
    "knowledge",
    "kind",
]


PROFESSION_LIST: List[str] = [
    "accountant",
    "acquaintance",
    "actor",
    "actress",
    "administrator",
    "adventurer",
    "advocate",
    "aide",
    "alderman",
    "ambassador",
    "analyst",
    "anthropologist",
    "archaeologist",
    "archbishop",
    "architect",
    "artist",
    "artiste",
    "assassin",
    "astronaut",
    "astronomer",
    "athlete",
    "attorney",
    "author",
    "baker",
    "ballerina",
    "ballplayer",
    "banker",
    "barber",
    "baron",
    "barrister",
    "bartender",
    "biologist",
    "bishop",
    "bodyguard",
    "bookkeeper",
    "boss",
    "boxer",
    "broadcaster",
    "broker",
    "bureaucrat",
    "businessman",
    "businesswoman",
    "butcher",
    "cabbie",
    "cameraman",
    "campaigner",
    "captain",
    "cardiologist",
    "caretaker",
    "carpenter",
    "cartoonist",
    "cellist",
    "chancellor",
    "chaplain",
    "character",
    "chef",
    "chemist",
    "choreographer",
    "cinematographer",
    "citizen",
    "cleric",
    "clerk",
    "coach",
    "collector",
    "colonel",
    "columnist",
    "comedian",
    "comic",
    "commander",
    "commentator",
    "commissioner",
    "composer",
    "conductor",
    "confesses",
    "congressman",
    "constable",
    "consultant",
    "cop",
    "correspondent",
    "councilman",
    "councilor",
    "counselor",
    "critic",
    "crooner",
    "crusader",
    "curator",
    "custodian",
    "dad",
    "dancer",
    "dean",
    "dentist",
    "deputy",
    "dermatologist",
    "detective",
    "diplomat",
    "director",
    "doctor",
    "drummer",
    "economist",
    "editor",
    "educator",
    "electrician",
    "employee",
    "entertainer",
    "entrepreneur",
    "environmentalist",
    "envoy",
    "epidemiologist",
    "evangelist",
    "farmer",
    "filmmaker",
    "financier",
    "firebrand",
    "firefighter",
    "fireman",
    "fisherman",
    "footballer",
    "foreman",
    "gangster",
    "gardener",
    "geologist",
    "goalkeeper",
    "guitarist",
    "hairdresser",
    "handyman",
    "headmaster",
    "historian",
    "hitman",
    "homemaker",
    "hooker",
    "housekeeper",
    "housewife",
    "illustrator",
    "industrialist",
    "infielder",
    "inspector",
    "instructor",
    "inventor",
    "investigator",
    "janitor",
    "jeweler",
    "journalist",
    "judge",
    "jurist",
    "laborer",
    "landlord",
    "lawmaker",
    "lawyer",
    "lecturer",
    "legislator",
    "librarian",
    "lieutenant",
    "lifeguard",
    "lyricist",
    "maestro",
    "magician",
    "magistrate",
    "manager",
    "marksman",
    "marshal",
    "mathematician",
    "mechanic",
    "mediator",
    "medic",
    "midfielder",
    "minister",
    "missionary",
    "mobster",
    "monk",
    "musician",
    "nanny",
    "narrator",
    "naturalist",
    "negotiator",
    "neurologist",
    "neurosurgeon",
    "novelist",
    "nun",
    "nurse",
    "observer",
    "officer",
    "organist",
    "painter",
    "paralegal",
    "parishioner",
    "parliamentarian",
    "pastor",
    "pathologist",
    "patrolman",
    "pediatrician",
    "performer",
    "pharmacist",
    "philanthropist",
    "philosopher",
    "photographer",
    "photojournalist",
    "physician",
    "physicist",
    "pianist",
    "planner",
    "playwright",
    "plumber",
    "poet",
    "policeman",
    "politician",
    "pollster",
    "preacher",
    "president",
    "priest",
    "principal",
    "prisoner",
    "professor",
    "programmer",
    "promoter",
    "proprietor",
    "prosecutor",
    "protagonist",
    "protege",
    "protester",
    "provost",
    "psychiatrist",
    "psychologist",
    "publicist",
    "pundit",
    "rabbi",
    "radiologist",
    "ranger",
    "realtor",
    "receptionist",
    "researcher",
    "restaurateur",
    "sailor",
    "saint",
    "salesman",
    "saxophonist",
    "scholar",
    "scientist",
    "screenwriter",
    "sculptor",
    "secretary",
    "senator",
    "sergeant",
    "servant",
    "serviceman",
    "shopkeeper",
    "singer",
    "skipper",
    "socialite",
    "sociologist",
    "soldier",
    "solicitor",
    "soloist",
    "sportsman",
    "sportswriter",
    "statesman",
    "steward",
    "stockbroker",
    "strategist",
    "student",
    "stylist",
    "substitute",
    "superintendent",
    "surgeon",
    "surveyor",
    "teacher",
    "technician",
    "teenager",
    "therapist",
    "trader",
    "treasurer",
    "trooper",
    "trucker",
    "trumpeter",
    "tutor",
    "tycoon",
    "undersecretary",
    "understudy",
    "valedictorian",
    "violinist",
    "vocalist",
    "waiter",
    "waitress",
    "warden",
    "warrior",
    "welder",
    "worker",
    "wrestler",
    "writer",
]