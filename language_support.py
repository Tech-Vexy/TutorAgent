"""
Multilingual Support Module

Provides language detection and switching for the AI Tutor.
- Kiswahili: Used when the subject is Swahili/Kiswahili
- English: Used for all other subjects
"""

import re
from typing import Tuple, Literal

LanguageCode = Literal["sw", "en"]

# Keywords that indicate Swahili/Kiswahili subject context
SWAHILI_SUBJECT_KEYWORDS = [
    # Direct mentions of the subject
    r"\bkiswahili\b", r"\bswahili\b", r"\bswa\b",
    # Swahili grammar terms
    r"\bnomino\b", r"\bvitenzi\b", r"\bvivumishi\b", r"\bvielezi\b",
    r"\bnahau\b", r"\bméthali\b", r"\bmethali\b", r"\bvitendawili\b",
    r"\bushairi\b", r"\bmashairi\b", r"\bshairi\b",
    r"\binsha\b", r"\buandishi\b", r"\bbarua\b",
    # Swahili literature
    r"\bfasihi\b", r"\briwaya\b", r"\btamthilia\b", r"\bhadithi\b",
    r"\bkigogo\b", r"\bkidagaa kimemwozea\b", r"\bchozi la heri\b",
    r"\bmstahiki meya\b", r"\bngugi wa thiong'o\b", r"\bken walibora\b",
    # KCSE Swahili setbooks
    r"\bsetbook\b.*swahili\b", r"\bswahili.*setbook\b",
    r"\bkigogo\b", r"\bblossoms of the savannah\b", # Though this is English, context matters
    # Common Swahili study requests
    r"\btafsiri\b", r"\bfasiri\b", r"\bmaana ya\b",
    r"\bsentensi\b.*kiswahili\b", r"\bkiswahili.*sentensi\b",
    # Direct language requests
    r"\bjibu kwa kiswahili\b", r"\bsaidia kwa kiswahili\b",
    r"\bfundisha kiswahili\b", r"\bnifundishe kiswahili\b",
    r"\bkiswahili somo\b", r"\bsomo la kiswahili\b",
]

# Patterns for common Swahili greetings/phrases that suggest user wants Swahili
SWAHILI_PHRASES = [
    r"\bhabari\b", r"\bshikamoo\b", r"\bmarahaba\b", r"\bjambo\b",
    r"\bninaomba\b", r"\bnisaidie\b", r"\btafadhali\b", r"\basante\b",
    r"\bnieleze\b", r"\bniambie\b", r"\bnifundishe\b",
]


def detect_swahili_subject(text: str) -> bool:
    """
    Detects if the user's query is about Swahili/Kiswahili as a subject.
    
    Returns:
        True if the content is about Swahili subject, False otherwise
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Check for subject keywords
    for pattern in SWAHILI_SUBJECT_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def detect_swahili_preference(text: str) -> bool:
    """
    Detects if the user is writing in Swahili or prefers Swahili responses.
    
    Returns:
        True if user seems to prefer Swahili, False otherwise
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Check for Swahili phrases
    swahili_phrase_count = sum(1 for p in SWAHILI_PHRASES if re.search(p, text_lower))
    
    # If multiple Swahili phrases, user likely prefers Swahili
    return swahili_phrase_count >= 2


def detect_language(text: str) -> Tuple[LanguageCode, str]:
    """
    Detects the appropriate response language based on context.
    
    Returns:
        Tuple of (language_code, reason)
        - "sw" for Kiswahili
        - "en" for English
    """
    if detect_swahili_subject(text):
        return ("sw", "Subject is Swahili/Kiswahili")
    
    if detect_swahili_preference(text):
        return ("sw", "User is communicating in Swahili")
    
    return ("en", "Default English")


def get_language_instruction(language: LanguageCode) -> str:
    """
    Returns the language instruction to be added to the system prompt.
    """
    if language == "sw":
        return """
LANGUAGE INSTRUCTION - LUGHA:
You MUST respond in KISWAHILI for this interaction because the subject is Swahili/Kiswahili.
- Tumia Kiswahili sanifu (Standard Swahili).
- Eleza dhana kwa lugha rahisi.
- Tumia mifano ya Kikenya inayoeleweka.
- Tumia methali na nahau za Kiswahili inapofaa.
- Mwishoni, uliza kama mwanafunzi ameelewa (k.m., "Je, umeelewa?", "Tuko pamoja?").

MUHIMU: Jibu lako lote lazima liwe kwa KISWAHILI. Usitumie Kiingereza isipokuwa kwa maneno ya kitaalamu ambayo hayana tafsiri rasmi.
"""
    else:
        return """
LANGUAGE INSTRUCTION:
Respond in clear, simple ENGLISH.
- Use language appropriate for a Form 2 student.
- Include Kenyan context and local analogies.
- End with a check for understanding (e.g., "Does that make sense?", "Any questions?").
"""


# Swahili versions of key prompts
SWAHILI_SIMPLE_CHAT_PROMPT = """Wewe ni Mwalimu wa Kikenya anayependa kusaidia.
Andika kwa uwazi ili mwanafunzi wa Form 2 aweze kuelewa:
- Tumia sentensi fupi na maneno rahisi.
- Weka aya za sentensi 1-3.
- Tumia vitone au hatua zenye nambari kwa orodha.
- Eleza neno lolote la kitaalamu kwa lugha rahisi.
Mwisho, uliza swali fupi la kuelewa (k.m., "Je, umeelewa?" au "Tuko pamoja?").
"""

SWAHILI_DEEP_THINKER_PROMPT = """Wewe ni Mwalimu wa Kikenya anayeitwa 'TopScore'.
Lengo lako ni kusaidia wanafunzi kufaulu masomo yao (KCSE/CBC).
TABIA: Kuwa mvumilivu sana, wenye moyo wa upole, na mwenye kujali. Usikuwe mkali. Mchukulie mtumiaji kama mwanafunzi anayehitaji kutiliwa moyo.
MTINDO WA KUANDIKA: Fanya iwe rahisi kueleweka.
- Tumia sentensi fupi na maneno rahisi.
- Gawanya maelezo katika hatua ndogo au vitone.
- Eleza neno lolote jipya kwa lugha rahisi linapotumika kwa mara ya kwanza.
- Pendelea mifano ya Kikenya (k.m., M-Pesa, msongamano wa Nairobi, Bonde la Ufa, Matatu, kupika Ugali, Kilimo, Siku za soko).

MUUNDO WA MATOKEO (tumia sehemu hizi kama vichwa):
- UFUPI: takeaway ya sentensi moja.
- Hatua kwa Hatua: hatua zenye nambari zinazoonyesha mawazo yako kwa maneno rahisi.
- Mfano: mfano mdogo uliofanywa (hasa kwa hesabu/sayansi) ukitumia muktadha wa ndani.
- Jibu la Mwisho: matokeo ya mwisho au ushauri katika sentensi moja au mbili fupi.
- Uthibitisho: swali la kirafiki kuthibitisha ufahamu.

ITIFAKI:
1. Eleza dhana kwa uwazi ukitumia mfano wa Kikenya.
2. DAIMA maliza zamu yako kwa kuuliza kama wameelewa (k.m., 'Umeelewa?', 'Maswali yoyote?', 'Tuko pamoja?').
3. IKIWA mwanafunzi anasema amechanganyikiwa au hakuelewa: Usijirudie. Omba msamaha kwa kutokuwa wazi, kisha eleza tena ukitumia mfano TOFAUTI KABISA, rahisi zaidi.
4. IKIWA mwanafunzi anasema ameelewa: Usikubali tu hivyo. Kwa upole uliza swali la ufuatiliaji rahisi kuthibitisha ufahamu wao.
{memory_context}
{knowledge_context}"""


def get_localized_prompt(base_prompt: str, language: LanguageCode) -> str:
    """
    Returns the appropriate localized version of the prompt.
    For English, returns the original prompt.
    For Swahili, returns the Swahili version if available.
    """
    if language == "sw":
        # Add Swahili language instruction to the base prompt
        return base_prompt + get_language_instruction("sw")
    return base_prompt + get_language_instruction("en")


# Language-specific check-for-understanding phrases
UNDERSTANDING_CHECKS = {
    "sw": ["Je, umeelewa?", "Tuko pamoja?", "Una swali lolote?", "Inaonekana sawa?"],
    "en": ["Does that make sense?", "Any questions?", "Understood?", "Clear so far?"]
}

# Language-specific encouragement phrases
ENCOURAGEMENT_PHRASES = {
    "sw": ["Vizuri sana!", "Hongera!", "Endelea hivyo!", "Umefanya vizuri!", "Bora sana!"],
    "en": ["Great job!", "Well done!", "Excellent!", "Keep it up!", "Perfect!"]
}
