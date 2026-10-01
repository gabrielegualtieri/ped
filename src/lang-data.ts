/**
 * Data for the language/script detection in lang.ts: a port of the tables in laya/lang.py from the
 * `laya` Python package 0.3.22 by Convai Innovations (Apache License 2.0, see licenses/laya-LICENSE).
 * The word lists and ranges are copied unchanged; the reasons behind each exclusion are in the original.
 */

/** Unicode blocks the English checkpoint (ModernBERT-large, 50k English BPE) cannot read. */
export const SCRIPT_RANGES: readonly (readonly [string, readonly (readonly [number, number])[]])[] = [
  [
    "greek",
    [
      [0x0370, 0x03ff],
      [0x1f00, 0x1fff],
    ],
  ],
  [
    "cyrillic",
    [
      [0x0400, 0x052f],
      [0x2de0, 0x2dff],
      [0xa640, 0xa69f],
    ],
  ],
  ["armenian", [[0x0530, 0x058f]]],
  ["hebrew", [[0x0590, 0x05ff]]],
  [
    "arabic",
    [
      [0x0600, 0x06ff],
      [0x0750, 0x077f],
      [0x08a0, 0x08ff],
      [0xfb50, 0xfdff],
      [0xfe70, 0xfeff],
    ],
  ],
  [
    "devanagari",
    [
      [0x0900, 0x097f],
      [0xa8e0, 0xa8ff],
    ],
  ],
  ["bengali", [[0x0980, 0x09ff]]],
  ["gurmukhi", [[0x0a00, 0x0a7f]]],
  ["gujarati", [[0x0a80, 0x0aff]]],
  ["oriya", [[0x0b00, 0x0b7f]]],
  ["tamil", [[0x0b80, 0x0bff]]],
  ["telugu", [[0x0c00, 0x0c7f]]],
  ["kannada", [[0x0c80, 0x0cff]]],
  ["malayalam", [[0x0d00, 0x0d7f]]],
  ["sinhala", [[0x0d80, 0x0dff]]],
  ["thai", [[0x0e00, 0x0e7f]]],
  ["lao", [[0x0e80, 0x0eff]]],
  ["tibetan", [[0x0f00, 0x0fff]]],
  ["myanmar", [[0x1000, 0x109f]]],
  ["georgian", [[0x10a0, 0x10ff]]],
  ["ethiopic", [[0x1200, 0x137f]]],
  ["khmer", [[0x1780, 0x17ff]]],
  [
    "hangul",
    [
      [0x1100, 0x11ff],
      [0x3130, 0x318f],
      [0xac00, 0xd7af],
    ],
  ],
  [
    "kana",
    [
      [0x3040, 0x309f],
      [0x30a0, 0x30ff],
      [0x31f0, 0x31ff],
    ],
  ],
  [
    "han",
    [
      [0x3400, 0x4dbf],
      [0x4e00, 0x9fff],
      [0xf900, 0xfaff],
    ],
  ],
];

/**
 * Function words per language, weighted per hit; Latin-script languages overlap heavily, so detection
 * requires a margin and a word of the language's own. Order matters: ties go to the earlier list.
 */
const STOP_WORDS: readonly (readonly [string, string])[] = [
  ["en", "the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which " + "please we i"],
  // unaccented function words too: a state whose accents a mail client stripped keeps only these
  [
    "fr",
    "le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce et du au ou je tu il elle ils " +
      "elles mon ton ma ta sa mes tes ses ces deux trois très bien tout tous toute fait veux veut peux peut dois doit merci bonjour jour " +
      "jours mois fois quand comment pourquoi alors donc",
  ],
  // `in` and `was` are shared with English on purpose: counted for English alone, they outvoted short German
  [
    "de",
    "der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber ich wir mir mich dir " +
      "dich uns mein meine meinen meinem meiner diese dieser diesen dieses einen einem einer wie wo wann welche im zum zur aus bei nach " +
      "noch bitte heute jetzt kann kannst habe gibt wird in was",
  ],
  // `de`/`en` stay out: common English tokens (`de facto`, `en-US`) with nothing to weigh them against
  [
    "es",
    "el los las que por con para una es se del como pero son está este esta todo más muy hay sus la un y al lo le les su mi tu nos ni " +
      "dos tres fue fueron ser tiene tienen tengo puede pueden quiero necesito hemos han sobre entre cuando donde porque aunque también " +
      "ya eso esto esa ese nada algo aquí hoy gracias",
  ],
  // `no` stays out (English); Brazilian support text with unaccented forms and chat abbreviations (`vc`, `pra`)
  [
    "pt",
    "os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela o e na nas nos ao aos por foi era ser " +
      "sou tem tenho pode podem quero preciso eu meu minha seu sua isso isto aqui ali como quando onde porque mais já ainda agora hoje " +
      "ontem dois três tudo nada obrigado olá você vocês voce voces vc vcs nao sao ja até tá pra gostaria obrigada também tambem estou " +
      "estamos meus minhas nosso nossa consigo cadê boa tarde noite depois antes então entao ninguém ninguem alguém alguem nenhum nenhuma " +
      "estava ficou fiz deu",
  ],
  // the articulated prepositions are Italian-only, so a state of shared articles (`la fattura`) still names it
  [
    "it",
    "il lo gli che di per con non è si del della sono questo questa anche come più nella alla la le un uno una e ed o da su tra fra mi " +
      "ci ne ho hai ha abbiamo avete hanno era stato stata devo deve devono voglio vorrei mio mia tuo sua quando dove perche molto poco " +
      "sempre mai già ancora adesso oggi ieri grazie ciao scusa nel nell negli sul sulla sulle dal dalla dallo dagli dei delle dello " +
      "degli agli alle col",
  ],
  ["nl", "het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt"],
  // only words the Romance neighbours do not share; `la`, `o`, `un`, `de`, `pe`, `ca` stay out
  ["ro", "și să este sunt care pentru din dar după până fără ale lui în fost acum vreau trebuie foarte acest această acesta aceasta mi ți vă " + "nu"],
  // romanized Bangla ("Banglish"); words that are also English (`ache`, `to`, `age`, `eta`) stay out
  [
    "bn",
    "ami amar amake amra amader apni apnar apnake apnara tumi tomar tomake tomra tader ota eita oita ekta ei oi ki keno kivabe kibhabe " +
      "kothay kokhon kobe koto kintu jodi tahole ar theke jonno sathe shathe diye niye moddhe kore korte korchi korsi korbo korechi " +
      "koreche korun koren korlam hobe hoyeche hoise hocche hoyni chai chaina lagbe parchi parbo parchina peyechi paini dite dilam " +
      "diyechi nai khub onek ekhon akhon ekhono abar ekbar duibar ajke kalke taka bhalo valo kharap shomossa somossa dhonnobad bhai shob " +
      "keu kichu bolte bolun parben asbe jabe pabo ferot dorkar hoye geche gese",
  ],
  // `her`, `ne`, `men`, `de`, `ki` collide with other lists and stay out
  [
    "az",
    "və ve bir bu üçün ucun ilə ile olan olub olmasa var yox yoxdur mən sən biz siz onlar daha çox cox hər nə kimi görə sonra əgər eger " +
      "deyil lakin amma ancaq artıq artiq də isə həm yalnız yalniz",
  ],
];

export const STOP: ReadonlyMap<string, ReadonlySet<string>> = new Map(STOP_WORDS.map(([lg, words]) => [lg, new Set(words.split(" "))]));

/** Letters ordinary English does not use: catches a Latin-script language with no stopword list here. */
export const NON_EN_DIACRITICS: ReadonlySet<string> = new Set(
  "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ" + // Western European
    "ăâîșțşţ" + // Romanian
    "ąćęłńśźż" + // Polish
    "čďěňřšťůž" + // Czech / Slovak
    "őű" + // Hungarian
    "ğı" + // Turkish (text is lowercased before matching)
    "āēģīķļņūž" + // Baltic
    "đ" + // Serbo-Croatian / Vietnamese
    "ə", // Azerbaijani
);

/**
 * Code points with a non-zero canonical combining class (Python's `unicodedata.combining`, Unicode 15.0),
 * as hex ranges. JavaScript has no accessor for it; generated with
 * `[cp for cp in range(0x110000) if unicodedata.combining(chr(cp))]`.
 */
const COMBINING_RANGES =
  "300-34e 350-36f 483-487 591-5bd 5bf 5c1-5c2 5c4-5c5 5c7 610-61a 64b-65f 670 6d6-6dc 6df-6e4 6e7-6e8 6ea-6ed 711 730-74a 7eb-7f3 7fd 816-819 " +
  "81b-823 825-827 829-82d 859-85b 898-89f 8ca-8e1 8e3-8ff 93c 94d 951-954 9bc 9cd 9fe a3c a4d abc acd b3c b4d bcd c3c c4d c55-c56 cbc ccd " +
  "d3b-d3c d4d dca e38-e3a e48-e4b eb8-eba ec8-ecb f18-f19 f35 f37 f39 f71-f72 f74 f7a-f7d f80 f82-f84 f86-f87 fc6 1037 1039-103a 108d " +
  "135d-135f 1714-1715 1734 17d2 17dd 18a9 1939-193b 1a17-1a18 1a60 1a75-1a7c 1a7f 1ab0-1abd 1abf-1ace 1b34 1b44 1b6b-1b73 1baa-1bab 1be6 " +
  "1bf2-1bf3 1c37 1cd0-1cd2 1cd4-1ce0 1ce2-1ce8 1ced 1cf4 1cf8-1cf9 1dc0-1dff 20d0-20dc 20e1 20e5-20f0 2cef-2cf1 2d7f 2de0-2dff 302a-302f " +
  "3099-309a a66f a674-a67d a69e-a69f a6f0-a6f1 a806 a82c a8c4 a8e0-a8f1 a92b-a92d a953 a9b3 a9c0 aab0 aab2-aab4 aab7-aab8 aabe-aabf aac1 aaf6 " +
  "abed fb1e fe20-fe2f 101fd 102e0 10376-1037a 10a0d 10a0f 10a38-10a3a 10a3f 10ae5-10ae6 10d24-10d27 10eab-10eac 10efd-10eff 10f46-10f50 " +
  "10f82-10f85 11046 11070 1107f 110b9-110ba 11100-11102 11133-11134 11173 111c0 111ca 11235-11236 112e9-112ea 1133b-1133c 1134d 11366-1136c " +
  "11370-11374 11442 11446 1145e 114c2-114c3 115bf-115c0 1163f 116b6-116b7 1172b 11839-1183a 1193d-1193e 11943 119e0 11a34 11a47 11a99 11c3f " +
  "11d42 11d44-11d45 11d97 11f41-11f42 16af0-16af4 16b30-16b36 16ff0-16ff1 1bc9e 1d165-1d169 1d16d-1d172 1d17b-1d182 1d185-1d18b 1d1aa-1d1ad " +
  "1d242-1d244 1e000-1e006 1e008-1e018 1e01b-1e021 1e023-1e024 1e026-1e02a 1e08f 1e130-1e136 1e2ae 1e2ec-1e2ef 1e4ec-1e4ef 1e8d0-1e8d6 " +
  "1e944-1e94a";

export const COMBINING: readonly (readonly [number, number])[] = COMBINING_RANGES.split(" ").map((r) => {
  const [lo = "", hi = lo] = r.split("-");
  return [parseInt(lo, 16), parseInt(hi, 16)];
});
