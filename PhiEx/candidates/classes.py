"""
PhiEx.candidates.classes — what can (and cannot) be asked of a candidate gene.

A transcriptomics candidate list mixes enzymes with known chemistry, binding
proteins, transcription factors, transporters, transposon-derived ORFs and
proteins of unknown function.  Docking only makes sense for the first two.
Each rule below maps an annotation (UniProt/InterPro description + CAZy
family) to:

    track     "dock"        enzyme / binding protein with a defined small-
                            molecule partner → dock the panel into its site
              "structure"   no known ligand → structure, pockets, catalytic
                            motifs, structural neighbours (Foldseek) to build
                            a function hypothesis
              "unsuited"    DNA/RNA-binding, membrane transporters, protein–
                            protein binders, transposon proteins: docking a
                            small molecule would produce meaningless numbers
    cofactor  what the active site needs to be complete (heme, Cu, FAD,
              NAD(P), PLP, Fe, Mn, SAM …) → prefer an experimental structure
              or an AlphaFill model that carries it
    panel     ligands with a role: substrate / product / inhibitor /
              cofactor / control.  Every dock panel has a negative control so
              the result is a *margin* (best binder − control), not a raw
              Vina score.

Ligands are SMILES where the structure is simple and unambiguous, and
PubChem names otherwise (resolved at run time by vina.resolve_ligand_input).
Rules are matched in order; the first hit wins.  They are deliberately
explicit so they can be read, argued with and edited.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class LigandSpec:
    label: str
    value: str           # SMILES or PubChem name
    role: str            # substrate | product | inhibitor | cofactor | control


@dataclass
class CandidateClass:
    key: str
    title: str
    track: str                          # dock | structure | unsuited
    pattern: str                        # regex on "description | family" (lower-case)
    cofactor: str = ""                  # what a complete site needs
    panel: list = field(default_factory=list)
    note: str = ""

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text) is not None


def L(label, value, role="substrate"):
    return LigandSpec(label, value, role)


# shared controls
GLC = L("D-glucose (control)", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "control")
GLY = L("glycerol (control)", "OCC(O)CO", "control")
# Default negative control: D-glucose (12 heavy atoms) is closer in size to
# typical substrates than glycerol (6).  Vina scores grow with size, so a
# tiny control inflates every margin; glucose keeps the comparison honest.
URE = L("urea (control)", "NC(N)=O", "control")

PHENOLICS = [
    L("guaiacol", "COc1ccccc1O"),
    L("2,6-dimethoxyphenol", "COc1cccc(OC)c1O"),
    L("coniferyl alcohol", "COc1cc(/C=C/CO)ccc1O"),
    L("sinapyl alcohol", "COc1cc(/C=C/CO)cc(OC)c1O"),
    L("p-coumaryl alcohol", "OC/C=C/c1ccc(O)cc1"),
    L("ferulic acid", "COc1cc(/C=C/C(=O)O)ccc1O"),
]
ALDEHYDES = [
    L("coniferaldehyde", "COc1cc(/C=C/C=O)ccc1O"),
    L("sinapaldehyde", "COc1cc(/C=C/C=O)cc(OC)c1O"),
    L("cinnamaldehyde", "O=C/C=C/c1ccccc1"),
    L("benzaldehyde", "O=Cc1ccccc1"),
    L("furfural", "O=Cc1ccco1"),
    L("5-hydroxymethylfurfural", "OCc1ccc(C=O)o1"),
]
ESTERS = [
    L("p-nitrophenyl acetate", "CC(=O)Oc1ccc(cc1)[N+](=O)[O-]"),
    L("p-nitrophenyl butyrate", "CCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]"),
    L("p-nitrophenyl octanoate", "CCCCCCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]"),
    L("triacetin", "CC(=O)OCC(COC(C)=O)OC(C)=O"),
    L("10,16-dihydroxyhexadecanoate (cutin monomer)", "OCCCCCCC(O)CCCCCCCCC(=O)O"),
]
ISOFLAVONOIDS = [
    L("daidzein", "O=c1c(-c2ccc(O)cc2)coc2cc(O)ccc12"),
    L("genistein", "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12"),
    L("formononetin", "COc1ccc(-c2coc3cc(O)ccc3c2=O)cc1"),
    L("glyceollin I", "glyceollin I"),
    L("medicarpin", "medicarpin"),
]
NUCLEOTIDES = [
    L("ATP", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
    L("ADP", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
    L("adenosine (control)", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "control"),
]


RULES: list[CandidateClass] = [
    # ── not suitable for small-molecule docking ─────────────────────────
    CandidateClass("transposon", "transposon-derived protein", "unsuited",
        r"reverse transcriptase|rna-directed dna polymerase|ribonuclease h|rnase h|integrase|"
        r"cenp-?b|transpos|retrotrans",
        note="transposable-element origin; re-quantify with a repeat-aware method before interpreting"),
    CandidateClass("tf", "transcription factor / DNA-binding", "unsuited",
        r"transcription factor|wrky|myb|nac domain|nac transcription|bhlh|ap2/erf|ethylene-responsive|"
        r"zinc finger|c2h2|dna binding|dna-binding|hth |tubby|vq motif|sar deficient|calmodulin-binding protein 60|"
        r"protein ataf|nucleic acid binding",
        note="acts through DNA/protein interactions; test by expression, ChIP or reporter assays, not docking"),
    CandidateClass("transporter", "membrane transporter / channel", "unsuited",
        r"transporter|permease|carrier protein|abc-?2? |abc transporter|major facilitator|mfs|channel|sweet|"
        r"efflux|mate |detoxification \d|nrt1|translocator|cotransporter|aquaporin|atpase, |phospholipid-transporting",
        note="membrane protein: an AlphaFold monomer in water is not its working state; substrate docking "
             "into a transporter needs the full transporter in a membrane model"),
    CandidateClass("ppi", "protein–protein binder / inhibitor / receptor ectodomain", "unsuited",
        r"kunitz|polygalacturonase inhibitor|pectinesterase inhibitor|leucine-rich repeat-containing n-terminal|"
        r"lrrnt|expansin|eg45|harpin|hydrophobin|cerato|small polypeptide|precursor of cep|"
        r"mlo-like|casp-like",
        note="works by binding proteins or cell-wall polymers; use the structure for interface analysis"),

    # ── fungal effectors / unknowns → structure track ───────────────────
    CandidateClass("effector_unknown", "predicted effector / unknown function", "structure",
        r"predicted effector|effector|uncharacteri[sz]ed|hypothetical protein|unknown function|duf\d|"
        r"signal peptide$|small secreted|wsc domain|gpi anchored|cfem|ecp2|phia|panther pthr|"
        r"fungal n-terminal|cyanovirin|^catalytic activity$|^electron transport$",
        note="no known ligand: use structure, pockets and structural neighbours (Foldseek) for a function "
             "hypothesis; AlphaFold confidence decides how far the structure can be trusted"),

    # ── redox enzymes with phenolic substrates ──────────────────────────
    CandidateClass("laccase", "laccase / multicopper oxidase (AA1)", "dock",
        r"laccase|multicopper oxidase|\baa1(_\d)?\b",
        cofactor="Cu (T1 + trinuclear cluster) — use AlphaFill",
        panel=[*PHENOLICS[:5], L("syringaldazine", "syringaldazine"),
               L("ABTS", "2,2'-azino-bis(3-ethylbenzothiazoline-6-sulfonic acid)"), GLC],
        note="substrates bind at the T1 copper; without Cu the pocket shape is still informative"),
    CandidateClass("peroxidase", "heme peroxidase (class III / AA2)", "dock",
        r"peroxidase|\baa2\b",
        cofactor="heme + Ca — use AlphaFill or an experimental structure",
        panel=[*PHENOLICS, L("L-ascorbate", "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"), GLC],
        note="monolignol oxidation = lignification/cross-linking"),
    CandidateClass("p450", "cytochrome P450 monooxygenase", "dock",
        r"cytochrome p450|p450|isoflavone 2'-hydroxylase|8'-hydroxylase|cytokinin hydroxylase|"
        r"ferruginol synthase|monooxygenase activity; heme",
        cofactor="heme — use AlphaFill",
        panel=[*ISOFLAVONOIDS, L("tebuconazole (azole inhibitor)", "tebuconazole", "inhibitor"),
               L("(+)-abscisic acid", "CC1=CC(=O)C[C@@](C)(C)[C@]1(O)/C=C/C(C)=C\\C(=O)O"), GLC],
        note="phytoalexin detoxification (glyceollin, isoflavonoids) and azole binding"),
    CandidateClass("dehydrogenase", "NAD(P)-dependent dehydrogenase / reductase", "dock",
        r"alcohol dehydrogenase|cinnamyl alcohol|aldo-keto reductase|short-chain dehydrogenase|"
        r"nad\(p\)-binding|rossmann|enoyl ?reductase|enoylreductase|11-beta-hydroxysteroid|"
        r"nadph dehydrogenase|oxidoreductas; alcohol",
        cofactor="NAD(P)H — use AlphaFill",
        panel=[*ALDEHYDES, GLC],
        note="aromatic and furan aldehydes: monolignol and lignocellulose-derived substrates"),
    CandidateClass("thn_reductase", "tetrahydroxynaphthalene reductase (melanin)", "dock",
        r"tetrahydroxynaphthalene reductase",
        cofactor="NADPH — use AlphaFill",
        panel=[L("1,3,6,8-tetrahydroxynaphthalene", "Oc1cc(O)c2cc(O)cc(O)c2c1"),
               L("scytalone", "scytalone", "product"),
               L("tricyclazole (MBI-R inhibitor)", "Cc1cccc2sc3nncn3c12", "inhibitor"), GLC]),
    CandidateClass("gmc_oxidase", "GMC / FAD-dependent oxidoreductase (AA3, AA7, BBE-like)", "dock",
        r"glucose-methanol-choline|gmc oxidoreductase|\baa3(_\d)?\b|\baa7\b|fad-dependent oxidoreductase|"
        r"berberine bridge|glucooligosaccharide oxidase|fad binding",
        cofactor="FAD — use AlphaFill",
        panel=[L("D-glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
               L("cellobiose", "cellobiose"),
               L("veratryl alcohol", "COc1ccc(CO)cc1OC"),
               L("trigalacturonic acid (OG)", "trigalacturonic acid"), URE]),
    CandidateClass("cro", "copper-radical oxidase (AA5)", "dock",
        r"\baa5(_\d)?\b|copper radical|galactose oxidase|glyoxal oxidase",
        cofactor="Cu — use AlphaFill",
        panel=[L("D-galactose", "D-galactose"), L("methylglyoxal", "CC(=O)C=O"),
               L("glycolaldehyde", "OCC=O"), L("raffinose", "raffinose"), URE]),
    CandidateClass("lpmo", "lytic polysaccharide monooxygenase (AA9–AA16)", "dock",
        r"\baa(9|10|11|13|14|16)\b|cbm20",
        cofactor="Cu — use AlphaFill",
        panel=[L("maltotriose", "maltotriose"), L("cellotriose", "cellotriose"), GLC],
        note="flat binding surface: docking scores are weak by nature; look at contacts"),

    # ── hydrolases ──────────────────────────────────────────────────────
    CandidateClass("cutinase_esterase", "cutinase / carboxylesterase / lipase (serine hydrolase)", "dock",
        r"cutinase|\bce5\b|\bce16\b|\bce2\b|sgnh|gdsl|esterase|lipase|alpha/beta hydrolase|alpha_beta-hydrolase|ab hydrolase|"
        r"dienelactone|carboxylesterase|acetylhydrolase|1-alkyl-2-acetylglycerophosphocholine",
        cofactor="none (Ser–His–Asp/Glu triad)",
        panel=[*ESTERS, GLC]),
    CandidateClass("epoxide_hydrolase", "epoxide hydrolase", "dock",
        r"epoxide hydrolase|stilbene-oxide",
        panel=[L("cis-stilbene oxide", "c1ccc(cc1)[C@@H]1O[C@@H]1c1ccccc1"),
               L("styrene oxide", "C1OC1c1ccccc1"),
               L("9,10-epoxystearic acid", "CCCCCCCCC1OC1CCCCCCCC(=O)O"), GLC]),
    CandidateClass("pectinase", "pectin-active enzyme (GH28, CE8, PL)", "dock",
        r"\bgh28\b|polygalacturonase|pectinesterase|pectin methylesterase|\bce8\b|pectate lyase(?!.*gh55)|"
        r"\bpl\d|pectin acetylesterase",
        panel=[L("D-galacturonic acid", "D-galacturonic acid"),
               L("digalacturonic acid", "digalacturonic acid"),
               L("trigalacturonic acid", "trigalacturonic acid"), GLC]),
    CandidateClass("b13_glucanase", "β-1,3-glucan-active enzyme (GH16/17/55/64/81/128, GH5 exo-1,3)", "dock",
        r"\bgh(16|17|55|64|81|128)(_\d+)?\b|1,3-beta|beta-1,3|endo-1,3|exo-1,3|laminarin|asl1-like",
        panel=[L("laminaribiose", "laminaribiose"), L("laminaritriose", "laminaritriose"),
               L("gentiobiose (β-1,6)", "gentiobiose"),
               L("cellobiose (β-1,4, control)", "cellobiose", "control")],
        note="β-1,3-glucan is the fungal wall polymer and, in plants, callose"),
    CandidateClass("mannosidase", "α-mannosidase (GH47 / GH92)", "dock",
        r"\bgh47\b|\bgh92\b|mannosidase",
        cofactor="Ca2+ (GH47) — use AlphaFill",
        panel=[L("D-mannose", "D-mannose"), L("α-1,2-mannobiose", "2-O-alpha-D-mannopyranosyl-D-mannose"),
               L("kifunensine (GH47 inhibitor)", "kifunensine", "inhibitor"),
               L("1-deoxymannojirimycin", "1-deoxymannojirimycin", "inhibitor"), GLC]),
    CandidateClass("chitinase", "chitin-active enzyme (GH18 / GH19 / GH20)", "dock",
        r"chitinase|\bgh18\b|\bgh19\b|\bgh20\b|hexosaminidase|glycoside hydrolase family 19",
        panel=[L("N-acetyl-D-glucosamine", "N-acetyl-D-glucosamine"), L("chitobiose", "chitobiose"),
               L("chitotriose", "chitotriose"), L("allosamidin (GH18 inhibitor)", "allosamidin", "inhibitor"),
               L("cellobiose (control)", "cellobiose", "control")]),
    CandidateClass("lysm", "LysM / CBM50 chitin-binding protein", "dock",
        r"lysm|\bcbm50\b",
        panel=[L("chitotriose", "chitotriose"), L("chitotetraose", "chitotetraose"),
               L("N-acetyl-D-glucosamine", "N-acetyl-D-glucosamine"),
               L("cellotriose (control)", "cellotriose", "control")],
        note="LysM effectors sequester chitin oligomers (chitin-shielding)"),
    CandidateClass("hemicellulase", "hemicellulose / side-chain glycosidase", "dock",
        r"xylanase|\bgh11\b|\bgh10\b|arabin|\bgh43\b|\bgh54\b|\bgh53\b|galactanase|rhamnosidase|\bgh78\b|"
        r"alpha-galactosidase|beta-galactosidase|xylosidase|fructofuranosidase|xyloglucan|\bgh114\b|\bgh154\b",
        panel=[L("xylobiose", "xylobiose"), L("L-arabinose", "L-arabinose"), L("L-rhamnose", "L-rhamnose"),
               L("D-galactose", "D-galactose"), L("melibiose", "melibiose"), GLC],
        note="broad panel — narrow it to the family's linkage once the target is chosen"),
    CandidateClass("cellulase", "cellulase / β-glucosidase (GH1, GH3, GH5, GH12)", "dock",
        r"cellulase|endoglucanase|\bgh(1|3|5|12)(_\d+)?\b(?!.*auxin)|beta-glucosidase|beta glucosidase|cellobiose glucosidase",
        panel=[L("cellobiose", "cellobiose"), L("cellotriose", "cellotriose"),
               L("gentiobiose", "gentiobiose"), GLC]),
    CandidateClass("amylase", "α-glucan-active enzyme (GH13, GT35)", "dock",
        r"amylase|\bgh13\b|glucan-branching|\bgt35\b|glucan phosphorylase|aamy",
        cofactor="Ca2+ (GH13) / PLP (GT35) — use AlphaFill",
        panel=[L("maltose", "maltose"), L("maltotriose", "maltotriose"),
               L("acarbose (inhibitor)", "acarbose", "inhibitor"), L("cellobiose (control)", "cellobiose", "control")]),
    CandidateClass("peptidase", "peptidase", "dock",
        r"peptidase|protease|oligopeptidase|endopeptidase",
        panel=[L("Gly-Pro", "NCC(=O)N1CCC[C@H]1C(=O)O"), L("Ala-Pro", "C[C@H](N)C(=O)N1CCC[C@H]1C(=O)O"),
               L("Z-Pro-prolinal (POP inhibitor)", "Z-Pro-prolinal", "inhibitor"),
               L("pepstatin A (aspartic inhibitor)", "pepstatin A", "inhibitor"), GLC],
        note="peptide substrates are long and flexible; trust contacts more than scores"),
    CandidateClass("ceramidase", "ceramidase / sphingolipid hydrolase", "dock",
        r"ceramidase",
        cofactor="Zn — use AlphaFill",
        panel=[L("C2-ceramide", "CC(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCCCCC"),
               L("sphingosine", "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO", "product"),
               L("palmitic acid", "CCCCCCCCCCCCCCCC(=O)O"), GLC],
        note="sphingolipid signalling and cell death"),

    # ── lectins ─────────────────────────────────────────────────────────
    CandidateClass("lectin_fucose", "fucose-specific lectin (Ssp1-like)", "dock",
        r"fucose",
        panel=[L("L-fucose", "L-fucose"), L("methyl α-L-fucoside", "methyl alpha-L-fucopyranoside"),
               L("2'-fucosyllactose", "2'-fucosyllactose"),
               L("D-galactose (control: fucose = 6-deoxy-galactose)", "D-galactose", "control"), GLC]),
    CandidateClass("lectin_gal", "GalNAc/Gal lectin (S. sclerotiorum agglutinin, β-trefoil)", "dock",
        r"agglutinin|lectin(?!.*receptor)|ricin",
        panel=[L("N-acetyl-D-galactosamine", "N-acetyl-D-galactosamine"), L("D-galactose", "D-galactose"),
               L("lactose", "lactose"), L("L-fucose", "L-fucose"),
               L("D-mannose (control)", "D-mannose", "control"), GLC],
        note="SSA has a crystal structure with GalNAc (Sulzenbacher et al. 2010) → re-docking validates"),

    # ── sterol / isoprenoid ─────────────────────────────────────────────
    CandidateClass("sterol_mt", "sterol 24-C-methyltransferase (ERG6)", "dock",
        r"sterol 24-c-methyltransferase|erg6",
        cofactor="SAM — use AlphaFill",
        panel=[L("zymosterol", "zymosterol"), L("lanosterol", "lanosterol"), L("ergosterol", "ergosterol", "product"),
               L("S-adenosyl-L-methionine", "S-adenosyl-L-methionine", "cofactor"), GLC]),
    CandidateClass("osc", "oxidosqualene cyclase (lanosterol synthase, ERG7)", "dock",
        r"lanosterol synthase|oxidosqualene cyclase",
        panel=[L("(S)-2,3-oxidosqualene", "2,3-oxidosqualene"), L("lanosterol", "lanosterol", "product"),
               L("Ro 48-8071 (inhibitor)", "Ro 48-8071", "inhibitor"), GLC]),
    CandidateClass("squalene_epoxidase", "squalene monooxygenase (ERG1)", "dock",
        r"squalene monooxygenase|squalene epoxidase",
        cofactor="FAD — use AlphaFill",
        panel=[L("squalene", "squalene"), L("terbinafine (inhibitor)", "terbinafine", "inhibitor"), GLC]),
    CandidateClass("diterpene_synthase", "terpene / prenyl transferase", "dock",
        r"copalyl diphosphate|kaurene|terpene synthase|phytyltransferase|geranylgeranyltransferase|ubia",
        cofactor="Mg2+",
        panel=[L("geranylgeranyl diphosphate", "geranylgeranyl diphosphate"),
               L("homogentisic acid", "OC(=O)Cc1cc(O)ccc1O"), GLC]),

    # ── respiration (pathogen) ──────────────────────────────────────────
    CandidateClass("sdh", "succinate dehydrogenase subunit (SDHI target)", "dock",
        r"succinate dehydrogenase",
        cofactor="complex II: the SDHI/ubiquinone site is formed by SdhB, SdhC and SdhD",
        panel=[L("boscalid", "boscalid", "inhibitor"), L("fluopyram", "fluopyram", "inhibitor"),
               L("carboxin", "carboxin", "inhibitor"), L("ubiquinone-1", "ubiquinone-1"), GLC],
        note="a single subunit cannot form the SDHI pocket — model the SdhB/C/D complex (AlphaFold-Multimer) "
             "before trusting any score"),
    CandidateClass("aox", "alternative oxidase", "dock",
        r"alternative oxidase|f420h2 dehydrogenase",
        cofactor="di-iron — use AlphaFill (trypanosome AOX structures carry Fe)",
        panel=[L("salicylhydroxamic acid (SHAM)", "ONC(=O)c1ccccc1O", "inhibitor"),
               L("ascofuranone", "ascofuranone", "inhibitor"), L("n-propyl gallate", "CCCOC(=O)c1cc(O)c(O)c(O)c1", "inhibitor"),
               L("ubiquinol-1", "ubiquinol-1"), GLC],
        note="membrane-associated diiron enzyme; scores are qualitative"),

    # ── host defence chemistry ──────────────────────────────────────────
    CandidateClass("pal", "phenylalanine ammonia-lyase", "dock",
        r"phenylalanine ammonia-lyase",
        cofactor="MIO (post-translational Ala-Ser-Gly) — absent from predicted models",
        panel=[L("L-phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O"), L("L-tyrosine", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
               L("trans-cinnamic acid", "OC(=O)/C=C/c1ccccc1", "product"),
               L("AIP (2-aminoindan-2-phosphonic acid)", "NC1(Cc2ccccc2C1)P(=O)(O)O", "inhibitor"), GLC]),
    CandidateClass("chs", "chalcone synthase (type III PKS)", "dock",
        r"chalcone synthase",
        panel=[L("naringenin chalcone", "O=C(/C=C/c1ccc(O)cc1)c1c(O)cc(O)cc1O", "product"),
               L("isoliquiritigenin", "O=C(/C=C/c1ccc(O)cc1)c1ccc(O)cc1O", "product"),
               L("p-coumaric acid", "OC(=O)/C=C/c1ccc(O)cc1"), L("resveratrol", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"), GLC]),
    CandidateClass("chi", "chalcone isomerase", "dock",
        r"chalcone.{0,3}flavanone isomerase|chalcone isomerase",
        panel=[L("isoliquiritigenin", "O=C(/C=C/c1ccc(O)cc1)c1ccc(O)cc1O"),
               L("naringenin chalcone", "O=C(/C=C/c1ccc(O)cc1)c1c(O)cc(O)cc1O"),
               L("liquiritigenin", "O=C1C[C@@H](c2ccc(O)cc2)Oc2cc(O)ccc12", "product"),
               L("naringenin", "O=C1C[C@@H](c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "product"), GLC]),
    CandidateClass("acs", "ACC synthase (PLP)", "dock",
        r"1-aminocyclopropane-1-carboxylate synthase",
        cofactor="PLP — use AlphaFill",
        panel=[L("S-adenosyl-L-methionine", "S-adenosyl-L-methionine"),
               L("ACC", "NC1(CC1)C(=O)O", "product"),
               L("aminoethoxyvinylglycine (AVG)", "NCCO/C=C/[C@H](N)C(=O)O", "inhibitor"), GLC]),
    CandidateClass("aco", "ACC oxidase / 2OG-Fe(II) oxygenase", "dock",
        r"aminocyclopropanecarboxylate oxidase|2-oxoglutarate.{0,10}fe\(ii\)|2og|ec 1\.14\.11|jmjc|cysteine dioxygenase",
        cofactor="Fe(II) — use AlphaFill",
        panel=[L("ACC", "NC1(CC1)C(=O)O"), L("L-ascorbate", "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"),
               L("2-oxoglutarate", "OC(=O)CCC(=O)C(=O)O"),
               L("2-aminoisobutyric acid (inhibitor)", "CC(C)(N)C(=O)O", "inhibitor"), GLC]),
    CandidateClass("lox", "lipoxygenase", "dock",
        r"lipoxygenase",
        cofactor="non-heme Fe — use AlphaFill",
        panel=[L("linoleic acid", "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O"),
               L("α-linolenic acid", "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O"),
               L("nordihydroguaiaretic acid (inhibitor)", "CC(Cc1ccc(O)c(O)c1)C(C)Cc1ccc(O)c(O)c1", "inhibitor"), GLC]),
    CandidateClass("jmt", "SABATH methyltransferase (JMT)", "dock",
        r"jasmonate o-methyltransferase|jasmonic acid carboxyl methyltransferase",
        cofactor="SAM — use AlphaFill",
        panel=[L("jasmonic acid", "CC/C=C\\C[C@@H]1[C@@H](CC(=O)O)CCC1=O"),
               L("salicylic acid", "OC(=O)c1ccccc1O"), L("S-adenosyl-L-methionine", "S-adenosyl-L-methionine", "cofactor"), GLC]),
    CandidateClass("oxalate", "oxalate-active enzyme (germin / oxalyl-CoA decarboxylase / cupin)", "dock",
        r"germin|oxalyl-coa|oxalate|cupin",
        cofactor="Mn (germin) / TPP (oxalyl-CoA decarboxylase) — use AlphaFill",
        panel=[L("oxalate", "[O-]C(=O)C(=O)[O-]"), L("glyoxylate", "O=CC(=O)[O-]"),
               L("malonate", "[O-]C(=O)CC(=O)[O-]"), L("formate (product)", "[O-]C=O", "product"), GLC],
        note="oxalate degradation counters the main virulence acid of Sclerotinia"),
    CandidateClass("ugt", "UDP-glycosyltransferase", "dock",
        r"udp-glycosyltransferase|glucosyltransferase|\bgt1\b|\bgt28\b|anthocyanidin",
        panel=[L("UDP-glucose", "UDP-glucose", "cofactor"), *ISOFLAVONOIDS[:2], L("ergosterol", "ergosterol"), GLC]),
    CandidateClass("gh3_amido", "GH3 acyl-acid amido synthetase", "dock",
        r"auxin responsive gh3|gh3 gene family|amido synthetase|iaa-amino acid hydrolase",
        panel=[L("indole-3-acetic acid", "OC(=O)Cc1c[nH]c2ccccc12"), L("jasmonic acid", "CC/C=C\\C[C@@H]1[C@@H](CC(=O)O)CCC1=O"),
               L("L-aspartate", "N[C@@H](CC(=O)O)C(=O)O"), *NUCLEOTIDES[:1], GLC]),
    CandidateClass("gst", "glutathione transferase", "dock",
        r"glutathione transferase|glutathione s-transferase",
        panel=[L("glutathione", "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O"),
               L("CDNB", "O=[N+]([O-])c1ccc(Cl)c([N+](=O)[O-])c1"), ISOFLAVONOIDS[3], GLC]),
    CandidateClass("cytokinin", "cytokinin-metabolising enzyme", "dock",
        r"cytokinin dehydrogenase|cytokinin riboside|phosphoribohydrolase",
        cofactor="FAD (CKX) — use AlphaFill",
        panel=[L("trans-zeatin", "OC/C(C)=C/CNc1ncnc2nc[nH]c12"),
               L("isopentenyladenine", "CC(C)=CCNc1ncnc2nc[nH]c12"), *NUCLEOTIDES[2:], GLC]),

    # ── general metabolism (pathogen hubs) ──────────────────────────────
    CandidateClass("tpp_decarboxylase", "TPP-dependent decarboxylase / phosphoketolase", "dock",
        r"pyruvate decarboxylase|phosphoketolase",
        cofactor="TPP + Mg — use AlphaFill",
        panel=[L("pyruvate", "CC(=O)C(=O)[O-]"), L("phenylpyruvate", "O=C([O-])C(=O)Cc1ccccc1"),
               L("D-xylulose 5-phosphate", "D-xylulose 5-phosphate"), GLC]),
    CandidateClass("glyoxylate_cycle", "isocitrate lyase / TCA-adjacent enzyme", "dock",
        r"isocitrate lyase|methylcitrate|phosphoenolpyruvate carboxykinase|acetate kinase",
        panel=[L("isocitrate", "OC(=O)C[C@H](C(=O)O)[C@@H](O)C(=O)O"), L("glyoxylate", "O=CC(=O)[O-]", "product"),
               L("succinate", "[O-]C(=O)CCC(=O)[O-]", "product"), L("oxaloacetate", "O=C(CC(=O)[O-])C(=O)[O-]"),
               L("3-nitropropionate (ICL inhibitor)", "OC(=O)CC[N+](=O)[O-]", "inhibitor"), GLC]),
    CandidateClass("amino_acid_dh", "amino-acid dehydrogenase / oxidase / transaminase", "dock",
        r"glutamate dehydrogenase|saccharopine|proline dehydrogenase|aminotransferase|amine oxidase|"
        r"5-oxoprolinase|indoleamine|homogentisate 1,2-dioxygenase|homocysteine methyltransferase",
        panel=[L("L-glutamate", "N[C@@H](CCC(=O)O)C(=O)O"), L("2-oxoglutarate", "OC(=O)CCC(=O)C(=O)O"),
               L("L-lysine", "NCCCC[C@H](N)C(=O)O"), L("L-proline", "OC(=O)[C@@H]1CCCN1"),
               L("L-tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
               L("homogentisic acid", "OC(=O)Cc1cc(O)ccc1O"), L("putrescine", "NCCCCN"), GLC],
        note="broad panel: keep the ligands matching the enzyme's annotated reaction"),
    CandidateClass("sugar_metabolism", "glycogen / trehalose / amino-sugar metabolism", "dock",
        r"glycogen|trehalose|glucosamine-6-phosphate|adenylosuccinate|glycogenin|aldose 1-epimerase|"
        r"inositol 3-phosphate|ketopantoate",
        panel=[L("UDP-glucose", "UDP-glucose"), L("glucose 6-phosphate", "D-glucose 6-phosphate"),
               L("trehalose 6-phosphate", "trehalose 6-phosphate"), L("glucosamine 6-phosphate", "D-glucosamine 6-phosphate"),
               L("trehalose", "trehalose"), GLC]),
    CandidateClass("catalase", "catalase / heme-handling enzyme", "structure",
        r"catalase|heme oxygenase|coproporphyrinogen",
        note="substrates (H2O2, heme, porphyrinogens) are too small or too large for meaningful docking"),
    CandidateClass("phosphatase", "phosphatase / nucleotidase", "dock",
        r"phosphatase|nucleotidase|had-superfamily|sure-like|phosphoglycerate mutase",
        cofactor="metal ions — use AlphaFill",
        panel=[L("p-nitrophenyl phosphate", "O=[N+]([O-])c1ccc(OP(=O)([O-])[O-])cc1"),
               L("AMP", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
               L("phosphoserine", "N[C@@H](COP(=O)(O)O)C(=O)O"), GLC]),

    # ── signalling with nucleotide sites ────────────────────────────────
    CandidateClass("nlr", "NB-ARC / NACHT / TIR immune-type protein", "dock",
        r"nacht|nb-arc|disease resistance|rpp13|toll-interleukin|tir domain|heterokaryon incompatibility|"
        r"adp binding",
        panel=[*NUCLEOTIDES, L("NAD+ (TIR NADase)", "NAD+")],
        note="nucleotide pocket only; activation needs the full oligomeric receptor"),
    CandidateClass("kinase", "protein kinase / receptor-like kinase", "dock",
        r"kinase|mapk|pbl10|ccr3|rbk2|crk\d",
        cofactor="Mg-ATP",
        panel=[*NUCLEOTIDES, L("staurosporine (pan-kinase inhibitor)", "staurosporine", "inhibitor")],
        note="ATP-site docking only; signalling output is protein–protein"),
    CandidateClass("gtpase", "small GTPase", "dock",
        r"gtp-binding|rac-like|\brop\b|ras-related",
        cofactor="Mg2+",
        panel=[L("GTP", "Nc1nc2n(cnc2c(=O)[nH]1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
               L("GDP", "Nc1nc2n(cnc2c(=O)[nH]1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
               NUCLEOTIDES[0]]),
    CandidateClass("parp", "ADP-ribosyltransferase", "dock",
        r"adp-ribosyltransferase|parp",
        panel=[L("NAD+", "NAD+"), L("nicotinamide", "NC(=O)c1cccnc1", "product"), L("olaparib", "olaparib", "inhibitor"), GLC]),
    CandidateClass("rboh", "NADPH oxidase (RBOH)", "unsuited",
        r"respiratory burst oxidase|nadph oxidase",
        note="membrane-embedded multi-domain enzyme; dock NADPH only in a full-length model"),
    CandidateClass("carbonic_anhydrase", "carbonic anhydrase", "dock",
        r"carbonic anhydrase",
        cofactor="Zn — use AlphaFill",
        panel=[L("acetazolamide (inhibitor)", "CC(=O)Nc1nnc(S(N)(=O)=O)s1", "inhibitor"),
               L("bicarbonate", "OC(=O)[O-]"), GLC]),
    CandidateClass("housekeeping", "housekeeping / structural / photosynthetic machinery", "unsuited",
        r"ribosomal|ribosome|translation initiation|elongation factor|trna ligase|trna deacylase|chaperone|"
        r"heat shock|dnaj|tubulin|actin|photosystem|light-harvesting|chlorophyll a-b binding|ferredoxin-1$|"
        r"oleosin|annexin|ubiquitin|f-box|u-box|histone|pseudo-response regulator|sigma factor|"
        r"curvature thylakoid|rna-binding|polyribonucleotide|cell division cycle|zip4|artemis|"
        r"ethylene response sensor|reversion-to-ethylene|phosphotransfer protein|calcium-binding protein cml|"
        r"ef-hand|wax synthase|heavy metal-associated|c2 domain|gem-like|saur|"
        r"syringolide|sob five|root hair defective|trichome birefringence",
        note="housekeeping, structural, membrane or signalling-scaffold protein: not a small-molecule target"),
    CandidateClass("opr", "12-oxophytodienoate reductase (jasmonate biosynthesis)", "dock",
        r"oxophytodienoate reductase|\bopr\d",
        cofactor="FMN — use AlphaFill",
        panel=[L("12-oxophytodienoic acid (OPDA)", "12-oxophytodienoic acid"),
               L("2-cyclohexen-1-one (model enone)", "O=C1C=CCCC1"), GLC],
        note="jasmonate biosynthesis"),
    CandidateClass("hmg", "HMG-CoA synthase / reductase (mevalonate → sterols)", "dock",
        r"hydroxymethylglutaryl|hmg-coa|3-hydroxy-3-methylglutaryl",
        panel=[L("mevalonate", "C[C@](O)(CCO)CC(=O)O", "product"), L("lovastatin (HMGR inhibitor)", "lovastatin", "inhibitor"),
               L("hymeglusin (HMGS inhibitor)", "hymeglusin", "inhibitor"), GLC],
        note="mevalonate pathway, upstream of ergosterol / phytosterols"),
    CandidateClass("lectin_man", "mannose-binding antifungal lectin (Gnk2 / DUF26)", "dock",
        r"gnk2|salt stress response/antifungal|duf26",
        panel=[L("D-mannose", "D-mannose"), L("methyl α-D-mannoside", "methyl alpha-D-mannopyranoside"),
               L("α-1,2-mannobiose", "2-O-alpha-D-mannopyranosyl-D-mannose"),
               L("D-galactose (control)", "D-galactose", "control"), GLC],
        note="Gnk2 family members bind mannose and inhibit fungi"),
    CandidateClass("metabolic_other", "other annotated enzyme (no curated panel)", "structure",
        r"dehydrogenase|reductase|synthase|synthetase|transferase|isomerase|epimerase|ligase|oxidase|"
        r"hydrolase|desaturase|decarboxylase|cyclohydrolase|transaminase|lyase|mutase|dehydratase|"
        r"oxygenase|hydroxylase|ase\b",
        note="annotated enzyme without a curated ligand panel: open it and dock its annotated substrate "
             "by typing it in the Ligand box"),
    CandidateClass("metacaspase", "metacaspase / cell-death protease", "structure",
        r"metacaspase|programmed cell death",
        note="Cys protease activated by Ca2+ and processing; pocket analysis, not docking"),
]

# Order matters (first match wins): things that must never be docked first,
# then specific chemistry before broad chemistry (peptidase before the
# "lipase"/"esterase" catch-all, β-1,3-glucanase families before pectinases,
# pectinases before esterases), and the effector/unknown catch-all last so an
# annotated CAZyme that is *also* a predicted effector is still docked.
_ORDER = ["transposon", "tf", "transporter", "ppi", "rboh", "housekeeping",
          "laccase", "peroxidase", "p450", "thn_reductase", "dehydrogenase", "gmc_oxidase", "cro", "lpmo",
          "opr", "hmg", "gh3_amido", "peptidase", "b13_glucanase", "pectinase", "cutinase_esterase", "epoxide_hydrolase",
          "mannosidase", "chitinase", "lysm", "hemicellulase", "cellulase", "amylase", "ceramidase",
          "lectin_fucose", "lectin_gal", "lectin_man", "sterol_mt", "osc", "squalene_epoxidase", "diterpene_synthase",
          "sdh", "aox", "pal", "chs", "chi", "acs", "aco", "lox", "jmt", "oxalate", "ugt",
          "gst", "cytokinin", "tpp_decarboxylase", "glyoxylate_cycle", "amino_acid_dh", "sugar_metabolism",
          "catalase", "phosphatase", "nlr", "gtpase", "parp", "carbonic_anhydrase", "metacaspase",
          "kinase", "metabolic_other", "effector_unknown"]
_by_key = {r.key: r for r in RULES}
assert set(_by_key) == set(_ORDER), set(_by_key) ^ set(_ORDER)
RULES = [_by_key[k] for k in _ORDER]

FALLBACK = CandidateClass("unclassified", "no matching rule", "structure", r"$^",
                          note="no rule matched the annotation; inspect the structure and pockets")


def classify(description: str, family: str = "") -> CandidateClass:
    fam = (family or "").strip()
    # "GH43_1" also matches rules written for the family "GH43"
    fam_tokens = " ".join({fam, re.sub(r"_\d+$", "", fam)}) if fam and fam.upper() != "NA" else ""
    text = f"{description or ''} | {fam_tokens}".lower()
    for rule in RULES:
        if rule.matches(text):
            return rule
    return FALLBACK


def all_panel_smiles() -> list[tuple[str, str]]:
    """(class, value) for every ligand in every panel — used by the tests to
    check that every SMILES parses."""
    out = []
    for r in RULES:
        for lig in r.panel:
            out.append((r.key, lig.value))
    return out
