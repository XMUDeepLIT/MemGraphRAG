PROMPT = {}
PROMPT['entity_type_extraction'] = {
    "system": """You are an expert entity type classifier. Given a passage and a list of entities extracted from that passage, your task is to classify each entity into its most appropriate type.

TASK
For EACH entity string, assign exactly ONE entity type from the Allowed Types list below.
- output ONLY one type label per entity (no hierarchies like "Organization/Company", no multi-label).
- Use the passage to disambiguate (e.g., "Apple" could be a Company or a Food).
- Do NOT add new entities. Do NOT remove entities.
- Keep the output order exactly the same as the input entity list (including duplicates if any).
- If NONE of the allowed types fits well, return other domain-specific types as appropriate

Entity types should be specific and meaningful. Common types include but are not limited to:
<People & Social Entities>
- Person (named individual human)
- FictionalCharacter (fictional person/character)
- GroupOfPeople (named group: “the committee”, “residents”, “engineers”)
- DemographicGroup (age/gender/etc group: “children”, “women”)
- EthnicGroup (ethnicity: “Han Chinese”, “Kurdish”)
- NationalityOrCitizenship (e.g., “French”, “Chinese” when used as identity)
- RoleTitle (job title/position: “CEO”, “President”, “Professor”)
- Occupation (profession: “doctor”, “lawyer” used generically)
- SocialMediaAccount (handle/account name: “@xxx”, channel account) 

<Organizations & Institutions>
- Organization (generic organization when subtype unclear)
- Company (commercial company)
- Startup (early-stage company explicitly described)
- Brand (brand/trademark name used as brand)
- NonProfit (nonprofit/charity)
- NGO (non-governmental organization)
- Foundation (foundation as org)
- Government (sovereign government as an entity)
- GovernmentAgency (agency/bureau/authority)
- MinistryDepartment (ministry/department/commission)
- LocalGovernment (city/municipal/provincial government body)
- MilitaryUnit (army/navy/air force unit or command)
- LawEnforcementAgency (police, sheriff, etc.)
- IntelligenceAgency (intelligence services)
- Court (court as institution)
- LegislativeBody (parliament/congress/council)
- PoliticalParty (political party)
- InternationalOrganization (UN-like intergovernmental org)
- EducationalInstitution (school/university/college)
- ResearchInstitute (institute/academy/research center)
- Laboratory (lab as org/unit)
- HospitalOrganization (hospital as institution)
- ClinicOrganization (clinic as institution)
- SportsTeam (team/club)
- SportsLeague (league/association)
- ReligiousOrganization (church/temple/diocese)
- MediaOrganization (media outlet/company)
- Publisher (publishing house)
- FinancialInstitution (generic)
- Bank
- InsuranceCompany
- InvestmentFirm (asset manager)
- VentureCapitalFirm
- HedgeFund
- ConsultingFirm
- LawFirm
- AccountingFirm
- ProfessionalAssociation (society/association)
- LaborUnion (union)
- StandardsBody (standards-setting org)

<Locations & Geography>
- Location (generic place when subtype unclear)
- Country
- Territory (dependency/special territory)
- StateProvince (state/province/prefecture)
- County (county-level)
- City
- TownVillage
- NeighborhoodDistrict
- StreetRoad
- Address (full/partial address)
- Building (named building)
- Facility (named facility/site)
- Campus (campus/site complex)
- OfficeSite (office location)
- StadiumArena
- Airport
- TrainStation
- PortHarbor
- Bridge
- Park
- Mountain
- River
- Lake
- Sea
- Ocean
- Island
- Region (broad region: “Middle East”, “Bay Area”)
- Continent
- GeoCoordinate (lat/long)

<Time & Temporal>
- Date (specific calendar date)
- Time (time of day)
- DateRange (start–end date range)
- TimeRange (time interval within day)
- Duration (e.g., “three years”)
- PeriodEra (e.g., “Q3 2024”, “the 1990s”)
- Season (spring/summer/etc)
- Holiday (named holiday/festival)

<Events & Happenings>
- Event (generic event)
- Meeting (meeting/appointment)
- Conference (conference/summit/forum)
- WorkshopTraining (workshop/training)
- Webinar (online seminar)
- InterviewEvent (interview as event)
- SpeechEvent (speech/address)
- Ceremony (ceremony/commemoration)
- AwardEvent (award ceremony)
- Election (election)
- Referendum (referendum)
- Protest (protest/demonstration)
- Strike (labor strike)
- SportsEvent (sports event broadly)
- SportsMatch (single match/game)
- Tournament (tournament/league season)
- War (war)
- Battle (battle)
- MilitaryOperation (operation/campaign)
- DisasterEvent (natural/large disaster)
- AccidentEvent (accident)
- CrimeIncident (crime incident)
- InvestigationEvent (investigation)
- TrialEvent (trial hearing)
- CourtCase (case/lawsuit as proceeding)
- ContractSigning (signing event)
- PolicyAnnouncement (policy announced)
- ProductLaunchEvent (launch event)
- ReleaseEvent (release/drop)
- AcquisitionEvent (acquisition)
- MergerEvent (merger)
- PartnershipEvent (partnership formed)
- FundingRoundEvent (funding round)
- IPOEvent (IPO)
- BankruptcyEvent (bankruptcy filing)
- RecallEvent (product recall)
- ClinicalTrialEvent (trial as event)
- StudyExperimentEvent (experiment/study as event)

<Products, Services, Tech & Artefacts>
- Product (generic product)
- Service (service offering)
- Software (software/system)
- MobileApp (app)
- Website (website)
- OnlinePlatformProduct (platform/product like a SaaS platform)
- HardwareDevice (device)
- ElectronicComponent (chip/sensor/etc)
- Vehicle (vehicle type)
- VehicleModel (specific model name)
- Aircraft
- Spacecraft
- Drone
- Robot
- ToolInstrument (tool/instrument)
- MachineEquipment (equipment)
- Weapon (weapon system)
- MedicalDevice (medical device)
- DrugMedication (drug/medicine)
- Vaccine (vaccine)
- ChemicalSubstance (chemical)
- Material (material: “steel”, “lithium”)
- FoodDish (dish/food item)
- Beverage (drink)
- EnergySource (oil/solar/etc)

<Finance & Commerce>
- Currency (USD, RMB, etc.)
- CryptoAsset (Bitcoin, etc.)
- FinancialInstrument (generic: stock/bond/etc when unclear)
- StockTicker (ticker symbol like AAPL)
- Bond (bond)
- Derivative (options/futures)
- InsurancePolicyProduct (policy product)
- MonetaryAmount (money figure: “$5M”)
- Price (price figure)
- Percentage (percent figure)
- Rate (interest/growth rate)
- MetricKPI (named metric: “CPI”, “ARR”)
- Budget (budget allocation)

<Documents, Media, IP & Knowledge Objects>
- Document (generic document)
- ContractDocument (contract/agreement text)
- LawRegulationDocument (specific law/regulation document)
- Report (report/whitepaper)
- AcademicPaper (paper/journal article)
- Book
- NewsArticle (article)
- BlogPost
- Dataset (dataset)
- Database (database)
- StandardDocument (published standard spec)
- Patent (patent)
- LicenseCertificate (license/certification)
- FormTemplate (form)
- PolicyDocument (policy as document)
- Presentation (slides/deck)

<Creative Works & Entertainment>
- Film
- TVSeries
- TVEpisode
- Song
- Album
- Video (general video)
- VideoGame
- Podcast
- Artwork (art piece)
- Photograph
- Novel
- ComicManga
- PlayTheatreWork
- PerformanceShow

<Science, Health & Nature>
- Species (animal/plant species)
- Gene
- Protein
- CellType
- OrganAnatomy
- Symptom
- DiseaseCondition
- TreatmentProcedure (therapy/surgery)
- MedicalTest (test/assay)
- Biomarker

<Computing & Data Structures>
- Algorithm (algorithm name)
- MLModel (named model like “BERT”)
- ProgrammingLanguage
- LibraryFramework (PyTorch, React)
- API (API name)
- Protocol (HTTP, TCP)
- FileFormat (PDF, JSON)
- Identifier (generic ID)
- PhoneNumber
- EmailAddress
- URL
- IPAddress
- HashChecksum

<Abstract Concepts (non-physical)>  
- Concept (generic abstract concept)
- Topic (topic/subject)
- TechnologyConcept (tech as concept)
- ScientificField (field/discipline)
- Theory (theory/hypothesis)
- MethodTechnique (method/technique)
- StrategyPlan (strategy/plan)
- PolicyConcept (policy as concept, not a document)
- LawRegulationConcept (law/regulation as concept)
- StandardConcept (standard as concept)
- InitiativeProgram (initiative/program)
- Project (project name)
- Campaign (campaign/initiative)
- ProblemIssue (problem/issue)
- Risk (risk type)
- GoalObjective (goal/objective)

Output MUST be a valid JSON object following the required schema.""",

    "user": """Given the following passage and entities, classify each entity into its most appropriate type.

Passage:
${passage}

Entities:
${entities}

Output a JSON object with the following structure:
{
  "entity_types": [
    {
      "entity": "entity name",
      "type": "entity type"
    }
  ]
}

For each entity, provide:
1. The exact entity name as it appears in the entities list
2. The most specific and appropriate type for that entity based on the passage context

JSON payload:
"""
}

# Ontology extraction prompt: map triples to type-level triples
PROMPT['ontology_extraction'] = {
    "system": """You are an ontology converter. Given a passage and a list of factual triples (head, relation, tail) extracted from that passage, output ontology-level triples by replacing the head and tail entities with their most appropriate entity TYPE while keeping the relation unchanged.

Requirements:
- Infer types from the passage context; be as specific as possible but concise (one label).
- Do NOT invent new relations; keep relation text exactly as provided.
- Preserve one-to-one alignment: each input triple must have exactly one ontology triple.
- If a type is unclear, return "Unknown".
- Output MUST be valid JSON.

Common types include but are not limited to:
- <CARDINAL>: Numerals that do not fall under another type,
- <DATE>: Absolute or relative dates or periods,
- <EVENT>: Named hurricanes, battles, wars, sports events, etc.,
- <FAC>: Buildings, airports, highways, bridges, etc.,
- <GPE>: Countries, cities, states,
- <LANGUAGE>: Any named language,
- <LAW>: Named documents made into laws.,
- <LOC>: Non-GPE locations, mountain ranges, bodies of water,
- <MONEY>: Monetary values, including unit,
- <NORP>: Nationalities or religious or political groups,
- <ORDINAL>: \"first\", \"second\", etc.,
- <ORG>: Companies, agencies, institutions, etc.,
- <PERCENT>: Percentage, including \"%\",
- <PERSON>: People, including fictional,
- <PRODUCT>: Objects, vehicles, foods, etc. (not services),
- <QUANTITY>: Measurements, as of weight or distance,
- <TIME>: Times smaller than a day,
- <WORK_OF_ART>: Titles of books, songs, etc.
""",
    "user": """Convert the following triples to ontology-level triples by replacing head and tail with their entity types. Keep relation unchanged.

Passage:
${passage}

Triples:
${triples}

Return JSON with structure:
{
  "ontology_triples": [
    {
      "triple": ["head", "relation", "tail"],
      "ontology": ["head_type", "relation", "tail_type"]
    }
  ]
}

JSON payload:
"""
}

# Conflict detection prompt: detect conflicts between triples
PROMPT['conflict_detection'] = {
    "system": """You are an expert knowledge graph fact checker.

Given ONE target triple and multiple related triples, detect whether the target conflicts with each related triple.
Be conservative: avoid false positives.

Conflict categories:
1) mutual: truly mutually exclusive facts.
2) temporal: conflict depends on time overlap.
3) granularity: different specificity level (often compatible, not hard conflict).

Important anti-false-positive rules:
- Exact duplicates are NOT conflicts.
  If two triples are semantically the same after normalization (case/punctuation/spacing), mark as "duplicate".
- One-to-many predicates are usually NOT mutual conflicts.
  For predicates like "located in", "contains", "member of", "speaks", "has part", "includes", "adjacent to",
  multiple objects may coexist. Do not mark as mutual unless clearly impossible.
- For same subject + predicate + different objects:
  first decide predicate uniqueness:
  - globally_unique (normally only one true object ever, e.g., biological mother, birth_date)
  - moment_unique (one true object at a single time, e.g., current CEO)
  - multi_valued (multiple objects can be true)
  Only globally_unique/moment_unique can become hard mutual conflicts.
- Temporal facts:
  if times overlap and claims are incompatible -> temporal hard conflict;
  if times do not overlap -> not conflict;
  if time is missing -> mark uncertainty, do not force hard conflict.
- Granularity:
  if one object contains/is-parent-of the other (city-country etc.), treat as compatible granularity difference.

When uncertain, prefer "no hard conflict" and explain uncertainty.
Output MUST be valid JSON only.""",

    "user": """Analyze the following triples for conflicts.

Target Triple:
${target_triple}

Related Triples:
${related_triples}

Output a JSON object with the following structure:
{
  "has_conflict": true/false,
  "conflicts": [
    {
      "triple1": ["head", "relation", "tail"],
      "triple2": ["head", "relation", "tail"],
      "conflict_type": "mutual|temporal|granularity|duplicate|none|uncertain",
      "predicate_uniqueness_judgment": "globally_unique|moment_unique|multi_valued|unknown",
      "is_hard_conflict": true/false,
      "confidence": 0.0,
      "needs_resolution": true/false,
      "conflict_reason": "brief explanation"
    }
  ],
  "conflicting_triple_ids": ["id1", "id2", ...]
}

Rules for output fields:
- "is_hard_conflict" is true only for real contradictions requiring curation.
- "needs_resolution" is true only when "is_hard_conflict" is true.
- "duplicate", "granularity" (compatible), and "uncertain" should have is_hard_conflict=false and needs_resolution=false.
- "has_conflict" should indicate whether any hard conflict exists.
- Include IDs only for hard conflicts in "conflicting_triple_ids".

JSON payload:
"""
}

# Conflict resolution prompt: resolve conflicts between triples using source passages
PROMPT['conflict_resolution'] = {
    "system": """You are an expert knowledge graph curator. Given a set of conflicting triples and their source passages, your task is to resolve the conflicts and produce corrected triples.

Conflict Resolution Strategies:

1. Mutual Conflict (type = "mutual"):
   - These are contradictory claims about the same entity (e.g., same subject-predicate but different objects)
   - Resolution: Analyze the source passages to determine which triple is more accurate
   - Keep only the CORRECT triple, discard the incorrect one(s)
   - If both seem equally valid based on context, prefer the one with more specific/credible source

2. Temporal Conflict (type = "temporal"):
   - These are time-dependent facts where time scopes overlap or are missing
   - Resolution: Add time information to the relation to distinguish the facts
   - Modify the predicate to include time context (e.g., "was president of [2000-2005]" vs "was president of [2005-2010]")
   - If time info is not in sources, note it as "temporal_conflict_unresolved"

3. Granularity Conflict (type = "granularity"):
   - These are facts at different levels of specificity (e.g., "born in Shanghai" vs "born in China")
   - Resolution: Add granularity description to the relation to clarify the scope
   - Modify the predicate to include granularity context (e.g., "was born in [city: Shanghai]" vs "was born in [country: China]")
   - Both can be kept if they are compatible (containment relationship)

Output MUST be a valid JSON object following the required schema.""",

    "user": """Resolve the following conflicting triples using their source passages.

Conflicting Triples and Their Sources:
${conflicting_triples_with_sources}

Output a JSON object with the following structure:
{
  "resolved_triples": [
    {
      "original_triple": ["head", "relation", "tail"],
      "triple_id": "fact_id",
      "conflict_type": "mutual|temporal|granularity",
      "resolution": "kept|discarded|modified",
      "resolved_triple": ["head", "modified_relation", "tail"] or null if discarded,
      "reason": "explanation of why this resolution was chosen"
    }
  ],
  "unresolved_conflicts": [
    {
      "triple_ids": ["id1", "id2"],
      "reason": "reason why conflict could not be resolved"
    }
  ],
  "summary": "brief summary of how conflicts were resolved"
}

For each conflicting triple:
- If resolution is "kept": Keep the triple as is (it's correct)
- If resolution is "discarded": The triple is incorrect, set resolved_triple to null
- If resolution is "modified": Provide the modified triple with time/granularity info in the relation

JSON payload:
"""
}
