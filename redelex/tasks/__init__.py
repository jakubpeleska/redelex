from relbench.tasks import register_task

from .ctu_entity_task_base import CTUBaseEntityTask
from .ctu_entity_task_classic import CTUEntityTask
from .ctu_entity_task_temporal import CTUEntityTaskTemporal

# fmt: off
from .ctu_tasks import (
    AccidentsOriginalTask, AccidentsTemporalTask, AdventureWorksOriginalTask, 
    AdventureWorksTemporalTask, AirlineOriginalTask, AirlineTemporalTask, 
    AtherosclerosisOriginalTask, BasketballMenOriginalTask, BasketballWomenOriginalTask,
    BiodegradabilityOriginalTask, BupaOriginalTask, CarcinogenesisOriginalTask,
    CDESchoolsOriginalTask, ChessOriginalTask, ClassicModelsOriginalTask,
    ClassicModelsTemporalTask, CORAOriginalTask, CountriesOriginalTask,
    CraftBeerOriginalTask, CreditOriginalTask, DallasOriginalTask, DallasTemporalTask,
    DCGOriginalTask, DiabetesOriginalTask, DunurOriginalTask, EltiOriginalTask,
    EmployeeOriginalTask, EmployeeTemporalTask, ErgastF1OriginalTask,
    ErgastF1TemporalTask, ExpendituresOriginalTask, FinancialOriginalTask,
    FinancialTemporalTask, FNHKOriginalTask, FNHKTemporalTask, FTPOriginalTask,
    FTPTemporalTask, GeneeaOriginalTask, GeneeaTemporalTask, GenesOriginalTask,
    GOSalesOriginalTask, GOSalesTemporalTask, GrantsOriginalTask, GrantsTemporalTask,
    HepatitisOriginalTask, HockeyOriginalTask, HockeyTemporalTask, IMDbOriginalTask,
    IMDbTemporalTask, LahmanOriginalTask, LahmanTemporalTask, LegalActsOriginalTask,
    LegalActsTemporalTask, MeshOriginalTask, MondialOriginalTask, MooneyOriginalTask,
    MovieLensOriginalTask, MuskLargeOriginalTask, MuskSmallOriginalTask,
    MutagenesisOriginalTask, NCAAOriginalTask, NCAATemporalTask, NorthwindOriginalTask,
    NorthwindTemporalTask, PimaOriginalTask, PremiereLeagueOriginalTask,
    PremiereLeagueTemporalTask, RestbaseOriginalTask, SakilaOriginalTask,
    SakilaTemporalTask, SalesOriginalTask, SameGenOriginalTask, SAPOriginalTask,
    SAPSalesTask, SAPSalesTemporalTask, SatelliteOriginalTask, SeznamOriginalTask,
    SeznamTemporalTask, SFScoresOriginalTask, SFScoresTemporalTask,
    ShakespeareOriginalTask, StatsOriginalTask, StatsTemporalTask, StudentLoanOriginalTask,
    ThrombosisOriginalTask, ThrombosisTemporalTask, ToxicologyOriginalTask,
    TPCCOriginalTask, TPCDOriginalTask, TPCDTemporalTask, TPCDSOriginalTask,
    TPCDSTemporalTask, TPCHOriginalTask, TPCHTemporalTask, TriazineOriginalTask,
    UWCSEOriginalTask, VisualGenomeOriginalTask, VOCOriginalTask, VOCTemporalTask,
    WalmartOriginalTask, WalmartTemporalTask, WebKPOriginalTask, WorldOriginalTask
)
# fmt: on

register_task("ctu-accidents", "accidents-original", AccidentsOriginalTask)

register_task("ctu-accidents", "accidents-temporal", AccidentsTemporalTask)

register_task("ctu-adventureworks", "adventureworks-original", AdventureWorksOriginalTask)
register_task("ctu-adventureworks", "adventureworks-temporal", AdventureWorksTemporalTask)

register_task("ctu-airline", "airline-original", AirlineOriginalTask)
register_task("ctu-airline", "airline-temporal", AirlineTemporalTask)

register_task(
    "ctu-atherosclerosis", "atherosclerosis-original", AtherosclerosisOriginalTask
)
register_task("ctu-basketballmen", "basketballmen-original", BasketballMenOriginalTask)
register_task(
    "ctu-basketballwomen", "basketballwomen-original", BasketballWomenOriginalTask
)
register_task(
    "ctu-biodegradability", "biodegradability-original", BiodegradabilityOriginalTask
)
register_task("ctu-bupa", "bupa-original", BupaOriginalTask)
register_task("ctu-carcinogenesis", "carcinogenesis-original", CarcinogenesisOriginalTask)
register_task("ctu-cde", "cde-original", CDESchoolsOriginalTask)
register_task("ctu-chess", "chess-original", ChessOriginalTask)

register_task("ctu-classicmodels", "classicmodels-original", ClassicModelsOriginalTask)
register_task("ctu-classicmodels", "classicmodels-temporal", ClassicModelsTemporalTask)

register_task("ctu-cora", "cora-original", CORAOriginalTask)
register_task("ctu-countries", "countries-original", CountriesOriginalTask)
register_task("ctu-craftbeer", "craftbeer-original", CraftBeerOriginalTask)
register_task("ctu-credit", "credit-original", CreditOriginalTask)

register_task("ctu-dallas", "dallas-original", DallasOriginalTask)
register_task("ctu-dallas", "dallas-temporal", DallasTemporalTask)

register_task("ctu-dcg", "dcg-original", DCGOriginalTask)
register_task("ctu-diabetes", "diabetes-original", DiabetesOriginalTask)
register_task("ctu-dunur", "dunur-original", DunurOriginalTask)
register_task("ctu-elti", "elti-original", EltiOriginalTask)

register_task("ctu-employee", "employee-original", EmployeeOriginalTask)
register_task("ctu-employee", "employee-temporal", EmployeeTemporalTask)

register_task("ctu-ergastf1", "ergastf1-original", ErgastF1OriginalTask)
# register_task("ctu-ergastf1", "ergastf1-temporal", ErgastF1TemporalTask)

register_task("ctu-expenditures", "expenditures-original", ExpendituresOriginalTask)

register_task("ctu-financial", "financial-original", FinancialOriginalTask)
register_task("ctu-financial", "financial-temporal", FinancialTemporalTask)

register_task("ctu-fnhk", "fnhk-original", FNHKOriginalTask)
register_task("ctu-fnhk", "fnhk-temporal", FNHKTemporalTask)

register_task("ctu-ftp", "ftp-original", FTPOriginalTask)
register_task("ctu-ftp", "ftp-temporal", FTPTemporalTask)

register_task("ctu-geneea", "geneea-original", GeneeaOriginalTask)
register_task("ctu-geneea", "geneea-temporal", GeneeaTemporalTask)

register_task("ctu-genes", "genes-original", GenesOriginalTask)

register_task("ctu-gosales", "gosales-original", GOSalesOriginalTask)
register_task("ctu-gosales", "gosales-temporal", GOSalesTemporalTask)

register_task("ctu-grants", "grants-original", GrantsOriginalTask)
register_task("ctu-grants", "grants-temporal", GrantsTemporalTask)

register_task("ctu-hepatitis", "hepatitis-original", HepatitisOriginalTask)

register_task("ctu-hockey", "hockey-original", HockeyOriginalTask)
# register_task("ctu-hockey", "hockey-temporal", HockeyTemporalTask)

register_task("ctu-imdb", "imdb-original", IMDbOriginalTask)
# register_task("ctu-imdb", "imdb-temporal", IMDbTemporalTask)

register_task("ctu-lahman", "lahman-original", LahmanOriginalTask)
register_task("ctu-lahman", "lahman-temporal", LahmanTemporalTask)

register_task("ctu-legalacts", "legalacts-original", LegalActsOriginalTask)
register_task("ctu-legalacts", "legalacts-temporal", LegalActsTemporalTask)

register_task("ctu-mesh", "mesh-original", MeshOriginalTask)
register_task("ctu-mondial", "mondial-original", MondialOriginalTask)
register_task("ctu-mooney", "mooney-original", MooneyOriginalTask)
register_task("ctu-movielens", "movielens-original", MovieLensOriginalTask)
register_task("ctu-musklarge", "musklarge-original", MuskLargeOriginalTask)
register_task("ctu-musksmall", "musksmall-original", MuskSmallOriginalTask)
register_task("ctu-mutagenesis", "mutagenesis-original", MutagenesisOriginalTask)

register_task("ctu-ncaa", "ncaa-original", NCAAOriginalTask)
# register_task("ctu-ncaa", "ncaa-temporal", NCAATemporalTask)

register_task("ctu-northwind", "northwind-original", NorthwindOriginalTask)
register_task("ctu-northwind", "northwind-temporal", NorthwindTemporalTask)

register_task("ctu-pima", "pima-original", PimaOriginalTask)

register_task("ctu-premiereleague", "premiereleague-original", PremiereLeagueOriginalTask)
register_task("ctu-premiereleague", "premiereleague-temporal", PremiereLeagueTemporalTask)

register_task("ctu-restbase", "restbase-original", RestbaseOriginalTask)

register_task("ctu-sakila", "sakila-original", SakilaOriginalTask)
register_task("ctu-sakila", "sakila-temporal", SakilaTemporalTask)

register_task("ctu-sales", "sales-original", SalesOriginalTask)
register_task("ctu-samegen", "samegen-original", SameGenOriginalTask)

register_task("ctu-sap", "sap-original", SAPOriginalTask)
register_task("ctu-sap", "sap-sales", SAPSalesTask)
register_task("ctu-sap", "sap-sales-temporal", SAPSalesTemporalTask)

register_task("ctu-satellite", "satellite-original", SatelliteOriginalTask)

register_task("ctu-seznam", "seznam-original", SeznamOriginalTask)
register_task("ctu-seznam", "seznam-temporal", SeznamTemporalTask)

register_task("ctu-sfscores", "sfscores-original", SFScoresOriginalTask)
register_task("ctu-sfscores", "sfscores-temporal", SFScoresTemporalTask)

register_task("ctu-shakespeare", "shakespeare-original", ShakespeareOriginalTask)

register_task("ctu-stats", "stats-original", StatsOriginalTask)
register_task("ctu-stats", "stats-temporal", StatsTemporalTask)

register_task("ctu-studentloan", "studentloan-original", StudentLoanOriginalTask)

register_task("ctu-thrombosis", "thrombosis-original", ThrombosisOriginalTask)
# register_task("ctu-thrombosis", "thrombosis-temporal", ThrombosisTemporalTask)

register_task("ctu-toxicology", "toxicology-original", ToxicologyOriginalTask)
register_task("ctu-tpcc", "tpcc-original", TPCCOriginalTask)

register_task("ctu-tpcd", "tpcd-original", TPCDOriginalTask)
# register_task("ctu-tpcd", "tpcd-temporal", TPCDTemporalTask)

register_task("ctu-tpcds", "tpcds-original", TPCDSOriginalTask)
register_task("ctu-tpcds", "tpcds-temporal", TPCDSTemporalTask)

register_task("ctu-tpch", "tpch-original", TPCHOriginalTask)
# register_task("ctu-tpch", "tpch-temporal", TPCHTemporalTask)

register_task("ctu-triazine", "triazine-original", TriazineOriginalTask)
register_task("ctu-uwcse", "uwcse-original", UWCSEOriginalTask)
register_task("ctu-visualgenome", "visualgenome-original", VisualGenomeOriginalTask)

register_task("ctu-voc", "voc-original", VOCOriginalTask)
register_task("ctu-voc", "voc-temporal", VOCTemporalTask)

register_task("ctu-walmart", "walmart-original", WalmartOriginalTask)
register_task("ctu-walmart", "walmart-temporal", WalmartTemporalTask)

register_task("ctu-webkp", "webkp-original", WebKPOriginalTask)
register_task("ctu-world", "world-original", WorldOriginalTask)


# fmt: off
__all__ = [
    "CTUBaseEntityTask", "CTUEntityTask", "CTUEntityTaskTemporal",
    
    "AccidentsOriginalTask", "AccidentsTemporalTask", "AdventureWorksOriginalTask", 
    "AdventureWorksTemporalTask", "AirlineOriginalTask", "AirlineTemporalTask", 
    "AtherosclerosisOriginalTask", "BasketballMenOriginalTask", "BasketballWomenOriginalTask",
    "BiodegradabilityOriginalTask", "BupaOriginalTask", "CarcinogenesisOriginalTask",
    "CDESchoolsOriginalTask", "ChessOriginalTask", "ClassicModelsOriginalTask",
    "ClassicModelsTemporalTask", "CORAOriginalTask", "CountriesOriginalTask",
    "CraftBeerOriginalTask", "CreditOriginalTask", "DallasOriginalTask", "DallasTemporalTask",
    "DCGOriginalTask", "DiabetesOriginalTask", "DunurOriginalTask", "EltiOriginalTask",
    "EmployeeOriginalTask", "EmployeeTemporalTask", "ErgastF1OriginalTask", "ErgastF1TemporalTask",
    "ExpendituresOriginalTask", "FinancialOriginalTask", "FinancialTemporalTask", "FNHKOriginalTask",
    "FNHKTemporalTask", "FTPOriginalTask", "FTPTemporalTask", "GeneeaOriginalTask",
    "GeneeaTemporalTask", "GenesOriginalTask", "GOSalesOriginalTask", "GOSalesTemporalTask",
    "GrantsOriginalTask", "GrantsTemporalTask", "HepatitisOriginalTask", "HockeyOriginalTask",
    "HockeyTemporalTask", "IMDbOriginalTask", "IMDbTemporalTask", "LahmanOriginalTask",
    "LahmanTemporalTask", "LegalActsOriginalTask", "LegalActsTemporalTask", "MeshOriginalTask",
    "MondialOriginalTask", "MooneyOriginalTask", "MovieLensOriginalTask", "MuskLargeOriginalTask",
    "MuskSmallOriginalTask", "MutagenesisOriginalTask", "NCAAOriginalTask", "NCAATemporalTask",
    "NorthwindOriginalTask", "NorthwindTemporalTask", "PimaOriginalTask", "PremiereLeagueOriginalTask",
    "PremiereLeagueTemporalTask", "RestbaseOriginalTask", "SakilaOriginalTask", "SakilaTemporalTask",
    "SalesOriginalTask", "SameGenOriginalTask", "SAPOriginalTask", "SAPSalesTask", "SAPSalesTemporalTask",
    "SatelliteOriginalTask", "SeznamOriginalTask", "SeznamTemporalTask", "SFScoresOriginalTask",
    "SFScoresTemporalTask", "ShakespeareOriginalTask", "StatsOriginalTask", "StatsTemporalTask",
    "StudentLoanOriginalTask", "ThrombosisOriginalTask", "ThrombosisTemporalTask",
    "ToxicologyOriginalTask", "TPCCOriginalTask", "TPCDOriginalTask", "TPCDTemporalTask",
    "TPCDSOriginalTask", "TPCDSTemporalTask", "TPCHOriginalTask", "TPCHTemporalTask",
    "TriazineOriginalTask", "UWCSEOriginalTask", "VisualGenomeOriginalTask", "VOCOriginalTask",
    "VOCTemporalTask", "WalmartOriginalTask", "WalmartTemporalTask", "WebKPOriginalTask", "WorldOriginalTask"
]
# fmt: on
