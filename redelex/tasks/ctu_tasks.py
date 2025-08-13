import pandas as pd
from relbench.base import TaskType

from .ctu_entity_task_classic import CTUEntityTask
from .ctu_entity_task_temporal import CTUEntityTaskTemporal
from .ctu_link_task import CTULinkTask


class AccidentsOriginalTask(CTUEntityTask):
    entity_table = "nesreca"
    target_col = "klas_nesreca"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class AccidentsTemporalTask(CTUEntityTaskTemporal):
    entity_table = "nesreca"
    target_col = "klas_nesreca"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("2004-03-01")
    test_timestamp = pd.Timestamp("2005-03-01")


class AdventureWorksOriginalTask(CTUEntityTask):
    entity_table = "SalesOrderHeader"
    target_col = "TotalDue"
    task_type = TaskType.REGRESSION


class AdventureWorksTemporalTask(CTUEntityTaskTemporal):
    entity_table = "SalesOrderHeader"
    target_col = "TotalDue"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-05-01")


# TODO: remove duplicate target cols
class AirlineOriginalTask(CTUEntityTask):
    entity_table = "On_Time_On_Time_Performance_2016_1"
    target_col = "ArrDelay"
    task_type = TaskType.REGRESSION


# TODO: remove duplicate target cols
class AirlineTemporalTask(CTUEntityTaskTemporal):
    entity_table = "On_Time_On_Time_Performance_2016_1"
    target_col = "ArrDelay"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2016-01-18")
    test_timestamp = pd.Timestamp("2016-01-25")


class AtherosclerosisOriginalTask(CTUEntityTask):
    entity_table = "Entry"
    target_col = "STAV"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class BasketballMenOriginalTask(CTUEntityTask):
    entity_table = "teams"
    target_col = "rank"
    task_type = TaskType.REGRESSION


class BasketballWomenOriginalTask(CTUEntityTask):
    entity_table = "teams"
    target_col = "playoff"
    task_type = TaskType.BINARY_CLASSIFICATION


class BiodegradabilityOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    task_type = TaskType.REGRESSION


class BupaOriginalTask(CTUEntityTask):
    entity_table = "bupa"
    target_col = "arg2"
    task_type = TaskType.BINARY_CLASSIFICATION


class CarcinogenesisOriginalTask(CTUEntityTask):
    entity_table = "canc"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class CDESchoolsOriginalTask(CTUEntityTask):
    entity_table = "satscores"
    target_col = "PctGE1500"
    task_type = TaskType.REGRESSION


class ChessOriginalTask(CTUEntityTask):
    entity_table = "game"
    target_col = "game_result"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class ClassicModelsOriginalTask(CTUEntityTask):
    entity_table = "payments"
    target_col = "amount"
    task_type = TaskType.REGRESSION


class ClassicModelsTemporalTask(CTUEntityTaskTemporal):
    entity_table = "payments"
    target_col = "amount"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2004-11-01")
    test_timestamp = pd.Timestamp("2005-02-01")


class CORAOriginalTask(CTUEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CountriesOriginalTask(CTUEntityTask):
    entity_table = "target"
    target_col = "2012"
    task_type = TaskType.REGRESSION


class CraftBeerOriginalTask(CTUEntityTask):
    entity_table = "breweries"
    target_col = "state"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CreditOriginalTask(CTULinkTask):
    entity_table = "member"
    link_table = "region"
    task_type = TaskType.LINK_PREDICTION
    val_timestamp = pd.Timestamp("1999-09-01")
    test_timestamp = pd.Timestamp("1999-10-01")


class DallasOriginalTask(CTUEntityTask):
    entity_table = "incidents"
    target_col = "subject_statuses"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class DallasTemporalTask(CTUEntityTaskTemporal):
    entity_table = "incidents"
    target_col = "subject_statuses"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("2014-01-01")
    test_timestamp = pd.Timestamp("2015-01-01")


class DCGOriginalTask(CTUEntityTask):
    entity_table = "sentences"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class DiabetesOriginalTask(CTUEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class DunurOriginalTask(CTULinkTask):
    entity_table = "target"
    target_col = "is_dunur"
    task_type = TaskType.LINK_PREDICTION


class EltiOriginalTask(CTULinkTask):
    entity_table = "target"
    target_col = "is_elti"
    task_type = TaskType.LINK_PREDICTION


class EmployeeOriginalTask(CTUEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION


class EmployeeTemporalTask(CTUEntityTaskTemporal):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2000-01-01")
    test_timestamp = pd.Timestamp("2001-01-01")


class ErgastF1OriginalTask(CTUEntityTask):
    entity_table = "target"
    target_col = "win"
    task_type = TaskType.BINARY_CLASSIFICATION


class ErgastF1TemporalTask(CTUEntityTaskTemporal):
    entity_table = "target"
    target_col = "win"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("2014-01-01")
    test_timestamp = pd.Timestamp("2016-01-01")


class ExpendituresOriginalTask(CTUEntityTask):
    entity_table = "EXPENDITURES"
    target_col = "GIFT"
    task_type = TaskType.BINARY_CLASSIFICATION


class FinancialOriginalTask(CTUEntityTask):
    entity_table = "loan"
    target_col = "status"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class FinancialTemporalTask(CTUEntityTaskTemporal):
    entity_table = "loan"
    target_col = "status"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("1998-01-01")
    test_timestamp = pd.Timestamp("1998-07-01")


class FNHKOriginalTask(CTUEntityTask):
    entity_table = "pripady"
    target_col = "Delka_hospitalizace"
    task_type = TaskType.REGRESSION


class FNHKTemporalTask(CTUEntityTaskTemporal):
    entity_table = "pripady"
    target_col = "Delka_hospitalizace"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2014-09-01")
    test_timestamp = pd.Timestamp("2014-11-01")


class FTPOriginalTask(CTUEntityTask):
    entity_table = "session"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class FTPTemporalTask(CTUEntityTaskTemporal):
    entity_table = "session"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("2014-12-15")
    test_timestamp = pd.Timestamp("2014-12-19")


class GeneeaOriginalTask(CTUEntityTask):
    entity_table = "hl_hlasovani"
    target_col = "vysledek"
    task_type = TaskType.BINARY_CLASSIFICATION


class GeneeaTemporalTask(CTUEntityTaskTemporal):
    entity_table = "hl_hlasovani"
    target_col = "vysledek"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("2015-06-01")
    test_timestamp = pd.Timestamp("2015-10-01")


class GenesOriginalTask(CTUEntityTask):
    entity_table = "Classification"
    target_col = "Localization"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class GOSalesOriginalTask(CTUEntityTask):
    entity_table = "go_daily_sales"
    target_col = "Quantity"
    task_type = TaskType.REGRESSION


class GOSalesTemporalTask(CTUEntityTaskTemporal):
    entity_table = "go_daily_sales"
    target_col = "Quantity"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2017-10-01")
    test_timestamp = pd.Timestamp("2018-03-01")


class GrantsOriginalTask(CTUEntityTask):
    entity_table = "awards"
    target_col = "award_amount"
    task_type = TaskType.REGRESSION


class GrantsTemporalTask(CTUEntityTaskTemporal):
    entity_table = "awards"
    target_col = "award_amount"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2010-10-01")
    test_timestamp = pd.Timestamp("2014-01-01")


class HepatitisOriginalTask(CTUEntityTask):
    entity_table = "dispat"
    target_col = "Type"
    task_type = TaskType.BINARY_CLASSIFICATION


class HockeyOriginalTask(CTUEntityTask):
    entity_table = "Master"
    target_col = "shootCatch"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class HockeyTemporalTask(CTUEntityTaskTemporal):
    entity_table = "Master"
    target_col = "shootCatch"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("2007-01-01")
    test_timestamp = pd.Timestamp("2009-01-01")


class IMDbOriginalTask(CTUEntityTask):
    entity_table = "actors"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class IMDbTemporalTask(CTUEntityTaskTemporal):
    entity_table = "actors"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("1998-01-01")
    test_timestamp = pd.Timestamp("2002-01-01")


class LahmanOriginalTask(CTUEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION


class LahmanTemporalTask(CTUEntityTaskTemporal):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2010-01-01")
    test_timestamp = pd.Timestamp("2012-01-01")


class LegalActsOriginalTask(CTUEntityTask):
    entity_table = "legalacts"
    target_col = "ActKind"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class LegalActsTemporalTask(CTUEntityTaskTemporal):
    entity_table = "legalacts"
    target_col = "ActKind"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("2012-02-01")
    test_timestamp = pd.Timestamp("2012-05-01")


class MeshOriginalTask(CTUEntityTask):
    entity_table = "mesh"
    target_col = "num"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MondialOriginalTask(CTUEntityTask):
    entity_table = "target"
    target_col = "Target"
    task_type = TaskType.BINARY_CLASSIFICATION


class MooneyOriginalTask(CTULinkTask):
    entity_table = "uncle"
    task_type = TaskType.LINK_PREDICTION


class MovieLensOriginalTask(CTUEntityTask):
    entity_table = "users"
    target_col = "u_gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class MuskSmallOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class MuskLargeOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class MutagenesisOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "mutagenic"
    task_type = TaskType.BINARY_CLASSIFICATION


class NCAAOriginalTask(CTUEntityTask):
    entity_table = "target"
    target_col = "team_id1_wins"
    task_type = TaskType.BINARY_CLASSIFICATION


class NCAATemporalTask(CTUEntityTaskTemporal):
    entity_table = "target"
    target_col = "team_id1_wins"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("2010-11-01")
    test_timestamp = pd.Timestamp("2012-11-05")


class NorthwindOriginalTask(CTUEntityTask):
    entity_table = "Orders"
    target_col = "Freight"
    task_type = TaskType.REGRESSION


class NorthwindTemporalTask(CTUEntityTaskTemporal):
    entity_table = "Orders"
    target_col = "Freight"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("1998-02-01")
    test_timestamp = pd.Timestamp("1998-04-01")


class PimaOriginalTask(CTUEntityTask):
    entity_table = "pima"
    target_col = "arg2"
    task_type = TaskType.BINARY_CLASSIFICATION


class PremiereLeagueOriginalTask(CTUEntityTask):
    entity_table = "Matches"
    target_col = "ResultOfTeamHome"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class PremiereLeagueTemporalTask(CTUEntityTaskTemporal):
    entity_table = "Matches"
    target_col = "ResultOfTeamHome"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("2012-04-01")
    test_timestamp = pd.Timestamp("2012-05-01")


class RestbaseOriginalTask(CTUEntityTask):
    entity_table = "generalinfo"
    target_col = "review"
    task_type = TaskType.REGRESSION


class SakilaOriginalTask(CTUEntityTask):
    entity_table = "payment"
    target_col = "amount"
    task_type = TaskType.REGRESSION


class SakilaTemporalTask(CTUEntityTaskTemporal):
    entity_table = "payment"
    target_col = "amount"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2005-08-19")
    test_timestamp = pd.Timestamp("2005-08-21")


class SalesOriginalTask(CTUEntityTask):
    entity_table = "Sales"
    target_col = "Quantity"
    task_type = TaskType.REGRESSION


class SameGenOriginalTask(CTULinkTask):
    entity_table = "target"
    target_col = "target"
    task_type = TaskType.BINARY_CLASSIFICATION


class SAPOriginalTask(CTUEntityTask):
    entity_table = "Mailings"
    target_col = "RESPONSE"
    task_type = TaskType.BINARY_CLASSIFICATION


class SAPSalesTask(CTUEntityTask):
    entity_table = "Sales"
    target_col = "AMOUNT"
    task_type = TaskType.REGRESSION


class SAPSalesTemporalTask(CTUEntityTaskTemporal):
    entity_table = "Sales"
    target_col = "AMOUNT"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2007-06-10")
    test_timestamp = pd.Timestamp("2007-06-20")


class SatelliteOriginalTask(CTULinkTask):
    entity_table = "tm"
    link_table = "fault"
    task_type = TaskType.LINK_PREDICTION


class SeznamOriginalTask(CTUEntityTask):
    entity_table = "probehnuto"
    target_col = "kc_proklikano"
    task_type = TaskType.REGRESSION


class SeznamTemporalTask(CTUEntityTaskTemporal):
    entity_table = "probehnuto"
    target_col = "kc_proklikano"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")


class SFScoresOriginalTask(CTUEntityTask):
    entity_table = "inspections"
    target_col = "score"
    task_type = TaskType.REGRESSION


class SFScoresTemporalTask(CTUEntityTaskTemporal):
    entity_table = "inspections"
    target_col = "score"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2016-03-01")
    test_timestamp = pd.Timestamp("2016-07-01")


class ShakespeareOriginalTask(CTULinkTask):
    entity_table = "paragraphs"
    link_table = "characters"
    task_type = TaskType.LINK_PREDICTION


class StatsOriginalTask(CTUEntityTask):
    entity_table = "users"
    target_col = "Reputation"
    task_type = TaskType.REGRESSION


class StatsTemporalTask(CTUEntityTaskTemporal):
    entity_table = "users"
    target_col = "Reputation"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")


class StudentLoanOriginalTask(CTUEntityTask):
    entity_table = "no_payment_due"
    target_col = "bool"
    task_type = TaskType.BINARY_CLASSIFICATION


class ThrombosisOriginalTask(CTUEntityTask):
    entity_table = "Examination"
    target_col = "Thrombosis"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class ThrombosisTemporalTask(CTUEntityTaskTemporal):
    entity_table = "Examination"
    target_col = "Thrombosis"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("1996-01-01")
    test_timestamp = pd.Timestamp("1997-01-01")


class ToxicologyOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "label"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCCOriginalTask(CTUEntityTask):
    entity_table = "C_Customer"
    target_col = "c_credit"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCDOriginalTask(CTUEntityTask):
    entity_table = "dss_customer"
    target_col = "c_mktsegment"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class TPCDTemporalTask(CTUEntityTaskTemporal):
    entity_table = "dss_customer"
    target_col = "c_mktsegment"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("1997-01-01")
    test_timestamp = pd.Timestamp("1998-01-01")


class TPCDSOriginalTask(CTUEntityTask):
    entity_table = "customer"
    target_col = "c_preferred_cust_flag"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCDSTemporalTask(CTUEntityTaskTemporal):
    entity_table = "customer"
    target_col = "c_preferred_cust_flag"
    task_type = TaskType.BINARY_CLASSIFICATION
    val_timestamp = pd.Timestamp("2001-01-01")
    test_timestamp = pd.Timestamp("2002-01-01")


class TPCHOriginalTask(CTUEntityTask):
    entity_table = "customer"
    target_col = "c_acctbal"
    task_type = TaskType.REGRESSION


class TPCHTemporalTask(CTUEntityTaskTemporal):
    entity_table = "customer"
    target_col = "c_acctbal"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("1997-09-01")
    test_timestamp = pd.Timestamp("1998-03-01")


class TriazineOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    task_type = TaskType.REGRESSION


class UWCSEOriginalTask(CTUEntityTask):
    entity_table = "person"
    target_col = "inPhase"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class VisualGenomeOriginalTask(CTUEntityTask):
    entity_table = "IMG_OBJ"
    target_col = "OBJ_CLASS"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class VOCOriginalTask(CTUEntityTask):
    entity_table = "voyages"
    target_col = "arrival_harbour"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class VOCTemporalTask(CTUEntityTaskTemporal):
    entity_table = "voyages"
    target_col = "arrival_harbour"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    val_timestamp = pd.Timestamp("1887-01-01")
    test_timestamp = pd.Timestamp("1902-01-01")


class WalmartOriginalTask(CTUEntityTask):
    entity_table = "train_table"
    target_col = "units"
    task_type = TaskType.REGRESSION


class WalmartTemporalTask(CTUEntityTaskTemporal):
    entity_table = "train_table"
    target_col = "units"
    task_type = TaskType.REGRESSION
    val_timestamp = pd.Timestamp("2014-01-01")
    test_timestamp = pd.Timestamp("2014-06-01")


class WebKPOriginalTask(CTUEntityTask):
    entity_table = "webpage"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class WorldOriginalTask(CTUEntityTask):
    entity_table = "Country"
    target_col = "Continent"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


# fmt: off
__all__ = [
    "AccidentsOriginalTask", "AccidentsTemporalTask",
    "AdventureWorksOriginalTask", "AdventureWorksTemporalTask",
    "AirlineOriginalTask", "AirlineTemporalTask",
    "AtherosclerosisOriginalTask", 
    "BasketballMenOriginalTask",
    "BasketballWomenOriginalTask", 
    "BiodegradabilityOriginalTask",
    "BupaOriginalTask", 
    "CarcinogenesisOriginalTask", 
    "CDESchoolsOriginalTask", 
    "ChessOriginalTask", 
    "ClassicModelsOriginalTask", "ClassicModelsTemporalTask",
    "CORAOriginalTask", 
    "CountriesOriginalTask", 
    "CraftBeerOriginalTask", 
    "CreditOriginalTask", 
    "DallasOriginalTask", "DallasTemporalTask",
    "DCGOriginalTask", 
    "DiabetesOriginalTask",
    "DunurOriginalTask", 
    "EltiOriginalTask", 
    "EmployeeOriginalTask", "EmployeeTemporalTask",
    "ErgastF1OriginalTask", "ErgastF1TemporalTask",
    "ExpendituresOriginalTask", 
    "FinancialOriginalTask", "FinancialTemporalTask",
    "FNHKOriginalTask", "FNHKTemporalTask",
    "FTPOriginalTask", "FTPTemporalTask",
    "GeneeaOriginalTask", "GeneeaTemporalTask",
    "GenesOriginalTask", 
    "GOSalesOriginalTask", "GOSalesTemporalTask",
    "GrantsOriginalTask", "GrantsTemporalTask",
    "HepatitisOriginalTask", 
    "HockeyOriginalTask", "HockeyTemporalTask",
    "IMDbOriginalTask", "IMDbTemporalTask",
    "LahmanOriginalTask", "LahmanTemporalTask",
    "LegalActsOriginalTask", "LegalActsTemporalTask",
    "MeshOriginalTask", 
    "MondialOriginalTask", 
    "MooneyOriginalTask", 
    "MovieLensOriginalTask", 
    "MuskLargeOriginalTask", 
    "MuskSmallOriginalTask", 
    "MutagenesisOriginalTask",
    "NCAAOriginalTask", "NCAATemporalTask",
    "NorthwindOriginalTask", "NorthwindTemporalTask",
    "PimaOriginalTask", 
    "PremiereLeagueOriginalTask", "PremiereLeagueTemporalTask",
    "RestbaseOriginalTask",
    "SakilaOriginalTask", "SakilaTemporalTask",
    "SalesOriginalTask", 
    "SameGenOriginalTask", 
    "SAPOriginalTask", "SAPSalesTask", "SAPSalesTemporalTask",
    "SatelliteOriginalTask", 
    "SeznamOriginalTask", "SeznamTemporalTask",
    "SFScoresOriginalTask", "SFScoresTemporalTask",
    "ShakespeareOriginalTask", 
    "StatsOriginalTask", "StatsTemporalTask",
    "StudentLoanOriginalTask", 
    "ThrombosisOriginalTask", "ThrombosisTemporalTask",
    "ToxicologyOriginalTask",
    "TPCCOriginalTask", 
    "TPCDOriginalTask", "TPCDTemporalTask",
    "TPCDSOriginalTask", "TPCDSTemporalTask",
    "TPCHOriginalTask", "TPCHTemporalTask",
    "TriazineOriginalTask",
    "UWCSEOriginalTask", 
    "VisualGenomeOriginalTask", 
    "VOCOriginalTask", "VOCTemporalTask",
    "WalmartOriginalTask", "WalmartTemporalTask",
    "WebKPOriginalTask", 
    "WorldOriginalTask"
]
# fmt: on
